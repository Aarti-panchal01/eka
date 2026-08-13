"""
================================================================================
EKA — embedding model fine-tune (BGE-base + triplets)  |  Kaggle T4
================================================================================
FULLY SELF-CONTAINED. Nothing imported from the Eka project.

This is the model that decides which memories Eka recalls. Base BGE already
retrieves well on generic text; fine-tuning on Eka's own triplets teaches it
that "startup fear" and "business anxiety" are the same thing, while pushing
apart superficially-similar messages that belong to different personas.

BEFORE YOU RUN
--------------
1. Accelerator = GPU T4, Internet = ON
2. Secrets: HF_TOKEN, HF_USERNAME, (optional) WANDB_API_KEY
3. ml/scripts/upload_to_hf.py must already have pushed embedding_triplets.jsonl

ESTIMATED TIME ON T4
--------------------
    6000 triplets, batch 64, 3 epochs  =  ~280 steps  =  ~35-50 min
    (The build plan budgets 4-5 hrs; that was sized for a much larger corpus.
     If you generate 50k+ triplets later, expect the longer number.)
================================================================================
"""

# =============================================================================
# HOW TO ADD SECRETS IN KAGGLE  (do this before you hit Run)
# 1. Open the notebook -> Add-ons -> Secrets
# 2. Add: HF_TOKEN      = your_huggingface_write_token
# 3. Add: WANDB_API_KEY = your_wandb_key
# 4. Add: HF_USERNAME   = amijackofalltrades
# 5. Enable "Internet" in Settings (right-hand panel)
# 6. Enable "GPU T4 x1" in Settings  (Accelerator)
# 7. Enable "Background Execution"  <- CRITICAL: keeps training after you
#    close the browser. Without it the session dies when the tab closes.
#    (In the current Kaggle UI this is "Save Version -> Save & Run All (Commit)".)
# 8. Click Run All, then close the browser tab safely.
#
# Each secret must be "attached" to the notebook after adding it — Kaggle shows
# a checkbox next to each secret in the Add-ons -> Secrets panel. An unattached
# secret reads back as empty and the auth cell will exit with a clear error.
# =============================================================================

# ==============================================================================
# SECTION 1 — INSTALL
# ==============================================================================
# %%capture
# (nothing to install — see the note below)

import json
import os
import random
import subprocess  # noqa: F401  (kept: the builder's section parser expects it)
import sys  # noqa: F401

# NOTHING IS INSTALLED HERE, DELIBERATELY.
#
# Probed on the live image 2026-08-14: sentence-transformers 5.4.1, datasets
# 5.0.0, huggingface-hub 1.11.0 and torch 2.10.0+cu128 are all preinstalled.
# The old pins (sentence-transformers==3.0.1, datasets==2.19.1) dragged a
# mid-2024 tree backwards over a stack built for the current one, which is
# exactly what cost the persona runs two failed sessions — a NumPy ABI split
# and then `No module named 'triton.ops'`.
#
# wandb is gone too: NumPy 2 makes it unusable on this image, so it is a
# credential and a dependency that buy nothing.
if os.environ.get("EKA_SKIP_INSTALL") == "never":  # pragma: no cover
    pass


# ==============================================================================
# SECTION 2 — AUTH
# ==============================================================================
SECRETS_DATASET = "/kaggle/input/eka-secrets/secrets.json"


def _load_secrets() -> dict:
    """Credentials from the attached dataset first, then Kaggle Secrets, then env.

    `kaggle kernels push` cannot attach Kaggle Secrets and DETACHES them from a
    notebook that had them, so an API-pushed notebook used to die here until
    someone ticked two boxes in a browser. `dataset_sources` IS honoured by
    push, which is what makes an unattended launch possible.
    """
    names = ["HF_TOKEN", "HF_USERNAME", "WANDB_API_KEY"]
    found = {n: "" for n in names}

    try:
        with open(SECRETS_DATASET, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        for name in names:
            if blob.get(name):
                found[name] = str(blob[name]).strip()
        print(f"\u2713 secrets from attached dataset ({SECRETS_DATASET})")
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"! could not read {SECRETS_DATASET}: {type(exc).__name__}: {exc}")

    if not all(found[n] for n in ("HF_TOKEN", "HF_USERNAME")):
        try:
            from kaggle_secrets import UserSecretsClient

            client = UserSecretsClient()
            for name in names:
                if not found[name]:
                    try:
                        found[name] = client.get_secret(name)
                    except Exception:
                        pass
        except Exception:
            pass

    for name in names:
        if not found[name]:
            found[name] = os.environ.get(name, "")
    for name, value in found.items():
        if value:
            os.environ[name] = value

    missing = [n for n in ("HF_TOKEN", "HF_USERNAME") if not found.get(n)]
    if missing:
        raise SystemExit(
            f"Missing secret(s): {', '.join(missing)}. Attach the eka-secrets "
            f"dataset, or add them under Add-ons -> Secrets."
        )
    return found


SECRETS = _load_secrets()

# One GPU only. Kaggle's "GPU T4 x2" is genuinely two cards and HF Trainer uses
# every visible one via nn.DataParallel, which split tensors across cuda:0 and
# cuda:1 and killed the founder run at its first step. Set before torch loads.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from huggingface_hub import HfApi, hf_hub_download, login  # noqa: E402

login(token=SECRETS["HF_TOKEN"])
print("✓ Hugging Face authenticated")

USE_WANDB = bool(SECRETS.get("WANDB_API_KEY"))
if USE_WANDB:
    import wandb

    wandb.login(key=SECRETS["WANDB_API_KEY"])
    wandb.init(project="eka", name="eka-embeddings-v1")
    print("✓ WandB authenticated")


# ==============================================================================
# SECTION 3 — CONFIG
# ==============================================================================
HF_USERNAME = os.environ["HF_USERNAME"]
BASE_MODEL = "BAAI/bge-base-en-v1.5"  # 110M params, 768-dim
DATASET_REPO = f"{HF_USERNAME}/eka-datasets"
OUTPUT_REPO = f"{HF_USERNAME}/eka-embeddings"
OUTPUT_DIR = "/kaggle/working/eka-embeddings"

BATCH_SIZE = 64
EPOCHS = 3
LR = 2e-5
WARMUP_STEPS = 50
EVAL_STEPS = 200
VAL_FRACTION = 0.10

random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'=' * 70}")
print(f"  EKA EMBEDDINGS")
print(f"  base   : {BASE_MODEL}")
print(f"  data   : {DATASET_REPO}/embedding_triplets.jsonl")
print(f"  output : {OUTPUT_REPO}")
print(f"{'=' * 70}\n")

import torch  # noqa: E402

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (slow)'}")


# ==============================================================================
# SECTION 4 — LOAD TRIPLETS
# ==============================================================================
triplet_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="embedding_triplets.jsonl",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"],
)

from sentence_transformers import InputExample  # noqa: E402

examples = []
skipped = 0
with open(triplet_path, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        anchor = (row.get("anchor") or "").strip()
        positive = (row.get("positive") or "").strip()
        negative = (row.get("negative") or "").strip()
        # A triplet where the positive equals the anchor teaches nothing and
        # drags the loss to zero; drop those rather than train on them.
        if not (anchor and positive and negative) or anchor == positive:
            skipped += 1
            continue
        examples.append(InputExample(texts=[anchor, positive, negative]))

print(f"\nLoaded {len(examples)} triplets ({skipped} skipped)")
if len(examples) < 100:
    raise SystemExit(
        "Too few triplets to train on. Run:\n"
        "  python ml/scripts/generate_embedding_triplets.py\n"
        "  python ml/scripts/upload_to_hf.py"
    )

random.shuffle(examples)
split = int(len(examples) * (1 - VAL_FRACTION))
train_examples, val_examples = examples[:split], examples[split:]
print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")


# ==============================================================================
# SECTION 5 — TRAIN
# ==============================================================================
from sentence_transformers import SentenceTransformer, losses  # noqa: E402
from sentence_transformers.evaluation import TripletEvaluator  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

model = SentenceTransformer(BASE_MODEL)
print(f"\nEmbedding dim: {model.get_sentence_embedding_dimension()} (backend expects 768)")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
# MultipleNegativesRankingLoss treats every other in-batch example as an extra
# negative, so a batch of 64 gives 63 free negatives per anchor. This is why
# batch size matters more than epoch count here.
train_loss = losses.MultipleNegativesRankingLoss(model)
evaluator = TripletEvaluator.from_input_examples(val_examples, name="eka-val")

print("\nBaseline (before fine-tuning):")
baseline = evaluator(model)
print(f"  {baseline}")

steps_per_epoch = max(1, len(train_dataloader))
print(f"\nTraining: {EPOCHS} epochs x {steps_per_epoch} steps = "
      f"{EPOCHS * steps_per_epoch} total\n")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=EPOCHS,
    evaluation_steps=EVAL_STEPS,
    warmup_steps=WARMUP_STEPS,
    output_path=OUTPUT_DIR,
    optimizer_params={"lr": LR},
    show_progress_bar=True,
    use_amp=True,
)

print("\nAfter fine-tuning:")
final = evaluator(model)
print(f"  {final}")


# ==============================================================================
# SECTION 6 — SANITY CHECKS
# These matter more than the triplet accuracy number. If the pairs below don't
# separate, the model will recall the wrong memories no matter what loss says.
# ==============================================================================
from sentence_transformers import util  # noqa: E402

CHECKS = [
    ("startup fear", "business anxiety", "should be HIGH", 0.80, "high"),
    ("startup fear", "what is pizza", "should be LOW", 0.30, "low"),
    ("I keep procrastinating on the launch",
     "I can't get myself to ship the product", "should be HIGH", 0.75, "high"),
    ("my co-founder is hiding revenue",
     "my partner is not being transparent about money", "should be HIGH", 0.75, "high"),
    ("what is my dharma", "how do I price my SaaS", "should be LOW", 0.45, "low"),
]

print(f"\n{'=' * 70}")
print("  SEMANTIC SANITY CHECKS")
print(f"{'=' * 70}")
passed = 0
for text_a, text_b, note, threshold, direction in CHECKS:
    vectors = model.encode([text_a, text_b], convert_to_tensor=True,
                           normalize_embeddings=True)
    score = float(util.cos_sim(vectors[0], vectors[1]))
    ok = score >= threshold if direction == "high" else score <= threshold
    passed += ok
    print(f"  {'✓' if ok else '✗'} {score:.3f}  ({note} {'≥' if direction == 'high' else '≤'} "
          f"{threshold})\n      '{text_a}'  vs  '{text_b}'")
print(f"\n  {passed}/{len(CHECKS)} sanity checks passed")
print(f"{'=' * 70}\n")


# ==============================================================================
# SECTION 7 — PUSH TO HUB
# ==============================================================================
model.save(OUTPUT_DIR)

card = f"""---
base_model: {BASE_MODEL}
library_name: sentence-transformers
pipeline_tag: feature-extraction
tags:
- eka
- sentence-similarity
- retrieval
---

# eka-embeddings

Retrieval model for **Eka**'s semantic memory. Fine-tuned from
`{BASE_MODEL}` on {len(examples)} (anchor, positive, negative) triplets drawn
from Eka's four persona datasets.

| | |
|---|---|
| dim | {model.get_sentence_embedding_dimension()} |
| loss | MultipleNegativesRankingLoss |
| batch / epochs / lr | {BATCH_SIZE} / {EPOCHS} / {LR} |
| train / val triplets | {len(train_examples)} / {len(val_examples)} |
| sanity checks | {passed}/{len(CHECKS)} |

Negatives are user messages from a *different* persona mode, so the model
learns to separate "what is my dharma" from "how do I price my SaaS".

Used by `backend/services/embedding_service.py` against a 768-dim Qdrant
collection.
"""
with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as handle:
    handle.write(card)

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=OUTPUT_REPO, private=True, exist_ok=True)
api.upload_folder(folder_path=OUTPUT_DIR, repo_id=OUTPUT_REPO,
                  ignore_patterns=["eval/*", "checkpoint-*"])

print(f"✅ Pushed to HF Hub: https://huggingface.co/{OUTPUT_REPO}")

if USE_WANDB:
    wandb.finish()

print("\n✅ EMBEDDINGS DONE.")
print("   The backend picks this up automatically once HF_USERNAME is set.")
