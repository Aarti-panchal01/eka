"""
================================================================================
EKA — FOUNDER persona QLoRA fine-tune  |  Kaggle T4 (single GPU)
================================================================================
FULLY SELF-CONTAINED. Imports nothing from the Eka project. Paste into a Kaggle
notebook (or upload as a script) and run top to bottom.

BEFORE YOU RUN
--------------
1. Kaggle -> Settings -> Accelerator = GPU T4 x1 (7B in 4-bit fits comfortably)
2. Kaggle -> Settings -> Internet = ON
3. Kaggle -> Add-ons -> Secrets, add all three:
       HF_TOKEN        (write permission)
       WANDB_API_KEY
       HF_USERNAME
4. No license to accept. Qwen/Qwen2.5-7B-Instruct is ungated and starts
   downloading immediately. (The base model was
   meta-llama/Meta-Llama-3-8B-Instruct until 2026-08-13; Meta approval for
   Llama 3.1/3.2 was still pending, so this moved to Qwen.)
5. Kaggle -> Save Version -> "Save & Run All (Commit)" so it keeps training
   after you close the browser. Kaggle sessions cap at 12h; this run needs
   far less, but the checkpoint-resume logic below survives a restart anyway.

ESTIMATED TIME ON A SINGLE T4
-----------------------------
    ~900 train examples, effective batch 16, 3 epochs  =  ~170 optimizer steps
    ~55-70 s per optimizer step at seq len 2048        =  ~2.5-3.5 hrs
    plus ~15 min model download on first run
If you raise MAX_SEQ_LEN or EPOCHS, scale that estimate linearly.

TO TRAIN A DIFFERENT PERSONA
----------------------------
Change MODE below to "chanakya" / "gita" / "reflection". Nothing else changes.
(The other three files in training/ are exactly this, with MODE pre-set.)
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
# In a notebook, put this in the first cell prefixed with %%capture
# ==============================================================================
# %%capture
# !pip install -q "numpy<2.0" transformers==4.41.0 peft==0.11.1 trl==0.8.6 \
#     datasets==2.19.1 accelerate==0.30.0 bitsandbytes==0.43.1 \
#     wandb==0.16.6 huggingface-hub==0.23.2

import os
import subprocess
import sys


def _pip_install() -> None:
    """Idempotent install so the script works as a plain .py run too."""
    packages = [
        # numpy<2 leads the list because it is a constraint the rest have to
        # resolve against, not a preference. Kaggle's image ships NumPy 2.x on
        # Python 3.12; wandb 0.17.0 still reaches for np.float_, removed in
        # NumPy 2.0, and the resulting AttributeError killed the founder run on
        # 2026-08-13 before a single training step.
        #
        # It must be in THIS pip call, not an earlier separate one — installing
        # the packages below afterwards would pull NumPy 2 straight back up.
        # Everything else here is pinned to mid-2024 releases built against
        # NumPy 1.x anyway, so this is the combination they expect.
        "numpy<2.0",
        "transformers==4.41.0",
        "peft==0.11.1",
        "trl==0.8.6",
        "datasets==2.19.1",
        "accelerate==0.30.0",
        "bitsandbytes==0.43.1",
        # Belt and braces. The numpy pin above is what actually guarantees
        # np.float_ exists; this keeps wandb on a release from the same era
        # as everything else.
        "wandb==0.16.6",
        "huggingface-hub==0.23.2",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages], check=False
    )


if os.environ.get("EKA_SKIP_INSTALL") != "1":
    _pip_install()

# Report which NumPy actually won, in 30 seconds, rather than finding out from
# a traceback 15 minutes in after the model download. A downgrade only affects
# modules not yet imported in THIS process, so if anything pulled NumPy 2 in
# before this ran, pip can report success and the import below still be 2.x.
#
# This warns rather than exits. Under NumPy 2 the only thing here that actually
# breaks is wandb (np.float_), which is optional and which SECTION 2 disables
# on its own — so stopping the run would cost a GPU session to protect a
# progress chart. Nothing else in the pinned set touches the removed aliases.
import numpy as _np  # noqa: E402

NUMPY_2 = int(_np.__version__.split(".")[0]) >= 2
if NUMPY_2:
    print(
        f"! numpy {_np.__version__} is live — the <2.0 pin did not take effect.\n"
        "  Either the install cell failed or numpy was imported before it.\n"
        "  Continuing without experiment tracking; training is unaffected.\n"
        "\n"
        "  Do NOT restart the kernel to force the downgrade. A restart inside a\n"
        "  'Save & Run All (Commit)' run ends the session, and on the way back\n"
        "  up this cell would simply restart it again."
    )
else:
    print(f"✓ numpy {_np.__version__}")


# ==============================================================================
# SECTION 2 — AUTH (Kaggle Secrets, with plain-env fallback)
# ==============================================================================
def _load_secrets() -> dict:
    """Read HF_TOKEN / WANDB_API_KEY / HF_USERNAME from Kaggle Secrets or env."""
    names = ["HF_TOKEN", "WANDB_API_KEY", "HF_USERNAME"]
    found = {}
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for name in names:
            try:
                found[name] = client.get_secret(name)
            except Exception:
                found[name] = os.environ.get(name, "")
    except Exception:
        for name in names:
            found[name] = os.environ.get(name, "")

    for name, value in found.items():
        if value:
            os.environ[name] = value

    missing = [n for n in ("HF_TOKEN", "HF_USERNAME") if not found.get(n)]
    if missing:
        raise SystemExit(
            f"Missing required secret(s): {', '.join(missing)}\n"
            "Add them under Kaggle -> Add-ons -> Secrets, then restart the session."
        )
    return found


SECRETS = _load_secrets()

from huggingface_hub import HfApi, login  # noqa: E402

login(token=SECRETS["HF_TOKEN"])
print("✓ Hugging Face authenticated")

# Experiment tracking is optional and must never be able to end a 3-hour run.
# On 2026-08-13 it did exactly that: wandb reached for np.float_ under NumPy 2
# and the AttributeError propagated straight out of the import, killing founder
# before step 1. The pins above are the fix; this is the seatbelt for the next
# incompatibility, which will not announce itself either. Losing the charts is
# an annoyance; losing the session is 3 hours of a 30 h weekly quota.
USE_WANDB = False
if not SECRETS.get("WANDB_API_KEY"):
    print("! WANDB_API_KEY not set — training without experiment tracking")
elif NUMPY_2:
    # Known-bad rather than discovered-bad: wandb reaches for np.float_, which
    # NumPy 2.0 removed. Skipping the import beats catching its traceback.
    print(
        "! skipping wandb — numpy 2 is live in this session and wandb still "
        "uses np.float_. Training runs, without experiment tracking."
    )
else:
    try:
        import wandb

        wandb.login(key=SECRETS["WANDB_API_KEY"])
        USE_WANDB = True
        print("✓ WandB authenticated")
    except Exception as exc:
        print(
            f"! WandB unavailable ({type(exc).__name__}: {exc}) — "
            f"training without experiment tracking"
        )


# ==============================================================================
# SECTION 3 — CONFIG
# ==============================================================================
MODE = "founder"  # <-- the ONLY line to change for the other three personas

HF_USERNAME = os.environ["HF_USERNAME"]
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATASET_REPO = f"{HF_USERNAME}/eka-datasets"
OUTPUT_REPO = f"{HF_USERNAME}/eka-{MODE}-qwen"
OUTPUT_DIR = f"/kaggle/working/{MODE}_lora"

MAX_SEQ_LEN = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8  # effective batch = 16
LR = 2e-4
SAVE_STEPS = 50
EVAL_STEPS = 50
WANDB_PROJECT = "eka"
RUN_NAME = f"eka-{MODE}-qwen-v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
if USE_WANDB:
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

print(f"\n{'=' * 70}")
print(f"  EKA {MODE.upper()} LoRA")
print(f"  base    : {BASE_MODEL}")
print(f"  data    : {DATASET_REPO}  ({MODE}_train.jsonl / {MODE}_val.jsonl)")
print(f"  output  : {OUTPUT_REPO}")
print(f"{'=' * 70}\n")

import torch  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit(
        "No GPU detected. Kaggle -> Settings -> Accelerator -> GPU T4 x1.\n"
        "(4-bit QLoRA on a 7B model is not viable on CPU.)"
    )

GPU_NAME = torch.cuda.get_device_name(0)
# Turing (T4) has no bfloat16 support. Ampere+ (A100/L4) does. Picking the
# wrong one here is the most common cause of "RuntimeError: expected scalar
# type" or silent NaN losses on Kaggle.
SUPPORTS_BF16 = torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if SUPPORTS_BF16 else torch.float16
print(f"GPU: {GPU_NAME}")
print(f"     bf16 supported: {SUPPORTS_BF16} -> compute dtype {COMPUTE_DTYPE}")
print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")


# ==============================================================================
# SECTION 4 — CHECKPOINT RESUME
# Kaggle sessions die. This makes a restart cost minutes, not hours.
# ==============================================================================
import glob  # noqa: E402


def find_latest_checkpoint(directory: str):
    checkpoints = glob.glob(os.path.join(directory, "checkpoint-*"))
    checkpoints = [c for c in checkpoints if os.path.isdir(c)]
    if not checkpoints:
        return None

    def step_of(path: str) -> int:
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return -1

    return max(checkpoints, key=step_of)


RESUME_FROM = find_latest_checkpoint(OUTPUT_DIR)
if RESUME_FROM:
    print(f"↻ Resuming from {RESUME_FROM}")
else:
    print("→ Starting fresh training")


# ==============================================================================
# SECTION 5 — LOAD DATA
# ==============================================================================
from datasets import load_dataset  # noqa: E402

dataset = load_dataset(
    DATASET_REPO,
    data_files={
        "train": f"{MODE}_train.jsonl",
        "validation": f"{MODE}_val.jsonl",
    },
    token=os.environ["HF_TOKEN"],
)
print(f"\nTrain: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

if "text" not in dataset["train"].column_names:
    raise SystemExit(
        f"Expected a 'text' column, got {dataset['train'].column_names}.\n"
        "Re-run ml/scripts/preprocess.py then ml/scripts/upload_to_hf.py."
    )

print("\n--- one training example (truncated) ---")
print(dataset["train"][0]["text"][:600])
print("---------------------------------------\n")


# ==============================================================================
# SECTION 6 — LOAD MODEL (4-bit NF4 quantization)
# ==============================================================================
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ["HF_TOKEN"])

# Do NOT set pad_token = eos_token here. On Llama-3 that was harmless, because
# its eos (<|end_of_text|>) is a different token from the one the template ends
# turns with (<|eot_id|>). On Qwen2.5 they are the same token: eos IS <|im_end|>.
# The SFT collator masks pad positions out of the labels, so pad == eos would
# mask every stop token the model is supposed to be learning, and the adapter
# would never stop generating. Qwen ships a separate <|endoftext|> for padding.
if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    if "<|endoftext|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|endoftext|>"
    else:
        print("! no distinct pad token found — stop tokens may be masked in labels")
print(f"     pad={tokenizer.pad_token!r} ({tokenizer.pad_token_id})  "
      f"eos={tokenizer.eos_token!r} ({tokenizer.eos_token_id})")
tokenizer.padding_side = "right"  # left padding corrupts causal LM training

print("Loading base model in 4-bit (first run downloads ~15GB)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},  # pin to GPU 0; "auto" can spill layers to CPU on T4
    trust_remote_code=True,
    token=os.environ["HF_TOKEN"],
)
model.config.use_cache = False  # incompatible with gradient checkpointing
model.config.pretraining_tp = 1
print("✓ Base model loaded")


# ==============================================================================
# SECTION 7 — LORA
# ==============================================================================
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    # All attention + MLP projections. Attention-only (q,v) trains faster but
    # learns persona style noticeably worse.
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # expect ~40M trainable / ~0.5% of 7B


# ==============================================================================
# SECTION 8 — TRAIN
# ==============================================================================
from trl import SFTTrainer  # noqa: E402

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    warmup_ratio=0.03,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    weight_decay=0.001,
    # Match the quantization compute dtype or loss goes NaN on T4.
    fp16=not SUPPORTS_BF16,
    bf16=SUPPORTS_BF16,
    optim="paged_adamw_8bit",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,  # Kaggle /kaggle/working is capped at 20GB
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb" if USE_WANDB else "none",
    run_name=RUN_NAME,
    seed=42,
    group_by_length=True,  # big speedup: batches similar-length sequences
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,  # persona examples are short; packing blurs turn boundaries
)

print(f"\n{'=' * 70}")
print(f"  TRAINING START — {RUN_NAME}")
print(f"  ~{len(dataset['train']) * EPOCHS // (BATCH_SIZE * GRAD_ACCUM)} optimizer steps")
print(f"{'=' * 70}\n")

trainer.train(resume_from_checkpoint=RESUME_FROM)

metrics = trainer.evaluate()
print(f"\n✓ Final eval loss: {metrics.get('eval_loss'):.4f}")


# ==============================================================================
# SECTION 9 — SAVE + PUSH TO HUB
# ==============================================================================
print("\nSaving adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

card = f"""---
base_model: {BASE_MODEL}
library_name: peft
tags:
- eka
- lora
- qlora
- {MODE}
---

# eka-{MODE}-qwen

QLoRA adapter giving Qwen2.5-7B-Instruct Eka's **{MODE}** persona.

| | |
|---|---|
| base | `{BASE_MODEL}` |
| rank / alpha | {LORA_R} / {LORA_ALPHA} |
| epochs | {EPOCHS} |
| effective batch | {BATCH_SIZE * GRAD_ACCUM} |
| lr / schedule | {LR} cosine |
| max seq len | {MAX_SEQ_LEN} |
| train / val | {len(dataset['train'])} / {len(dataset['validation'])} |
| final eval loss | {metrics.get('eval_loss', float('nan')):.4f} |
| trained on | {GPU_NAME} |

Merge and serve with `ml/scripts/merge_lora.py --mode {MODE}`.
"""
with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as handle:
    handle.write(card)

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=OUTPUT_REPO, private=True, exist_ok=True)
api.upload_folder(
    folder_path=OUTPUT_DIR,
    repo_id=OUTPUT_REPO,
    # Checkpoints are large and already superseded by the final adapter.
    ignore_patterns=["checkpoint-*", "*.pt", "runs/*"],
)
print(f"\n✅ Pushed to HF Hub: https://huggingface.co/{OUTPUT_REPO}")


# ==============================================================================
# SECTION 10 — SANITY GENERATION
# Does it actually sound like the persona? Read the output, don't trust the loss.
# ==============================================================================
PROBES = {
    "founder": "I have 3 paying customers at 2000/month and 4 months of runway. Should I raise?",
    "chanakya": "My business partner is hiding revenue numbers from me.",
    "gita": "I did everything right and still lost. What was the point?",
    "reflection": "I keep quitting things right before they start working.",
}

model.eval()
prompt = (
    "<|im_start|>user\n"
    f"{PROBES[MODE]}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

print(f"\n{'=' * 70}")
print(f"  SANITY CHECK — {MODE}")
print(f"{'=' * 70}")
print(f"USER: {PROBES[MODE]}\n")
print("EKA :", tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True))
print(f"{'=' * 70}\n")

if USE_WANDB:
    # The adapter is already pushed by this point, so a failure here costs
    # nothing real — but an unhandled one would still mark the whole notebook
    # failed and send you hunting through a successful 3-hour run.
    try:
        wandb.finish()
    except Exception as exc:
        print(f"! wandb.finish() failed ({type(exc).__name__}: {exc}) — ignoring")

print(f"✅ {MODE} DONE. Next: train the remaining personas, then")
print(f"   python ml/scripts/merge_lora.py --mode {MODE}")
