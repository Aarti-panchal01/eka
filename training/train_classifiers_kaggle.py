"""
================================================================================
EKA — three small models in one Kaggle session  |  T4
================================================================================
FULLY SELF-CONTAINED. Nothing imported from the Eka project.

    PART 1  complexity  DistilBERT      4-class   -> how much context to retrieve
    PART 2  sentiment   DistilRoBERTa   6-class   -> mood trend for daily insights
    PART 3  summarizer  T5-small        seq2seq   -> daily conversation summaries

Three models in one session because each one is small and the GPU is otherwise
idle between them. GPU memory is explicitly freed between parts — skipping that
is what makes part 3 OOM.

BEFORE YOU RUN
--------------
1. Accelerator = GPU T4, Internet = ON
2. Secrets: HF_TOKEN, HF_USERNAME, (optional) WANDB_API_KEY
3. complexity_labeled.jsonl must already be on the Hub
   (ml/scripts/generate_complexity_data.py -> upload_to_hf.py)

ESTIMATED TIME ON T4
--------------------
    part 1  ~15-25 min   (2000 examples, 5 epochs, tiny model)
    part 2  ~60-90 min   (~43k GoEmotions rows, 3 epochs)
    part 3  ~45-70 min   (~15k SAMSum dialogues, 3 epochs)
    total   ~2-3 hrs     (the build plan's 6-7 hrs is a safe upper bound)

You can run the parts independently: set EKA_PARTS="1", "2", "3", or "1,3".
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
# !pip install -q evaluate

import gc
import json
import os
import subprocess
import sys

# ONLY `evaluate` IS MISSING. Probed on the live image 2026-08-14:
# transformers 5.0.0, datasets 5.0.0, accelerate 1.13.0, scikit-learn 1.6.1,
# rouge-score 0.1.2 and huggingface-hub 1.11.0 all ship with it; `evaluate`
# does not and installs cleanly (0.4.6).
#
# Everything else is unpinned on purpose. Pinning a mid-2024 stack onto this
# image is what cost the persona runs two sessions — a NumPy ABI split, then
# `No module named 'triton.ops'` from a bitsandbytes built against Triton 2.
if os.environ.get("EKA_SKIP_INSTALL") != "1":
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "evaluate"],
        check=False,
    )


# ==============================================================================
# SECTION 2 — AUTH + SHARED HELPERS
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download, login  # noqa: E402

login(token=SECRETS["HF_TOKEN"])
HF_USERNAME = os.environ["HF_USERNAME"]
HF_TOKEN = os.environ["HF_TOKEN"]
DATASET_REPO = f"{HF_USERNAME}/eka-datasets"
API = HfApi(token=HF_TOKEN)

HAS_GPU = torch.cuda.is_available()
USE_FP16 = HAS_GPU
print(f"✓ Authenticated as {HF_USERNAME}")
print(f"GPU: {torch.cuda.get_device_name(0) if HAS_GPU else 'CPU (will be very slow)'}")

PARTS = {p.strip() for p in os.environ.get("EKA_PARTS", "1,2,3").split(",") if p.strip()}
print(f"Running parts: {sorted(PARTS)}\n")


def free_gpu(*objects) -> None:
    """Between-parts cleanup. Without this, part 3 OOMs on a 16GB T4."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  GPU freed — {torch.cuda.memory_allocated() / 1e9:.2f} GB still allocated")


def push(local_dir: str, repo_name: str, card: str) -> None:
    repo_id = f"{HF_USERNAME}/{repo_name}"
    with open(os.path.join(local_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(card)
    API.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    API.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        ignore_patterns=["checkpoint-*", "runs/*", "*.pt"],
    )
    print(f"✅ Pushed https://huggingface.co/{repo_id}")


from transformers import (  # noqa: E402
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

RESULTS = {}


# ==============================================================================
# PART 1 — COMPLEXITY CLASSIFIER (DistilBERT, 4-class)
# This one is load-bearing: it decides retrieval depth for every message.
# ==============================================================================
def part1_complexity():
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    print(f"\n{'=' * 70}\n  PART 1 — COMPLEXITY (DistilBERT, 4-class)\n{'=' * 70}")

    BASE = "distilbert-base-uncased"
    OUT = "/kaggle/working/eka-complexity"
    LABELS = ["simple", "normal", "complex", "deep"]
    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="complexity_labeled.jsonl",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("label") in label2id and row.get("query"):
                rows.append({"text": row["query"], "label": label2id[row["label"]]})
    print(f"Loaded {len(rows)} labelled queries")
    for label, idx in label2id.items():
        print(f"  {label:<9} {sum(1 for r in rows if r['label'] == idx)}")

    dataset = Dataset.from_list(rows).train_test_split(test_size=0.15, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(BASE)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=5,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        # warmup_steps, not warmup_ratio: Kaggle's transformers 5.0 only
        # DEPRECATED the ratio form, but a newer transformers removed it
        # outright and raises TypeError. warmup_steps works on both.
        warmup_steps=27,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=USE_FP16,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    final = trainer.evaluate()
    accuracy = final["eval_accuracy"]
    print(f"\n  accuracy {accuracy:.4f}  (target >0.85)  f1_macro {final['eval_f1_macro']:.4f}")

    predictions = np.argmax(trainer.predict(tokenized["test"]).predictions, axis=-1)
    print("\n" + classification_report(
        tokenized["test"]["label"], predictions, target_names=LABELS, digits=3
    ))

    # Read this, not just the accuracy: routing a "deep" message as "simple"
    # is the failure that makes Eka feel like it forgot you.
    for probe in [
        "hi",
        "I'm anxious about the funding round",
        "I've been running the company alone for two years and my health is slipping. "
        "How do I balance growth with my family?",
        "I realize I always quit right before things work. It happened with my first "
        "job and again with the startup. I think it's connected to how my father "
        "handled disappointment. How do I break this?",
    ]:
        inputs = tokenizer(probe, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
        idx = int(probabilities.argmax())
        print(f"  -> {id2label[idx]:<8} ({probabilities[idx]:.2f})  "
              f"\"{probe[:60]}{'...' if len(probe) > 60 else ''}\"")

    trainer.save_model(OUT)
    tokenizer.save_pretrained(OUT)
    push(OUT, "eka-complexity", f"""---
base_model: {BASE}
pipeline_tag: text-classification
tags: [eka, routing]
---

# eka-complexity

4-class router for **Eka**. Decides how much context the RAG pipeline retrieves
for an incoming message.

| label | memories | history | candidate pool |
|---|---|---|---|
| simple | 1 | 2 | 10 |
| normal | 2 | 3 | 15 |
| complex | 3 | 5 | 25 |
| deep | 5 | 7 | 40 |

Trained on {len(rows)} template-generated examples. Val accuracy **{accuracy:.3f}**,
macro F1 {final['eval_f1_macro']:.3f}.
""")
    RESULTS["complexity"] = f"accuracy {accuracy:.3f}"
    free_gpu(model, trainer)


# ==============================================================================
# PART 2 — SENTIMENT CLASSIFIER (DistilRoBERTa, 27 GoEmotions -> 6 Eka classes)
# ==============================================================================
def part2_sentiment():
    from datasets import load_dataset
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    print(f"\n{'=' * 70}\n  PART 2 — SENTIMENT (DistilRoBERTa, 6-class)\n{'=' * 70}")

    BASE = "distilroberta-base"
    OUT = "/kaggle/working/eka-sentiment"
    LABELS = ["positive", "neutral", "negative", "reflective", "anxious", "motivated"]
    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    # GoEmotions' 27 fine emotions collapsed into the 6 Eka tracks.
    COLLAPSE = {
        "positive": ["admiration", "amusement", "approval", "excitement", "joy",
                     "love", "optimism", "pride", "relief", "gratitude"],
        "neutral": ["neutral", "caring", "curiosity"],
        "negative": ["anger", "annoyance", "disappointment", "disgust",
                     "embarrassment", "grief", "remorse", "sadness"],
        "reflective": ["confusion", "realization", "surprise"],
        "anxious": ["fear", "nervousness"],
        "motivated": ["desire"],
    }

    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    source_names = raw["train"].features["labels"].feature.names
    print(f"GoEmotions: {len(source_names)} source emotions -> {len(LABELS)} Eka classes")

    fine_to_coarse = {}
    for coarse, fine_list in COLLAPSE.items():
        for fine in fine_list:
            if fine in source_names:
                fine_to_coarse[source_names.index(fine)] = label2id[coarse]
    unmapped = [n for i, n in enumerate(source_names) if i not in fine_to_coarse]
    if unmapped:
        print(f"  unmapped (dropped): {unmapped}")

    def remap(batch):
        texts, labels = [], []
        for text, label_ids in zip(batch["text"], batch["labels"]):
            # GoEmotions is multi-label; take the first emotion we can map.
            mapped = [fine_to_coarse[i] for i in label_ids if i in fine_to_coarse]
            if mapped:
                texts.append(text)
                labels.append(mapped[0])
        return {"text": texts, "label": labels}

    dataset = raw.map(
        remap, batched=True, remove_columns=raw["train"].column_names
    ).filter(lambda row: row["label"] is not None)
    print(f"  train {len(dataset['train'])} | val {len(dataset['validation'])} "
          f"| test {len(dataset['test'])}")
    counts = np.bincount(dataset["train"]["label"], minlength=len(LABELS))
    for label, count in zip(LABELS, counts):
        print(f"  {label:<11} {count}")

    tokenizer = AutoTokenizer.from_pretrained(BASE)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }

    args = TrainingArguments(
        output_dir=OUT,
        # 3 epochs, not 5: GoEmotions is 40x bigger than the complexity set and
        # distilroberta starts overfitting after 3.
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        # warmup_steps, not warmup_ratio: Kaggle's transformers 5.0 only
        # DEPRECATED the ratio form, but a newer transformers removed it
        # outright and raises TypeError. warmup_steps works on both.
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        fp16=USE_FP16,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    final = trainer.evaluate(tokenized["test"])
    accuracy = final["eval_accuracy"]
    print(f"\n  test accuracy {accuracy:.4f} (target >0.72) "
          f"| f1_weighted {final['eval_f1_weighted']:.4f}")

    predictions = np.argmax(trainer.predict(tokenized["test"]).predictions, axis=-1)
    truth = tokenized["test"]["label"]
    print("\n" + classification_report(truth, predictions, target_names=LABELS, digits=3))
    print("confusion matrix (rows = true, cols = predicted):")
    print(f"  {'':<12}" + "".join(f"{l[:6]:>8}" for l in LABELS))
    for label, row in zip(LABELS, confusion_matrix(truth, predictions)):
        print(f"  {label:<12}" + "".join(f"{v:>8}" for v in row))

    trainer.save_model(OUT)
    tokenizer.save_pretrained(OUT)
    push(OUT, "eka-sentiment", f"""---
base_model: {BASE}
pipeline_tag: text-classification
tags: [eka, emotion]
---

# eka-sentiment

6-class emotion classifier for **Eka**'s daily insights and mood trends.
GoEmotions' 27 emotions collapsed to: {", ".join(LABELS)}.

Test accuracy **{accuracy:.3f}**, weighted F1 {final['eval_f1_weighted']:.3f}.
6-class emotion is genuinely hard; >0.72 was the target.
""")
    RESULTS["sentiment"] = f"accuracy {accuracy:.3f}"
    free_gpu(model, trainer)


# ==============================================================================
# PART 3 — SUMMARIZER (T5-small on SAMSum: dialogue -> summary)
# ==============================================================================
def part3_summarizer():
    import evaluate
    from datasets import load_dataset

    print(f"\n{'=' * 70}\n  PART 3 — SUMMARIZER (T5-small, SAMSum)\n{'=' * 70}")

    BASE = "t5-small"
    OUT = "/kaggle/working/eka-summarizer"
    PREFIX = "summarize: "
    MAX_INPUT, MAX_TARGET = 512, 96

    # SAMSum has moved repos more than once and ships a loading script.
    dataset = None
    for repo, needs_trust in (
        ("Samsung/samsum", True),
        ("samsum", True),
        ("knkarthick/samsum", False),
    ):
        try:
            dataset = load_dataset(repo, trust_remote_code=True) if needs_trust \
                else load_dataset(repo)
            print(f"Loaded SAMSum from '{repo}'")
            break
        except Exception as exc:
            print(f"  '{repo}' unavailable: {str(exc)[:120]}")
    if dataset is None:
        print("! SAMSum could not be loaded — skipping part 3.")
        print("  The backend falls back to t5-small base for summaries, which")
        print("  works, just less tuned to conversation.")
        RESULTS["summarizer"] = "SKIPPED (dataset unavailable)"
        return

    dialogue_col = "dialogue" if "dialogue" in dataset["train"].column_names else "text"
    print(f"  train {len(dataset['train'])} | val {len(dataset['validation'])}")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)

    def preprocess(batch):
        inputs = [PREFIX + (d or "") for d in batch[dialogue_col]]
        encoded = tokenizer(inputs, max_length=MAX_INPUT, truncation=True)
        labels = tokenizer(
            text_target=[s or "" for s in batch["summary"]],
            max_length=MAX_TARGET,
            truncation=True,
        )
        encoded["labels"] = labels["input_ids"]
        return encoded

    tokenized = dataset.map(
        preprocess, batched=True, remove_columns=dataset["train"].column_names
    )
    rouge = evaluate.load("rouge")

    def metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        return {k: round(v * 100, 2) for k, v in scores.items()}

    args = Seq2SeqTrainingArguments(
        output_dir=OUT,
        num_train_epochs=3,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        # warmup_steps, not warmup_ratio: Kaggle's transformers 5.0 only
        # DEPRECATED the ratio form, but a newer transformers removed it
        # outright and raises TypeError. warmup_steps works on both.
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=200,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET,
        # T5 is numerically unstable in fp16 (known issue — produces NaN loss).
        fp16=False,
        report_to="none",
        seed=42,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=metrics,
    )
    trainer.train()
    final = trainer.evaluate()
    rouge_l = final.get("eval_rougeL", 0.0)
    print(f"\n  ROUGE-L {rouge_l:.2f} (target >35)  ROUGE-1 {final.get('eval_rouge1', 0):.2f}")

    sample = dataset["validation"][0]
    inputs = tokenizer(PREFIX + sample[dialogue_col], return_tensors="pt",
                       max_length=MAX_INPUT, truncation=True).to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=MAX_TARGET, num_beams=4)
    print(f"\n  reference : {sample['summary']}")
    print(f"  generated : {tokenizer.decode(generated[0], skip_special_tokens=True)}")

    trainer.save_model(OUT)
    tokenizer.save_pretrained(OUT)
    push(OUT, "eka-summarizer", f"""---
base_model: {BASE}
pipeline_tag: summarization
tags: [eka, summarization]
---

# eka-summarizer

Conversation summarizer for **Eka**'s daily insight cards. T5-small fine-tuned
on SAMSum (dialogue -> summary), which is the closest public analogue to a day
of Eka messages.

ROUGE-L **{rouge_l:.2f}**, ROUGE-1 {final.get('eval_rouge1', 0):.2f}.

Called with the `summarize: ` prefix by `backend/services/insight_service.py`.
""")
    RESULTS["summarizer"] = f"ROUGE-L {rouge_l:.2f}"
    free_gpu(model, trainer)


# ==============================================================================
# RUN
# ==============================================================================
if __name__ == "__main__":
    if "1" in PARTS:
        part1_complexity()
    if "2" in PARTS:
        part2_sentiment()
    if "3" in PARTS:
        part3_summarizer()

    print(f"\n{'=' * 70}\n  ALL PARTS DONE\n{'=' * 70}")
    for name, result in RESULTS.items():
        print(f"  {name:<12} {result}")
    print(f"\n  repos: {HF_USERNAME}/eka-complexity, /eka-sentiment, /eka-summarizer")
    print("  next: python training/train_ranker_local.py  (CPU, <10 min)")
