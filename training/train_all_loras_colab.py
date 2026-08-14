"""
================================================================================
EKA — all four persona QLoRA adapters, sequentially, on one Colab T4
================================================================================
founder -> chanakya -> gita -> reflection

BUILT TO BE RE-RUN. A free Colab session will not survive 14 hours: it caps out
around 12, disconnects after ~90 minutes idle, and sustained use pushes you into
a cooldown. So this notebook is idempotent — every persona whose adapter is
already on the Hub is skipped, and any persona with checkpoints in Drive resumes
from the last one instead of restarting. Disconnect, reconnect, Run all again,
and it continues where it stopped.

BEFORE YOU RUN
--------------
1. Runtime -> Change runtime type -> T4 GPU
2. /content/drive/MyDrive/eka-secrets/secrets.json with HF_TOKEN + HF_USERNAME
3. Run all. Then leave the tab open and the machine awake.

WHY SEQUENTIAL
--------------
Free Colab grants one GPU. Two personas at once means two 4-bit 7B models on a
16GB card, which OOMs before either finishes. Each model is fully released
before the next loads.
================================================================================
"""

# ==============================================================================
# SECTION 1 — CONFIG
# ==============================================================================
import gc
import json
import os
import time

DRIVE = "/content/drive/MyDrive"
SECRETS_PATH = f"{DRIVE}/eka-secrets/secrets.json"
CKPT_ROOT = f"{DRIVE}/eka_models"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# alpha/r = 2, the same ratio the original r=16/alpha=32 had. Halving r without
# halving alpha would have doubled the adapter's effective strength, which is a
# behaviour change disguised as a speed change.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM = 4          # effective batch 16
LR = 2e-4
# The longest example in any split is 1,078 tokens. 2048 never truncated
# anything — it only set the padding width, so half of every batch was pad
# tokens paid for at full price.
MAX_SEQ_LEN = 1152
SAVE_STEPS = 25         # denser than usual: Colab drops sessions
EVAL_STEPS = 25

PERSONAS = ["founder", "chanakya", "gita", "reflection"]

# Set to a single name to redo just one, e.g. ONLY = "gita"
ONLY = os.environ.get("EKA_ONLY", "").strip()
# "1" retrains personas whose adapter is already published.
FORCE = os.environ.get("EKA_FORCE", "") == "1"


# ==============================================================================
# SECTION 2 — AUTH
# ==============================================================================
def _load_secrets() -> dict:
    """Credentials from the Drive file the classifiers notebook already uses."""
    with open(SECRETS_PATH, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    missing = [k for k in ("HF_TOKEN", "HF_USERNAME") if not blob.get(k)]
    if missing:
        raise SystemExit(f"{SECRETS_PATH} is missing: {', '.join(missing)}")
    for key, value in blob.items():
        if value:
            os.environ[key] = str(value).strip()
    return blob


SECRETS = _load_secrets()
HF_USERNAME = os.environ["HF_USERNAME"]
HF_TOKEN = os.environ["HF_TOKEN"]

# One GPU only. Trainer uses every visible card via nn.DataParallel, and the
# model is pinned to cuda:0 — a second card puts replica 1 on cuda:1 against
# weights on cuda:0 and the first step dies.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from huggingface_hub import HfApi, login  # noqa: E402

login(token=HF_TOKEN)
API = HfApi(token=HF_TOKEN)
print(f"✓ authenticated as {HF_USERNAME}")

import torch  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("No GPU. Runtime -> Change runtime type -> T4 GPU.")

CAPABILITY = torch.cuda.get_device_capability(0)
if CAPABILITY < (7, 0):
    raise SystemExit(
        f"{torch.cuda.get_device_name(0)} is sm_{CAPABILITY[0]}{CAPABILITY[1]}; "
        "bitsandbytes 4-bit needs 7.0+. Reconnect for a different GPU."
    )

SUPPORTS_BF16 = torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if SUPPORTS_BF16 else torch.float16
print(f"GPU: {torch.cuda.get_device_name(0)} · bf16={SUPPORTS_BF16}")
os.makedirs(CKPT_ROOT, exist_ok=True)


# ==============================================================================
# SECTION 3 — HELPERS
# ==============================================================================
import glob  # noqa: E402
import time as _time  # noqa: E402

from datasets import load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer  # noqa: E402

RESULTS_PATH = f"{CKPT_ROOT}/lora_results.json"


def already_published(mode: str) -> bool:
    """An adapter on the Hub means this persona is done."""
    try:
        files = API.list_repo_files(f"{HF_USERNAME}/eka-{mode}-qwen")
        return any("adapter_model" in f for f in files)
    except Exception:
        return False


def latest_checkpoint(directory: str):
    found = [d for d in glob.glob(os.path.join(directory, "checkpoint-*")) if os.path.isdir(d)]
    if not found:
        return None
    return max(found, key=lambda p: int(p.rsplit("-", 1)[-1]))


class Progress(TrainerCallback):
    """Colab hides nothing, but a disconnected session shows nothing either.
    Written to Drive so a dead tab still leaves a record of the rate."""

    def __init__(self, mode):
        self.mode = mode

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = _time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.global_step:
            return
        elapsed = _time.time() - self.t0
        per_step = elapsed / state.global_step
        left = (state.max_steps - state.global_step) * per_step
        line = (
            f"{self.mode} step {state.global_step}/{state.max_steps} "
            f"{per_step:.1f}s/step eta {left / 3600:.2f}h "
            f"loss={(logs or {}).get('loss', '?')}"
        )
        print("  ⏱ " + line, flush=True)
        try:
            with open(f"{CKPT_ROOT}/progress.txt", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass


def record(mode: str, payload: dict) -> None:
    """Accumulate per-adapter metrics for the README table."""
    data = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            data = {}
    data[mode] = payload
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"  metrics -> {RESULTS_PATH}")


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU freed — {torch.cuda.memory_allocated() / 1e9:.2f} GB still allocated")


def train_one(mode: str) -> None:
    out_dir = f"{CKPT_ROOT}/{mode}_lora"
    repo = f"{HF_USERNAME}/eka-{mode}-qwen"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 70}\n  {mode.upper()}  ->  {repo}\n{'=' * 70}")

    dataset = load_dataset(
        f"{HF_USERNAME}/eka-datasets",
        data_files={"train": f"{mode}_train.jsonl", "validation": f"{mode}_val.jsonl"},
        token=HF_TOKEN,
    )
    print(f"  train {len(dataset['train'])} | val {len(dataset['validation'])}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    # Qwen's eos IS <|im_end|>, the token the ChatML template ends every turn
    # with. pad_token = eos would make the collator mask every stop token out
    # of the labels and the adapter would never learn to stop.
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|endoftext|>"
    tokenizer.padding_side = "right"
    print(f"  pad={tokenizer.pad_token!r} eos={tokenizer.eos_token!r}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        ),
        device_map={"": 0},
        token=HF_TOKEN,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        # warmup_steps, not warmup_ratio: current transformers removed the
        # ratio form and raises TypeError on it.
        warmup_steps=4,
        max_grad_norm=0.3,
        weight_decay=0.001,
        fp16=not SUPPORTS_BF16,
        bf16=SUPPORTS_BF16,
        optim="paged_adamw_8bit",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,     # Drive fills up fast; two is enough to resume
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",       # wandb reaches for np.float_ under NumPy 2
        seed=42,
        group_by_length=True,
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    trainer.add_callback(Progress(mode))

    resume = latest_checkpoint(out_dir)
    print(f"  {'resuming from ' + os.path.basename(resume) if resume else 'starting fresh'}")

    started = time.time()
    trainer.train(resume_from_checkpoint=resume)
    metrics = trainer.evaluate()
    minutes = (time.time() - started) / 60
    print(f"  eval_loss {metrics.get('eval_loss'):.4f} in {minutes:.0f} min")

    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    card = f"""---
base_model: {BASE_MODEL}
library_name: peft
tags: [eka, lora, qlora, {mode}]
---

# eka-{mode}-qwen

QLoRA adapter giving Qwen2.5-7B-Instruct Eka's **{mode}** persona.

| | |
|---|---|
| base | `{BASE_MODEL}` |
| rank / alpha | {LORA_R} / {LORA_ALPHA} |
| epochs / effective batch | {EPOCHS} / {BATCH_SIZE * GRAD_ACCUM} |
| max seq len | {MAX_SEQ_LEN} |
| train / val | {len(dataset['train'])} / {len(dataset['validation'])} |
| final eval loss | {metrics.get('eval_loss', float('nan')):.4f} |
| trained on | {torch.cuda.get_device_name(0)} (Colab) |
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)

    API.create_repo(repo_id=repo, private=True, exist_ok=True)
    API.upload_folder(
        folder_path=out_dir,
        repo_id=repo,
        ignore_patterns=["checkpoint-*", "*.pt", "runs/*"],
    )
    print(f"✅ pushed https://huggingface.co/{repo}")

    record(mode, {
        "train_rows": len(dataset["train"]),
        "val_rows": len(dataset["validation"]),
        "steps": int(trainer.state.max_steps),
        "eval_loss": round(float(metrics.get("eval_loss", float("nan"))), 4),
        "minutes": round(minutes, 1),
        "r": LORA_R, "alpha": LORA_ALPHA, "epochs": EPOCHS,
        "max_seq_len": MAX_SEQ_LEN,
    })

    del trainer, model
    free_gpu()


# ==============================================================================
# SECTION 4 — RUN ALL FOUR
# ==============================================================================
queue = [ONLY] if ONLY else PERSONAS
print(f"\nqueue: {queue}\n")

for name in queue:
    if not FORCE and already_published(name):
        print(f"⏭  {name}: adapter already on the Hub — skipping")
        continue
    try:
        train_one(name)
    except Exception as exc:
        # One persona failing must not cost the ones after it. Checkpoints are
        # in Drive, so re-running resumes this persona rather than restarting.
        print(f"❌ {name} failed: {type(exc).__name__}: {exc}")
        free_gpu()

print(f"\n{'=' * 70}\n  DONE\n{'=' * 70}")
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, encoding="utf-8") as fh:
        for mode, row in json.load(fh).items():
            print(f"  {mode:<11} eval_loss {row['eval_loss']:<8} "
                  f"{row['steps']} steps  {row['minutes']} min")
print("\nRe-run this notebook after any disconnect — finished personas are "
      "skipped and unfinished ones resume from their last checkpoint.")
