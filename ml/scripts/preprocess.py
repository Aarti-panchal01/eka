"""Turn the raw persona datasets into Llama-3 chat-formatted train/val splits.

    python ml/scripts/preprocess.py

Reads   ml/datasets/{mode}_dataset.json
Writes  ml/data/splits/{mode}_train.jsonl   (90%)
        ml/data/splits/{mode}_val.jsonl     (10%)

Each output line is {"text": "<full Llama-3 chat sequence>", ...metadata}, which
is exactly what SFTTrainer's dataset_text_field wants.

The split is stratified by topic tag so no topic lands entirely in validation.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Windows consoles default to cp1252, which cannot encode ✓/✅/⏭.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPTS_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = ML_DIR.parent
DATASETS_DIR = ML_DIR / "datasets"
SPLITS_DIR = ML_DIR / "data" / "splits"
PROMPTS_DIR = PROJECT_ROOT / "backend" / "prompts"

MODES = ("founder", "chanakya", "gita", "reflection")
VAL_FRACTION = 0.10
random.seed(42)

# Llama-3 instruct chat template. The trailing <|eot_id|> is what teaches the
# model to stop — leaving it out is the single most common fine-tune bug.
TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{user}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{assistant}<|eot_id|>"
)


def load_persona(mode: str) -> str:
    path = PROMPTS_DIR / f"{mode}.txt"
    if not path.exists():
        raise SystemExit(f"Persona prompt missing: {path}")
    return path.read_text(encoding="utf-8").strip()


def stratified_split(rows: List[dict]) -> tuple:
    """Group by topic tag, hold out VAL_FRACTION from each group."""
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        tags = row.get("tags") or ["_untagged"]
        buckets[tags[0]].append(row)

    train, val = [], []
    for topic, bucket in buckets.items():
        random.shuffle(bucket)
        n_val = max(1, round(len(bucket) * VAL_FRACTION)) if len(bucket) >= 10 else 0
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def to_records(rows: List[dict], system: str, mode: str) -> List[dict]:
    records = []
    for row in rows:
        user = (row.get("user") or "").strip()
        assistant = (row.get("eka_response") or "").strip()
        if not user or not assistant:
            continue
        records.append(
            {
                "text": TEMPLATE.format(system=system, user=user, assistant=assistant),
                "mode": mode,
                "tags": row.get("tags", []),
            }
        )
    return records


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def approx_tokens(text: str) -> int:
    """chars/4 is close enough for planning MAX_SEQ_LEN."""
    return len(text) // 4


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    print("EKA preprocess — Llama-3 chat format\n")
    grand_train = grand_val = 0
    longest = 0

    for mode in MODES:
        source = DATASETS_DIR / f"{mode}_dataset.json"
        if not source.exists():
            print(f"⏭  {mode:<11} no dataset yet ({source.name}) — skipped")
            continue

        rows = json.loads(source.read_text(encoding="utf-8"))
        system = load_persona(mode)
        train_rows, val_rows = stratified_split(rows)
        train = to_records(train_rows, system, mode)
        val = to_records(val_rows, system, mode)

        # An empty val split is not a small problem, it is a broken artefact.
        # stratified_split gives a topic bucket zero validation rows until it
        # holds 10, so a persona that is still generating produces a 0-byte
        # *_val.jsonl. Upload that and the Kaggle run trains for ~15 minutes,
        # reaches its first eval step, and dies there — the training args set
        # evaluation_strategy="steps" with load_best_model_at_end=True and
        # metric_for_best_model="eval_loss", none of which survive an empty
        # eval set. Refuse to write the file rather than hand a GPU run a
        # landmine. Observed 2026-08-12 with chanakya at 49/600.
        if train and not val:
            print(
                f"⏭  {mode:<11} train {len(train):>5} | val    0 — SKIPPED, no "
                f"validation rows yet (needs >=10 pairs in a topic). Re-run "
                f"once {mode} finishes generating."
            )
            continue

        write_jsonl(SPLITS_DIR / f"{mode}_train.jsonl", train)
        write_jsonl(SPLITS_DIR / f"{mode}_val.jsonl", val)

        lengths = [approx_tokens(r["text"]) for r in train + val] or [0]
        mode_max = max(lengths)
        longest = max(longest, mode_max)
        grand_train += len(train)
        grand_val += len(val)

        print(
            f"✓ {mode:<11} train {len(train):>5} | val {len(val):>4} | "
            f"~tokens avg {sum(lengths) // len(lengths):>4} max {mode_max:>5}"
        )

    if grand_train == 0:
        raise SystemExit(
            "\nNothing to preprocess. Generate data first:\n"
            "  python ml/scripts/generate_founder_data.py\n"
        )

    print(f"\n  total: {grand_train} train / {grand_val} val")
    print(f"  longest sequence ~{longest} tokens")
    if longest > 2048:
        print("  ⚠ exceeds MAX_SEQ_LEN=2048 in the Kaggle notebooks —")
        print("    raise MAX_SEQ_LEN or those examples get truncated.")
    print(f"\n✅ splits written to {SPLITS_DIR}")
    print("   next: python ml/scripts/upload_to_hf.py")


if __name__ == "__main__":
    main()
