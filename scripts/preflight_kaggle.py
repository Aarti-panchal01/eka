#!/usr/bin/env python3
"""Check a persona is actually ready to train before you spend a GPU slot on it.

    python scripts/preflight_kaggle.py              # every persona
    python scripts/preflight_kaggle.py --mode gita
    python scripts/preflight_kaggle.py --hub        # also check the Hub copy

Everything a Kaggle run needs from this repo is verifiable locally in about a
second. The alternative is finding out ~15 minutes in, AFTER the 16GB base
model download, which is the part that costs real time — and on a 30 h/week
quota a wasted session is expensive.

Checks, in the order a training run would hit them:

  1. the split files exist at all
  2. every row has a non-empty `text` — SFTTrainer is built with
     dataset_text_field="text" and a missing column fails at trainer setup
  3. every row carries the full ChatML scaffolding; a stray row without an
     <|im_end|> teaches the model not to stop
  4. nothing exceeds MAX_SEQ_LEN, which would be silently truncated mid-answer
  5. the validation split is non-empty — evaluation_strategy="steps" with
     load_best_model_at_end=True and metric_for_best_model="eval_loss" cannot
     survive an empty eval set
  6. the quality report says ok_to_train

Exit code is 0 only if every requested persona passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT = Path(__file__).resolve().parent.parent
SPLITS = PROJECT / "ml" / "data" / "splits"
DATASETS = PROJECT / "ml" / "datasets"

MODES = ("founder", "chanakya", "gita", "reflection")
MAX_SEQ_LEN = 2048          # must match training/train_*_lora_kaggle.py
CHARS_PER_TOKEN = 4         # rough; only used to flag rows near the ceiling

# ChatML — must match ml/scripts/preprocess.py's TEMPLATE. If you swap the
# base model, swap these too, or this check passes a dataset the new
# tokenizer cannot read as chat at all.
REQUIRED_MARKERS = (
    "<|im_start|>system",
    "<|im_start|>user",
    "<|im_start|>assistant",
    "<|im_end|>",
)


def read_jsonl(path: Path) -> list:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path.name} line {n}: not valid JSON — {exc}")
    return rows


def check_mode(mode: str) -> list:
    """Return a list of problems; empty means ready to train."""
    problems = []
    train_path = SPLITS / f"{mode}_train.jsonl"
    val_path = SPLITS / f"{mode}_val.jsonl"

    for path in (train_path, val_path):
        if not path.exists():
            problems.append(f"{path.name} missing — run ml/scripts/preprocess.py")
    if problems:
        return problems

    train, val = read_jsonl(train_path), read_jsonl(val_path)

    if not train:
        problems.append("train split is empty")
    if not val:
        # The expensive one: this survives upload and dies at the first eval.
        problems.append(
            "val split is empty — evaluation_strategy='steps' with "
            "load_best_model_at_end cannot run"
        )

    for label, rows in (("train", train), ("val", val)):
        if not rows:
            continue
        no_text = sum(
            1 for r in rows if not isinstance(r.get("text"), str) or not r["text"].strip()
        )
        if no_text:
            problems.append(f"{label}: {no_text} row(s) with no usable 'text' field")

        for marker in REQUIRED_MARKERS:
            missing = sum(1 for r in rows if marker not in r.get("text", ""))
            if missing:
                problems.append(f"{label}: {missing} row(s) missing {marker}")

        longest = max((len(r.get("text", "")) // CHARS_PER_TOKEN for r in rows), default=0)
        over = sum(
            1 for r in rows if len(r.get("text", "")) // CHARS_PER_TOKEN > MAX_SEQ_LEN
        )
        if over:
            problems.append(
                f"{label}: {over} row(s) over MAX_SEQ_LEN {MAX_SEQ_LEN} "
                f"(longest ~{longest}) — they will be truncated mid-answer"
            )

    report_path = DATASETS / f"{mode}_dataset_quality_report.json"
    if not report_path.exists():
        problems.append(f"{report_path.name} missing — persona never completed a run")
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("verdict") != "ok_to_train":
            issues = "; ".join(report.get("blocking_issues") or []) or "unknown"
            problems.append(f"quality verdict is {report.get('verdict')}: {issues}")

    return problems


def check_hub(modes) -> list:
    """Confirm the Hub copy exists — Kaggle reads from there, not from disk."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT / ".env")
    except ImportError:
        pass
    token = os.environ.get("HF_TOKEN", "").strip()
    user = os.environ.get("HF_USERNAME", "").strip()
    if not token or not user:
        return ["HF_TOKEN / HF_USERNAME not set — cannot check the Hub"]
    from huggingface_hub import HfApi

    files = set(
        HfApi(token=token).list_repo_files(f"{user}/eka-datasets", repo_type="dataset")
    )
    missing = [
        f"{m}_{s}.jsonl"
        for m in modes
        for s in ("train", "val")
        if f"{m}_{s}.jsonl" not in files
    ]
    return (
        [f"not on the Hub yet: {', '.join(missing)} — run ml/scripts/upload_to_hf.py"]
        if missing
        else []
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, help="check one persona")
    parser.add_argument(
        "--hub", action="store_true", help="also verify the splits reached the Hub"
    )
    args = parser.parse_args()

    modes = [args.mode] if args.mode else list(MODES)
    ready = []

    print("Kaggle pre-flight\n")
    for mode in modes:
        problems = check_mode(mode)
        if problems:
            print(f"  ✗ {mode}")
            for p in problems:
                print(f"      - {p}")
        else:
            train = len(read_jsonl(SPLITS / f"{mode}_train.jsonl"))
            val = len(read_jsonl(SPLITS / f"{mode}_val.jsonl"))
            print(f"  ✓ {mode:11} train {train:>5} | val {val:>4} | ok_to_train")
            ready.append(mode)

    # The Hub check gates the verdict, it does not merely annotate it. Kaggle
    # reads the splits from the dataset repo, never from this disk, so
    # "passes locally but is not uploaded" is a session that fails after the
    # model download — exactly the outcome this script exists to prevent.
    hub_problems = []
    if args.hub and ready:
        print()
        hub_problems = check_hub(ready)
        for problem in hub_problems or ["✓ all locally-ready personas are on the Hub"]:
            print(f"  {problem}")

    print()
    if len(ready) == len(modes) and not hub_problems:
        where = "locally and on the Hub" if args.hub else "locally"
        print(f"{len(ready)}/{len(modes)} ready {where} — safe to start those sessions")
        if not args.hub:
            print("  (re-run with --hub to confirm Kaggle can actually see them)")
        return 0
    if hub_problems:
        print(f"{len(ready)}/{len(modes)} pass locally but the Hub is not in sync — "
              f"do not start a session yet")
    else:
        print(f"{len(ready)}/{len(modes)} ready — do not start a session for the others")
    return 1


if __name__ == "__main__":
    sys.exit(main())
