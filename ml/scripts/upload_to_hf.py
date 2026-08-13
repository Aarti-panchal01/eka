"""Push every training split + auxiliary dataset to the HF Hub dataset repo.

    python ml/scripts/upload_to_hf.py [--repo eka-datasets] [--public]

Creates {HF_USERNAME}/eka-datasets (private by default) if it doesn't exist,
then uploads:
    ml/data/splits/{mode}_{train,val}.jsonl
    ml/datasets/embedding_triplets.jsonl
    ml/datasets/complexity_labeled.jsonl

The Kaggle notebooks read straight from this repo, which is why the training
step needs no Kaggle dataset upload at all.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode ✓/✅/⏭.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPTS_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = ML_DIR.parent
SPLITS_DIR = ML_DIR / "data" / "splits"
DATASETS_DIR = ML_DIR / "datasets"

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

MODES = ("founder", "chanakya", "gita", "reflection")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main(repo_name: str, private: bool) -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    username = os.environ.get("HF_USERNAME", "").strip()

    if not token:
        sys.exit("HF_TOKEN not set — add it to .env")
    if not username:
        sys.exit(
            "HF_USERNAME not set — add it to .env.\n"
            "  It's the name in huggingface.co/<username>."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface-hub")

    repo_id = repo_name if "/" in repo_name else f"{username}/{repo_name}"
    api = HfApi(token=token)

    print(f"EKA upload -> https://huggingface.co/datasets/{repo_id}\n")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    targets = []
    for mode in MODES:
        for split in ("train", "val"):
            targets.append(SPLITS_DIR / f"{mode}_{split}.jsonl")
    targets.append(DATASETS_DIR / "embedding_triplets.jsonl")
    targets.append(DATASETS_DIR / "complexity_labeled.jsonl")

    uploaded = skipped = 0
    for path in targets:
        if not path.exists():
            print(f"⏭  {path.name:<32} not found — skipped")
            skipped += 1
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"✓ Uploaded {path.name:<32} ({count_lines(path)} examples)")
        uploaded += 1

    # A README makes the repo readable on the Hub and documents the schema.
    readme = f"""---
license: other
tags:
- eka
- persona
- instruction-tuning
---

# eka-datasets

Training data for **Eka**, a lifelong AI companion with four fine-tuned personas.

## Files

| file | schema | used by |
|---|---|---|
| `{{mode}}_train.jsonl` / `{{mode}}_val.jsonl` | `{{"text": "<ChatML chat sequence>", "mode", "tags"}}` | `train_{{mode}}_lora_kaggle.py` |
| `embedding_triplets.jsonl` | `{{"anchor", "positive", "negative", "mode"}}` | `train_embeddings_kaggle.py` |
| `complexity_labeled.jsonl` | `{{"query", "label"}}` where label ∈ simple/normal/complex/deep | `train_classifiers_kaggle.py` |

Modes: {", ".join(MODES)}.

Generated with `ml/scripts/generate_*_data.py` (Groq, Llama 3.3 70B) and
formatted by `ml/scripts/preprocess.py`.
"""
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"✓ Uploaded README.md")

    print(f"\n✅ {uploaded} files uploaded, {skipped} skipped")
    print(f"   repo: https://huggingface.co/datasets/{repo_id}")
    print("   next: run training/train_founder_lora_kaggle.py as a Kaggle notebook")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="eka-datasets")
    parser.add_argument("--public", action="store_true", help="make the repo public")
    args = parser.parse_args()
    main(args.repo, private=not args.public)
