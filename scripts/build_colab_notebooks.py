#!/usr/bin/env python3
"""Generate Colab notebooks from the same training/ scripts Kaggle uses.

    python scripts/build_colab_notebooks.py            # all six
    python scripts/build_colab_notebooks.py --mode founder
    python scripts/build_colab_notebooks.py --check    # CI: fail if stale

training/ stays the single source of truth. This applies the handful of
platform substitutions (where credentials come from, where output goes) and
prepends the three cells Colab needs and Kaggle does not: a GPU check, a Drive
mount, and an explicit install.

Why the installs are explicit here and almost absent in the Kaggle build: the
Kaggle image was probed live, so we know exactly what it ships. Colab's image
cannot be probed from here, and the Kaggle version of this exercise cost two
wasted sessions to a stack mismatch. So Colab installs the full set unpinned
(-U), which resolves a mutually-consistent modern stack from PyPI rather than
betting on whatever the base image happens to carry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kaggle_notebooks import (  # noqa: E402
    NOTEBOOK_DIR,
    REPO,
    TARGETS,
    TRAINING_DIR,
    code,
    md,
    parse_sections,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DRIVE = "/content/drive/MyDrive"

# ---------------------------------------------------------------- rewrites
# Every Kaggle-specific path, and the one Kaggle-only import.
SUBS = [
    ('"/kaggle/input/eka-secrets/secrets.json"',
     f'"{DRIVE}/eka-secrets/secrets.json"'),
    ('f"/kaggle/working/{MODE}_lora"',
     f'f"{DRIVE}/eka_models/{{MODE}}_lora"'),
    ('"/kaggle/working/eka-embeddings"', f'"{DRIVE}/eka_models/eka-embeddings"'),
    ('"/kaggle/working/eka-complexity"', f'"{DRIVE}/eka_models/eka-complexity"'),
    ('"/kaggle/working/eka-sentiment"', f'"{DRIVE}/eka_models/eka-sentiment"'),
    ('"/kaggle/working/eka-summarizer"', f'"{DRIVE}/eka_models/eka-summarizer"'),
    ('PROGRESS_FILE = "/kaggle/working/progress.txt"',
     f'PROGRESS_FILE = "{DRIVE}/eka_models/progress.txt"'),
    # The loader tries Kaggle Secrets as its second source. On Colab that module
    # does not exist; raising inside the existing try/except skips the branch
    # cleanly and leaves no kaggle_secrets import in the notebook.
    ("from kaggle_secrets import UserSecretsClient",
     'raise ImportError("kaggle_secrets is Kaggle-only; Colab reads Drive")'),
    ("save_total_limit=3,  # Kaggle /kaggle/working is capped at 20GB",
     "save_total_limit=2,  # checkpoints land in Drive, which fills faster"),
    ("# mirrors it to /kaggle/working — committed output survives cancellation.",
     "# mirrors it to Drive, which survives the runtime being recycled."),
]

# Per-notebook install lines. Unpinned on purpose — see the module docstring.
INSTALLS = {
    "persona": "transformers trl peft accelerate datasets bitsandbytes huggingface_hub",
    "embeddings": "sentence-transformers datasets huggingface_hub",
    "classifiers": ("transformers datasets accelerate evaluate scikit-learn "
                    "rouge-score sentencepiece huggingface_hub"),
}

PERSONAS = ("founder", "chanakya", "gita", "reflection")


def install_key(mode: str) -> str:
    return mode if mode in ("embeddings", "classifiers") else "persona"


def preamble(mode: str) -> list[dict]:
    title = TARGETS[mode][2]
    cells = [
        md(
            f"# Eka — {title} (Colab)\n\n"
            f"**Generated from `training/{TARGETS[mode][0]}`** by "
            "`scripts/build_colab_notebooks.py`. Edit the script, not this "
            "notebook.\n\n"
            "**Before you run:**\n\n"
            "1. **Runtime → Change runtime type → T4 GPU**, then Save.\n"
            f"2. Put `secrets.json` at `{DRIVE}/eka-secrets/secrets.json` "
            "with `HF_TOKEN` and `HF_USERNAME`.\n"
            "3. Run all. The Drive mount will ask you to authorise once.\n\n"
            "Output goes to Drive, so a disconnect does not lose checkpoints — "
            "re-running resumes from the last one."
        ),
        code(
            "# ---- GPU check ----------------------------------------------\n"
            "# Colab hands out T4 / L4 / A100 depending on the day. Anything\n"
            "# Ampere or newer supports bf16; T4 does not, and the script picks\n"
            "# the compute dtype from this, so it is worth seeing up front.\n"
            "import torch\n"
            "\n"
            "if not torch.cuda.is_available():\n"
            "    raise SystemExit(\n"
            "        'No GPU. Runtime -> Change runtime type -> T4 GPU, then '\n"
            "        'Run all again.'\n"
            "    )\n"
            "print(torch.cuda.get_device_name(0))\n"
            "print('capability', torch.cuda.get_device_capability(0))\n"
            "print('bf16 supported:', torch.cuda.is_bf16_supported())\n"
            "print('VRAM GB:', round(\n"
            "    torch.cuda.get_device_properties(0).total_memory / 1e9, 1))"
        ),
        code(
            "# ---- Drive ---------------------------------------------------\n"
            "# Credentials come in here, and checkpoints go out here. Kaggle\n"
            "# used an attached dataset; Colab has no equivalent, and Drive is\n"
            "# the only store that survives a runtime being recycled.\n"
            "from google.colab import drive\n"
            "\n"
            "drive.mount('/content/drive')\n"
            "\n"
            "import os\n"
            "\n"
            f"os.makedirs('{DRIVE}/eka_models', exist_ok=True)\n"
            f"secrets_path = '{DRIVE}/eka-secrets/secrets.json'\n"
            "assert os.path.exists(secrets_path), (\n"
            "    f'{secrets_path} not found. Create it with HF_TOKEN and '\n"
            "    'HF_USERNAME before running.'\n"
            ")\n"
            "print('secrets found:', secrets_path)"
        ),
        code(
            "# ---- install -------------------------------------------------\n"
            "# Unpinned and -U on purpose. Colab's preinstalled stack cannot be\n"
            "# probed from outside, and pinning a stale set against an unknown\n"
            "# image is exactly what cost the Kaggle runs two sessions. Letting\n"
            "# pip resolve a consistent modern set is the safer bet; the\n"
            "# versions print below so a failure has evidence attached.\n"
            "%pip install -q -U " + INSTALLS[install_key(mode)] + "\n"
            "\n"
            "import importlib.metadata as _md\n"
            "\n"
            "for _p in '" + INSTALLS[install_key(mode)] + "'.split():\n"
            "    try:\n"
            "        print(f'{_p:24} {_md.version(_p)}')\n"
            "    except Exception:\n"
            "        print(f'{_p:24} (version unavailable)')\n"
            "\n"
            "import os\n"
            "\n"
            "# The script installs its own deps unless told not to; this cell\n"
            "# already did it.\n"
            "os.environ['EKA_SKIP_INSTALL'] = '1'"
        ),
    ]

    if mode in PERSONAS:
        cells.append(
            code(
                "# ---- smoke test toggle ---------------------------------------\n"
                "# '1' caps training at 12 steps (~15 min) and exits WITHOUT\n"
                "# pushing an adapter, which is enough to measure the real\n"
                "# s/step. The persona estimate has already been wrong by 4x\n"
                "# once; measuring costs a quarter hour, guessing cost twelve.\n"
                "# Now '0' — the 2026-08-14 smoke test measured 361 s/step and\n"
                "# the config was tuned off the back of it. Set to '1' to\n"
                "# re-measure after any change to r / batch / seq len.\n"
                "import os\n"
                "\n"
                "os.environ['EKA_SMOKE'] = '0'\n"
                "print('EKA_SMOKE =', os.environ['EKA_SMOKE'])"
            )
        )
    return cells


def build(mode: str) -> dict:
    source = (TRAINING_DIR / TARGETS[mode][0]).read_text(encoding="utf-8")

    for old, new in SUBS:
        source = source.replace(old, new)

    # A missed path is a run that writes into a directory that does not exist,
    # so this is a hard stop rather than a warning. Prose mentions of the old
    # platform are fine — only live code matters.
    leftover = [
        ln.strip() for ln in source.splitlines()
        if "/kaggle/" in ln and not ln.strip().startswith("#")
    ]
    if leftover:
        raise SystemExit(
            f"{mode}: unconverted Kaggle path(s) in code:\n  " + "\n  ".join(leftover)
        )

    _, sections = parse_sections(source)
    cells = preamble(mode)
    for section in sections:
        heading = f"## Section {section['number']} — {section['title']}"
        cells.append(md(heading + (f"\n\n{section['prose']}" if section["prose"] else "")))
        if section["body"]:
            cells.append(code(section["body"]))

    for position, cell in enumerate(cells):
        cell["id"] = f"{cell['cell_type'][:2]}-{position:02d}"

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(TARGETS))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    modes = [args.mode] if args.mode else list(TARGETS)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    stale = []
    for mode in modes:
        notebook = build(mode)
        target = NOTEBOOK_DIR / f"eka_{mode}_colab.ipynb"
        rendered = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
        if args.check:
            current = target.read_text(encoding="utf-8") if target.exists() else ""
            if current != rendered:
                stale.append(target.name)
            continue
        target.write_text(rendered, encoding="utf-8")
        n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
        print(f"✓ {target.relative_to(REPO)}  ({len(notebook['cells'])} cells, {n_code} code)")

    if args.check:
        if stale:
            print("stale: " + ", ".join(stale))
            return 1
        print("✓ all Colab notebooks match training/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
