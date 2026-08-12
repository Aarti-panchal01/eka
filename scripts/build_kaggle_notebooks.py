#!/usr/bin/env python3
"""
Build the four Kaggle persona-LoRA notebooks from the scripts in training/.

The .py files in training/ stay the source of truth. This script slices each one
at its `# SECTION n — TITLE` banners and emits a matching .ipynb into
ml/notebooks/, so a notebook can never drift from the script it came from.

    python scripts/build_kaggle_notebooks.py            # build all four
    python scripts/build_kaggle_notebooks.py --check    # CI: fail if stale
    python scripts/build_kaggle_notebooks.py --mode gita

Why notebooks at all when start_kaggle_training.md says "paste the script into a
cell"? A single 450-line cell gives you one traceback with no idea which stage
died, and no way to re-run just the push-to-Hub step after a 3-hour train
succeeded and the upload 401'd. Cell-per-section fixes both. The paste-one-cell
route still works and is still documented — this is the nicer path, not a
replacement.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

# A stock Windows console is cp1252 and dies on the ✓ below. Every other script
# here writes to a log file and never hits it; this one prints to a terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO / "training"
NOTEBOOK_DIR = REPO / "ml" / "notebooks"

MODES = ["founder", "chanakya", "gita", "reflection"]

SECTION_RE = re.compile(r"^# SECTION (\d+) [—-] (.+)$")
BANNER_RE = re.compile(r"^# ={10,}$")

# Kaggle stores per-notebook settings here. Import honours some of these, but
# not reliably across UI versions — the README tells the reader to eyeball the
# Settings panel anyway. Setting them costs nothing and usually saves two clicks.
KAGGLE_META = {
    "accelerator": "nvidiaTeslaT4",
    "dataSources": [],
    "isInternetEnabled": True,
    "isGpuEnabled": True,
    "language": "python",
    "sourceType": "notebook",
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list[str]:
    """nbformat wants a list of lines, each keeping its trailing newline."""
    text = text.strip("\n")
    if not text:
        return []
    parts = text.split("\n")
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


def parse_sections(source: str) -> tuple[str, list[dict]]:
    """Split a training script into (module docstring, [section, ...]).

    A section banner looks like:

        # =========================================
        # SECTION 3 — CONFIG
        # optional extra prose lines
        # =========================================
    """
    docstring = ast.get_docstring(ast.parse(source)) or ""
    lines = source.split("\n")

    starts = [i for i, line in enumerate(lines) if SECTION_RE.match(line)]
    if not starts:
        raise SystemExit("no '# SECTION n — TITLE' banners found — did the script format change?")

    sections = []
    for idx, marker in enumerate(starts):
        number, title = SECTION_RE.match(lines[marker]).groups()

        # Walk back to the banner's opening '# ====' rule.
        banner_top = marker
        while banner_top > 0 and not BANNER_RE.match(lines[banner_top - 1]):
            banner_top -= 1
        banner_top -= 1

        # Walk forward to the banner's closing rule; prose in between is the
        # section's own commentary and becomes the markdown cell.
        cursor = marker + 1
        prose = []
        while cursor < len(lines) and not BANNER_RE.match(lines[cursor]):
            prose.append(re.sub(r"^# ?", "", lines[cursor]))
            cursor += 1
        body_start = cursor + 1

        body_end = len(lines)
        if idx + 1 < len(starts):
            nxt = starts[idx + 1]
            while nxt > 0 and not BANNER_RE.match(lines[nxt - 1]):
                nxt -= 1
            body_end = nxt - 1

        sections.append(
            {
                "number": int(number),
                "title": title.strip(),
                "prose": "\n".join(prose).strip(),
                "body": "\n".join(lines[body_start:body_end]).strip("\n"),
            }
        )

    return docstring, sections


def pip_packages(source: str) -> list[str]:
    """Pull the pinned package list out of _pip_install so it stays in one place."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_pip_install":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.List):
                    names = [
                        elt.value
                        for elt in sub.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
                    if names:
                        return names
    raise SystemExit("could not find the pinned package list in _pip_install()")


def header_markdown(mode: str, docstring: str) -> str:
    """Turn the script's docstring preamble into the notebook's title cell."""
    # Everything before the first ALL-CAPS heading is the banner; drop the rules.
    body = "\n".join(
        line for line in docstring.split("\n") if not set(line.strip()) <= {"=", "-", ""}
    ).strip()
    # The docstring's "BEFORE YOU RUN" / "ESTIMATED TIME" headings render as
    # bold lines rather than markdown headings, which keeps the nav clean.
    body = re.sub(r"^([A-Z][A-Z &/\-]{3,})$", r"**\1**", body, flags=re.MULTILINE)
    body = body.replace(
        "Paste into a Kaggle\nnotebook (or upload as a script) and run top to bottom.",
        "Upload this notebook to Kaggle and run top to bottom.",
    )

    return (
        f"# Eka — `{mode}` persona QLoRA\n\n"
        f"{body}\n\n"
        "---\n\n"
        "**This notebook is generated.** Edit `training/"
        f"train_{mode}_lora_kaggle.py` and re-run "
        "`python scripts/build_kaggle_notebooks.py`; edits made here are "
        "overwritten on the next build.\n"
    )


def build(mode: str) -> dict:
    script = TRAINING_DIR / f"train_{mode}_lora_kaggle.py"
    source = script.read_text(encoding="utf-8")
    docstring, sections = parse_sections(source)
    packages = pip_packages(source)

    cells: list[dict] = [md(header_markdown(mode, docstring))]

    # Install as a real magic cell, then tell SECTION 1 to skip its own install
    # rather than paying for a second no-op pip resolve on every restart.
    installs = " \\\n    ".join(packages)
    cells.append(md("## Setup — install pinned dependencies\n\nRestart-safe: re-running this cell is a no-op once the versions match."))
    cells.append(
        code(
            "%%capture\n"
            f"!pip install -q {installs}\n\n"
            "import os\n"
            '# SECTION 1 below re-installs unless this is set; the cell above already did it.\n'
            'os.environ["EKA_SKIP_INSTALL"] = "1"'
        )
    )

    for section in sections:
        heading = f"## Section {section['number']} — {section['title']}"
        cells.append(md(heading + (f"\n\n{section['prose']}" if section["prose"] else "")))
        if section["body"]:
            cells.append(code(section["body"]))

    cells.append(
        md(
            "---\n\n## Done\n\n"
            "Confirm the adapter actually landed on the Hub before you count this "
            "run as finished — a repo with only a `README.md` means the push failed:\n\n"
            "```bash\n"
            "python -c \"\n"
            "from huggingface_hub import HfApi; import os\n"
            "from dotenv import load_dotenv; load_dotenv('.env')\n"
            "print(HfApi(token=os.getenv('HF_TOKEN')).list_repo_files(\n"
            f"    'amijackofalltrades/eka-{mode}-lora'))\"\n"
            "```\n\n"
            f"Then merge locally: `python ml/scripts/merge_lora.py --mode {mode}`\n"
        )
    )

    # nbformat >=4.5 requires a cell id, and warns loudly without one. Numbering
    # them by position keeps the id stable across rebuilds, so a regenerated
    # notebook diffs only where the script actually changed.
    for position, cell in enumerate(cells):
        cell["id"] = f"{cell['cell_type'][:2]}-{position:02d}"

    return {
        "cells": cells,
        "metadata": {
            "kaggle": KAGGLE_META,
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.13",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, help="build one persona instead of all four")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any notebook differs from what would be generated",
    )
    args = parser.parse_args()

    modes = [args.mode] if args.mode else MODES
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    stale = []
    for mode in modes:
        notebook = build(mode)
        target = NOTEBOOK_DIR / f"eka_{mode}_lora_kaggle.ipynb"
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
            print("stale notebooks: " + ", ".join(stale))
            print("run: python scripts/build_kaggle_notebooks.py")
            return 1
        print("✓ all notebooks match training/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
