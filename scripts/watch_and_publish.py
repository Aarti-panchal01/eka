"""Unattended watcher: when the datasets finish, preprocess and publish them.

    python scripts/watch_and_publish.py              # watch, then run the pipeline
    python scripts/watch_and_publish.py --once       # check once and exit
    python scripts/watch_and_publish.py --min-frac 0.9   # accept 90% of target

WHY THIS IS A SCRIPT AND NOT A PERSON WATCHING

The generation queue runs for hours and finishes at an unpredictable time —
whenever the last provider's daily quota allows. Nobody should have to sit up
for it, and an assistant cannot: it runs when invoked and then stops. So the
"when data finishes, publish it" step has to be an unattended process.

Run this alongside run_queue.py and it will:

    1. Poll every POLL_SECONDS until every persona dataset reaches its target
       (or --min-frac of it), or until the queue process exits
    2. Run preprocess.py to build the train/val splits
    3. Run upload_to_hf.py to push splits + aux datasets to the Hub
    4. Re-score every dataset and refuse to publish anything whose quality
       report says needs_review
    5. Print the exact Kaggle steps for the morning

It is safe to run more than once: preprocess and upload are both idempotent,
and step 4 is a gate, not a mutation.

QUALITY IS A GATE, NOT A WARNING. If a persona's report verdict is
needs_review, this script does NOT upload it. Publishing a dataset that failed
its own gates is worse than publishing nothing, because the next step is a
multi-hour Kaggle run that bakes the defects into weights.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
ML_SCRIPTS = PROJECT / "ml" / "scripts"
DATASETS = PROJECT / "ml" / "datasets"
PYTHON = sys.executable

POLL_SECONDS = 120

PERSONAS = [
    ("founder", "generate_founder_data.py", "founder_dataset.json"),
    ("chanakya", "generate_chanakya_data.py", "chanakya_dataset.json"),
    ("gita", "generate_gita_data.py", "gita_dataset.json"),
    ("reflection", "generate_reflection_data.py", "reflection_dataset.json"),
]

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def target_for(module: str) -> int:
    """Read TOPIC_COUNTS out of a generator without importing its deps."""
    text = (ML_SCRIPTS / module).read_text(encoding="utf-8")
    inside, total = False, 0
    for line in text.splitlines():
        if line.startswith("TOPIC_COUNTS"):
            inside = True
            continue
        if inside:
            if line.startswith("}"):
                break
            if ":" in line:
                tail = line.rsplit(":", 1)[1].strip().rstrip(",")
                if tail.isdigit():
                    total += int(tail)
    return total


def count(dataset: str) -> int:
    try:
        with open(DATASETS / dataset, "r", encoding="utf-8") as handle:
            return len(json.load(handle))
    except Exception:
        return 0


def snapshot(min_frac: float):
    """Return (rows, all_done, lines) for the current state."""
    rows, done, lines = {}, True, []
    for name, module, dataset in PERSONAS:
        have, want = count(dataset), target_for(module)
        need = int(want * min_frac)
        ok = have >= need
        done = done and ok
        rows[name] = (have, want, ok)
        lines.append(
            f"  {name:11} {have:>5}/{want:<5} "
            f"{'ready' if ok else f'{need - have} short of {min_frac:.0%}'}"
        )
    return rows, done, lines


def queue_running() -> bool:
    """True if a generator or the queue is still alive (Windows-safe)."""
    try:
        out = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return True  # can't tell — assume alive rather than publishing early
    # tasklist doesn't show command lines, so fall back to "any python running".
    # Deliberately conservative: a false "still running" only delays publishing,
    # while a false "finished" would publish a partial dataset.
    return "python.exe" in out.lower()


def run(label: str, args: list) -> bool:
    print(f"\n{'=' * 64}\n  {label}\n{'=' * 64}", flush=True)
    result = subprocess.run(args, cwd=str(PROJECT))
    ok = result.returncode == 0
    print(f"  -> exit {result.returncode} ({'ok' if ok else 'FAILED'})", flush=True)
    return ok


def quality_gate() -> bool:
    """Refuse to publish any persona whose own report says needs_review."""
    print(f"\n{'=' * 64}\n  QUALITY GATE\n{'=' * 64}")
    blocking = []
    for name, module, dataset in PERSONAS:
        report = DATASETS / (dataset.replace(".json", "") + "_quality_report.json")
        if not report.exists():
            print(f"  {name:11} no report — run --report-only for it")
            blocking.append(f"{name}: no quality report")
            continue
        data = json.loads(report.read_text(encoding="utf-8"))
        verdict = data.get("verdict", "unknown")
        print(
            f"  {name:11} {verdict:14} "
            f"pairs={data.get('total_pairs')} "
            f"marker={data.get('persona_marker_pass_rate')} "
            f"unique={data.get('unique_pair_guarantee')}"
        )
        if verdict != "ok_to_train":
            for issue in data.get("blocking_issues", []):
                print(f"      - {issue}")
            blocking.append(f"{name}: {verdict}")
    if blocking:
        print("\n  ✗ NOT PUBLISHING — these must be resolved first:")
        for b in blocking:
            print(f"      {b}")
        return False
    print("\n  ✓ all four personas cleared for training")
    return True


def kaggle_instructions() -> None:
    print(
        f"""
{'=' * 64}
  KAGGLE — next steps
{'=' * 64}
Datasets are on the Hub. Full detail: scripts/start_kaggle_training.md

Per session (four persona runs, ~2.5-3.5h each on one T4):
  1. kaggle.com/code -> New Notebook, name it for the run
  2. Settings: Accelerator = GPU T4 x2, Internet = On
  3. Add-ons -> Secrets: HF_TOKEN (write), HF_USERNAME, WANDB_API_KEY
     TICK the checkbox next to each one to attach it
  4. Accept the Llama 3 license as amijackofalltrades:
     huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
  5. Paste the whole file into one cell:
       training/train_founder_lora_kaggle.py
  6. Save Version -> "Save & Run All (Commit)"   <- background execution.
     The interactive Run All button dies when you close the tab.
  7. Verify the adapter landed before starting the next run:
       python -c "from huggingface_hub import HfApi; import os; \\
         from dotenv import load_dotenv; load_dotenv('.env'); \\
         print(HfApi(token=os.getenv('HF_TOKEN')).list_repo_files(
           'amijackofalltrades/eka-founder-lora'))"

Then repeat for chanakya, gita, reflection. Watch session 1 to completion
first — unattached secrets and the unaccepted Llama license both surface
there, and they are the two most common failures.
{'=' * 64}
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch datasets, then publish")
    parser.add_argument("--once", action="store_true", help="check once and exit")
    parser.add_argument(
        "--min-frac",
        type=float,
        default=1.0,
        help="fraction of target that counts as done (default 1.0)",
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="preprocess and gate only"
    )
    args = parser.parse_args()

    print(f"watcher started — polling every {POLL_SECONDS}s, "
          f"target = {args.min_frac:.0%} of each persona")

    while True:
        rows, done, lines = snapshot(args.min_frac)
        stamp = time.strftime("%H:%M:%S")
        print(f"\n[{stamp}] dataset state:")
        for line in lines:
            print(line)

        if done:
            print("\n✓ every persona has reached target — running the pipeline")
            break
        if args.once:
            print("\n--once: not yet complete, exiting without publishing")
            return 1
        if not queue_running():
            print(
                "\n⚠ no python process left but datasets are short — the queue "
                "stopped early (most likely a provider daily wall).\n"
                "  Re-run:  python ml/scripts/run_queue.py --all\n"
                "  Not publishing a partial dataset."
            )
            return 2
        time.sleep(POLL_SECONDS)

    if not run("PREPROCESS", [PYTHON, "ml/scripts/preprocess.py"]):
        print("preprocess failed — stopping before upload")
        return 1

    if not quality_gate():
        return 2

    if args.skip_upload:
        print("\n--skip-upload: splits built and gated, nothing pushed")
        return 0

    if not run("UPLOAD TO HUGGING FACE", [PYTHON, "ml/scripts/upload_to_hf.py"]):
        print("upload failed — check HF_TOKEN and network")
        return 1

    kaggle_instructions()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
