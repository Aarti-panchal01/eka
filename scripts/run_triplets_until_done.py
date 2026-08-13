"""Keep the triplet generator running until it reaches target.

The generator itself is already resume-safe and correct. What killed the
2026-08-13 run was nothing in this repo: Windows Update rebooted the machine
at 07:42 (TrustedInstaller-initiated restart, boot at 07:44) while the run was
at 847/6000. stderr was empty because there was no crash — the process was
terminated by the shutdown.

So the missing piece is not error handling, it is a thing that notices the run
is gone and starts it again. This supervisor is that thing. Paired with the
`eka-triplets` scheduled task (AtLogOn trigger), a reboot now costs the time
between the reboot and the next login, not the whole run.

    python scripts/run_triplets_until_done.py [--target 6000]

Stops when the target is reached, or after MAX_STALLED consecutive passes that
add nothing — a run that cannot make progress should stop and say so rather
than spin against a walled provider until Saturday.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = PROJECT_ROOT / "ml" / "scripts" / "generate_embedding_triplets.py"
OUTPUT = PROJECT_ROOT / "ml" / "datasets" / "embedding_triplets.jsonl"
LOG = PROJECT_ROOT / "ml" / "datasets" / "triplets_supervisor.log"
# Run detached, the generator has no console to print to, and its own output is
# where the rate-limit messages live. Losing them is how this morning's failure
# stayed unexplained for three hours — so tee the child into the run log.
CHILD_LOG = PROJECT_ROOT / "ml" / "datasets" / "triplets_run.log"

# A pass that gains nothing usually means every provider is walled. Three in a
# row is the difference between a transient 429 and an actual daily ceiling.
MAX_STALLED = 3
PAUSE_BETWEEN_PASSES = 60.0

# The generator's default pacing (~14.7s/call) is sized for 70B generation
# calls of ~3,800 tokens. A paraphrase is ~200 tokens on an 8B model, so that
# pacing would spend 13 hours on a 2.5 hour job. Set here rather than left to
# the caller: across a reboot there is no caller to remember it.
#
# 2.0, measured on 2026-08-13 after paraphrasing moved off Groq onto the
# Mistral pool. Two-minute samples against the live pool:
#
#     GEN_SLEEP=0.5  ->  12/min, but 12 rate-limit hits and throttle waves
#     GEN_SLEEP=2.0  ->  10/min, and ZERO rate limits or waves
#
# 0.5 is nominally 20% faster and is not worth it. The waves are 5-30s dead
# stops, the variance is much worse, and Mistral is the same pool the persona
# generators depend on — pushing it toward a daily wall to save twelve minutes
# on a two-hour job is the trade this repo keeps learning not to make. Override
# with GEN_SLEEP if you want the extra 20%.
#
# For contrast, on Groq's free 8B tier this job ran at 1.5/min: 2.5s asked for
# ~24 req/min against a ~6,000 TPM ceiling and collapsed under constant 429s,
# and 5.0s settled at the same place. The ceiling was the provider, not the
# pacing, which is why the fix was to change providers rather than keep tuning.
GEN_SLEEP = os.environ.get("GEN_SLEEP", "2.0")

TASK_NAME = "eka-triplets"


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def count() -> int:
    """Triplets currently on disk. The generator flushes per line, so this is
    accurate even against a run that was killed mid-write."""
    if not OUTPUT.exists():
        return 0
    with OUTPUT.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def unregister_task() -> None:
    """Once the target is reached, stop firing at every login."""
    if os.name != "nt":
        return
    result = subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log(f"scheduled task {TASK_NAME} removed — nothing left to resume")


def main(target: int) -> int:
    log(f"supervisor up — target {target}, {count()} on disk, GEN_SLEEP={GEN_SLEEP}")

    # PYTHONUNBUFFERED because stdout to a file is block-buffered: without it
    # the log lags ~8KB behind the run, which is useless for "is it alive?".
    env = {
        **os.environ,
        "GEN_SLEEP": GEN_SLEEP,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    stalled = 0
    passes = 0

    while True:
        have = count()
        if have >= target:
            log(f"target reached: {have}/{target}")
            unregister_task()
            log("next: python ml/scripts/upload_to_hf.py && "
                "python scripts/preflight_kaggle.py --hub")
            return 0

        if stalled >= MAX_STALLED:
            log(f"! {MAX_STALLED} passes in a row gained nothing — stopping at "
                f"{have}/{target}. Check provider state: "
                f"python ml/scripts/_gen_providers.py --check")
            return 1

        passes += 1
        log(f"pass {passes}: starting generator at {have}/{target}"
            f" (output -> {CHILD_LOG.name})")
        with CHILD_LOG.open("a", encoding="utf-8") as child_out:
            child_out.write(
                f"\n===== pass {passes} started "
                f"{datetime.now():%Y-%m-%d %H:%M:%S} =====\n"
            )
            child_out.flush()
            result = subprocess.run(
                [sys.executable, str(GENERATOR), "--target", str(target)],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=child_out,
                stderr=subprocess.STDOUT,
            )

        gained = count() - have
        log(f"pass {passes} ended (exit {result.returncode}), gained {gained}")

        # Exit code is deliberately not the stall signal. The generator exits 0
        # when it stops early on its own, and a reboot-killed run has no exit
        # code worth reading. Progress on disk is the only honest measure.
        stalled = 0 if gained > 0 else stalled + 1

        if count() < target:
            time.sleep(PAUSE_BETWEEN_PASSES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=6000)
    sys.exit(main(parser.parse_args().target))
