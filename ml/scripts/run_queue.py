"""Run the persona generators in sequence, resumably, across daily quotas.

    python ml/scripts/run_queue.py                      # chanakya, gita, reflection
    python ml/scripts/run_queue.py --all                # include founder
    python ml/scripts/run_queue.py --wait-for-pid 14560 # start after that PID exits
    python ml/scripts/run_queue.py --status             # what is done, no API calls

SEQUENTIAL ON PURPOSE. All personas draw on the same provider quotas, so running
them concurrently would contend for one tokens-per-minute window and 429 each
other — the rotator's pacers are per-provider, not per-process. One at a time is
both faster in wall-clock and cheaper in wasted tokens.

DESIGNED TO BE RE-RUN. The total need is ~2.4M tokens against a combined daily
ceiling near 1.1M, so this cannot finish in one day. Every generator is
resume-safe (state lives in its dataset JSON), so:

  * a persona already at target is skipped without a single API call
  * a persona stopped mid-way by a quota wall resumes where it left off
  * hitting the wall stops the QUEUE rather than marching on, because the next
    persona would only rediscover the same empty quota and burn requests doing it

So the intended usage is: run it, let it stop, run it again tomorrow. Nothing
needs to be tracked by hand between days.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# The child generators emit ✓ ↻ ⇄ ⏳ →, and this process relays every one of
# their lines to its own stdout. A stock Windows console is cp1252, where that
# relay raises UnicodeEncodeError and kills the QUEUE — not just the line —
# taking a running generation down with it. Measured 2026-08-12: a restart died
# on '→' four seconds in, after the child had already started work.
#
# The subprocess pipe below is explicitly utf-8/replace already; this gives the
# relay target the same treatment so the two ends match. errors="replace" rather
# than "ignore" so an unrenderable glyph shows as ? instead of vanishing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS = Path(__file__).resolve().parent
DATASETS = SCRIPTS.parent / "datasets"
PROJECT = SCRIPTS.parent.parent
PYTHON = sys.executable

# Order matters: reflection is last because it is the only persona that pays for
# an LLM judge call per surviving pair, so it is the most expensive per pair.
QUEUE = [
    ("chanakya", "generate_chanakya_data.py", "chanakya_dataset.json"),
    ("gita", "generate_gita_data.py", "gita_dataset.json"),
    ("reflection", "generate_reflection_data.py", "reflection_dataset.json"),
]
FOUNDER = ("founder", "generate_founder_data.py", "founder_dataset.json")

# Phrases the generator prints when it stops because a provider ran out rather
# than because it finished. Matching on these keeps "out of quota" distinct from
# "this persona is broken" — the first means come back tomorrow, the second
# means stop and look.
QUOTA_MARKERS = (
    "out of quota for today",
    "every configured provider is exhausted",
    "daily-quota wall",
    "reached its tracked daily limit",
    "DAILY quota wall",
)


def target_for(module: str) -> int:
    """Read a persona's target without importing its whole dependency chain."""
    text = (SCRIPTS / module).read_text(encoding="utf-8")
    inside = False
    total = 0
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


def count_on_disk(dataset: str) -> int:
    try:
        with open(DATASETS / dataset, "r", encoding="utf-8") as handle:
            return len(json.load(handle))
    except Exception:
        return 0


def pid_alive(pid: int) -> bool:
    """Windows-safe liveness check without extra dependencies."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        return str(pid) in out
    except Exception:
        # If we cannot tell, assume it is gone rather than block the queue
        # forever on a check that will never succeed.
        return False


def wait_for(pid: int) -> None:
    print(f"→ waiting for PID {pid} to finish before starting the queue")
    waited = 0
    while pid_alive(pid):
        time.sleep(20)
        waited += 20
        if waited % 600 == 0:
            print(f"   still waiting ({waited // 60} min)")
    print(f"→ PID {pid} has exited; starting queue\n")


def status() -> int:
    print(f"\n{'persona':12} {'on disk':>9} {'target':>8}  state")
    print("-" * 52)
    everything = [FOUNDER] + QUEUE
    for name, module, dataset in everything:
        have = count_on_disk(dataset)
        want = target_for(module)
        state = "complete" if have >= want else f"{want - have} short"
        print(f"{name:12} {have:>9} {want:>8}  {state}")
    usage = DATASETS / ".provider_usage.json"
    if usage.exists():
        print("\nspent today:")
        data = json.loads(usage.read_text(encoding="utf-8"))
        for day, providers in data.items():
            for provider, spent in sorted(providers.items()):
                # The rotator now records {"tokens": N, "requests": N} per
                # provider, because on some free tiers requests bind first
                # (Google: 20/day). Older files hold a bare token count.
                if isinstance(spent, dict):
                    tokens = spent.get("tokens", 0)
                    requests = spent.get("requests", 0)
                    print(f"  {provider:12} {tokens:>10,} tok  {requests:>5} req")
                else:
                    print(f"  {provider:12} {spent:>10,} tok")
    print()
    return 0


def run_one(name: str, module: str, dataset: str) -> str:
    """Run one generator. Returns 'complete' | 'quota' | 'short' | 'error'."""
    have = count_on_disk(dataset)
    want = target_for(module)
    if have >= want:
        print(f"\n=== {name}: already at target ({have}/{want}) — skipping\n")
        return "complete"

    print(f"\n{'=' * 64}")
    print(f"  {name}: {have}/{want} on disk, generating {want - have} more")
    print(f"{'=' * 64}\n", flush=True)

    log_path = DATASETS / f"{name}_run.log"
    captured = []
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            [PYTHON, "-u", f"ml/scripts/{module}"],
            cwd=str(PROJECT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            captured.append(line)
        process.wait()

    blob = "".join(captured)
    have = count_on_disk(dataset)
    if any(marker in blob for marker in QUOTA_MARKERS) and have < want:
        return "quota"
    if have >= want:
        return "complete"
    if process.returncode not in (0, 2):
        return "error"
    return "short"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Eka data generation in sequence")
    parser.add_argument("--all", action="store_true", help="include founder")
    parser.add_argument("--wait-for-pid", type=int, default=0,
                        help="block until this PID exits, then start")
    parser.add_argument("--status", action="store_true",
                        help="print progress and exit, no API calls")
    args = parser.parse_args()

    if args.status:
        return status()

    if args.wait_for_pid:
        wait_for(args.wait_for_pid)

    queue = ([FOUNDER] + QUEUE) if args.all else list(QUEUE)
    results = {}
    for name, module, dataset in queue:
        outcome = run_one(name, module, dataset)
        results[name] = outcome
        if outcome == "quota":
            print(
                f"\n{'=' * 64}\n"
                f"  STOPPING: {name} ran out of provider quota for today.\n"
                f"  Progress is saved. Re-run this script tomorrow and it will\n"
                f"  resume {name} and continue down the queue.\n"
                f"    python ml/scripts/run_queue.py\n"
                f"{'=' * 64}"
            )
            break
        if outcome == "error":
            print(
                f"\n  STOPPING: {name} failed for a reason that is not quota.\n"
                f"  See ml/datasets/{name}_run.log before re-running."
            )
            break

    print(f"\n{'=' * 64}\n  QUEUE SUMMARY\n{'=' * 64}")
    for name, _module, dataset in ([FOUNDER] + QUEUE):
        have = count_on_disk(dataset)
        want = target_for(_module)
        mark = {"complete": "✅", "quota": "⏸", "short": "⚠", "error": "❌"}.get(
            results.get(name, ""), "·"
        )
        print(f"  {mark} {name:12} {have:>5}/{want:<5} {results.get(name, 'not run')}")
    print()
    status()
    # 0 only when every queued persona actually reached its target.
    return 0 if all(v == "complete" for v in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
