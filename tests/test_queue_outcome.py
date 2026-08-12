"""A finished generator run must not stop the queue unless it really has to.

Regression test for the 2026-08-12 23:47 stall. founder finished its planned
work at 999/1000 — one pair kept failing regeneration — with all four Mistral
keys healthy (46/42/34/28 successful calls). run_queue classified that as
"quota" and broke out of the loop, so chanakya, gita and reflection never
started: 2,200 pairs blocked by one pair, none of which shared the problem.

The cause was a substring scan for five phrases, four of which are printed on
the normal healthy path every time a single secondary provider parks.

    python tests/test_queue_outcome.py
"""

import importlib.util
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "ml" / "scripts"))

spec = importlib.util.spec_from_file_location(
    "run_queue", PROJECT / "ml" / "scripts" / "run_queue.py"
)
rq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rq)

# Verbatim from ml/datasets/queue_run.log, the run that triggered this test.
ROUTINE_PARKING = """\
  ⇄ groq hit a DAILY quota wall — parking it and rotating
  ⇄ openrouter hit a DAILY quota wall — parking it and rotating
  ⇄ google hit a DAILY quota wall — parking it and rotating
  ⇄ mistral-1 reached its tracked daily limit (94,663/6,000,000) — rotating
✓ founder: 998/1000 (+1 kept, 0 rejected) | pivot decision
↻ founder: 999/1000 (+1 recovered, 1 still failing) | investor relationship
  per-provider outcomes:
    groq       calls=   0 EXHAUSTED
    mistral-1  calls=  46 accept=  53%
"""

# What the generator prints when it genuinely cannot call anything.
REAL_WALL = ROUTINE_PARKING + """
Every configured provider failed or is out of quota for today.
Progress is saved — re-run to resume.
"""

CASES = [
    # (name, blob, have, want, returncode, expected)
    (
        "THE BUG: short by one, secondary providers parked, keys healthy",
        ROUTINE_PARKING, 999, 1000, 0, "short",
    ),
    (
        "genuine wall: generator said it has nothing left to call",
        REAL_WALL, 640, 1000, 0, "quota",
    ),
    (
        "at target, with all the usual parking chatter present",
        ROUTINE_PARKING, 1000, 1000, 0, "complete",
    ),
    (
        "over target still counts as complete",
        ROUTINE_PARKING, 1002, 1000, 0, "complete",
    ),
    (
        "at target even if the generator also hit a real wall at the end",
        REAL_WALL, 1000, 1000, 0, "complete",
    ),
    (
        "crashed: non-zero, non-2 exit with no wall message",
        ROUTINE_PARKING, 400, 1000, 1, "error",
    ),
    (
        "exit 2 is the generator's own clean 'stopped short' code, not an error",
        ROUTINE_PARKING, 400, 1000, 2, "short",
    ),
]

# Only these two may halt the queue. This is the property that actually matters:
# everything else must let the remaining personas run.
STOPS_QUEUE = {"quota", "error"}


def main() -> int:
    failures = []
    for name, blob, have, want, code, expected in CASES:
        got = rq.classify(blob, have, want, code)
        if got != expected:
            failures.append(f"{name}\n          expected {expected!r}, got {got!r}")
        else:
            halt = "stops queue" if got in STOPS_QUEUE else "queue continues"
            print(f"  ok  {got:9} ({halt:15}) {name}")

    # The specific regression, stated as the consequence rather than the label:
    # a one-pair shortfall must never block the three personas behind it.
    if rq.classify(ROUTINE_PARKING, 999, 1000, 0) in STOPS_QUEUE:
        failures.append("a 1-pair shortfall still halts the queue")

    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print("all queue-outcome assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
