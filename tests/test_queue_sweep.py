"""The queue retries a persona that ends a few pairs short — once, and narrowly.

founder sat at 999/1000 across several runs tonight. That is structural, not
rare: as a topic fills, each remaining pair must be about that topic AND unlike
every pair already there, so the dedup gate rejects the last one far more often
than the first hundred. Downstream it is a hard block — any shortfall makes the
quality verdict needs_review, and watch_and_publish will not publish a persona
that is not ok_to_train. One missing pair at 03:00 means an empty Hub at 07:00.

What must hold:
  * a near-miss gets exactly one more attempt
  * a persona that lost its providers does NOT (same wall, wasted quota)
  * a real quota or error stop suppresses the sweep entirely
  * the sweep never loops

    python tests/test_queue_sweep.py
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

TARGETS = {"founder": 1000, "chanakya": 600, "gita": 600, "reflection": 1000}


def drive(first_pass, on_disk, second_pass_gain=1):
    """Run main()'s sweep with run_one and the disk stubbed out.

    Returns the list of personas that got a second attempt.
    """
    retried = []
    state = dict(on_disk)

    def fake_run_one(name, module, dataset):
        retried.append(name)
        state[name] += second_pass_gain
        return "complete" if state[name] >= TARGETS[name] else "short"

    rq.run_one = fake_run_one
    rq.count_on_disk = lambda dataset: state[dataset]
    rq.target_for = lambda module: TARGETS[module]
    rq.status = lambda: 0

    # queue entries are (name, module, dataset); stub them so all three are the
    # persona name, which is what the fakes above key on.
    rq.FOUNDER = ("founder", "founder", "founder")
    rq.QUEUE = [(n, n, n) for n in ("chanakya", "gita", "reflection")]

    sys.argv = ["run_queue", "--all"]

    # Pre-seed results as though the main loop had already run.
    original_run_one = fake_run_one
    calls = {"n": 0}

    def main_loop_run_one(name, module, dataset):
        calls["n"] += 1
        if calls["n"] <= len(first_pass):
            return first_pass[name]
        return original_run_one(name, module, dataset)

    rq.run_one = main_loop_run_one
    rq.main()
    # Everything after the first-pass results is a sweep attempt.
    return retried


def main() -> int:
    failures = []

    # 1. The founder-999 case: one pair short, must be retried exactly once.
    retried = drive(
        first_pass={"founder": "short", "chanakya": "complete",
                    "gita": "complete", "reflection": "complete"},
        on_disk={"founder": 999, "chanakya": 600, "gita": 600, "reflection": 1000},
    )
    if retried != ["founder"]:
        failures.append(f"near-miss: expected exactly ['founder'], got {retried}")
    else:
        print("  ok  999/1000 retried once")

    # 2. A persona well short of target must ALSO be retried, and bounded.
    #    An earlier version of this test asserted the opposite, because the
    #    sweep was sized on founder's 1-pair case before chanakya finished.
    #    chanakya then came in at 529/600 — 71 short, from a 0.42 rejection
    #    rate with 0.34 regeneration recovery, with an otherwise spotless
    #    quality report. A 10-pair gate would have skipped the exact case the
    #    sweep exists to rescue.
    retried = drive(
        first_pass={"founder": "complete", "chanakya": "short",
                    "gita": "complete", "reflection": "complete"},
        on_disk={"founder": 1000, "chanakya": 529, "gita": 600, "reflection": 1000},
    )
    if retried != ["chanakya"] * rq.MAX_TOPUP_SWEEPS:
        failures.append(
            f"far-short: expected {rq.MAX_TOPUP_SWEEPS} bounded retries, got {retried}"
        )
    else:
        print(f"  ok  529/600 retried, bounded at {rq.MAX_TOPUP_SWEEPS} sweeps")

    # 3. A sweep that closes the gap stops early instead of using its budget.
    retried = drive(
        first_pass={"founder": "complete", "chanakya": "short",
                    "gita": "complete", "reflection": "complete"},
        on_disk={"founder": 1000, "chanakya": 599, "gita": 600, "reflection": 1000},
        second_pass_gain=1,          # one pass reaches 600
    )
    if retried != ["chanakya"]:
        failures.append(f"convergence: expected one retry then stop, got {retried}")
    else:
        print("  ok  gap closed on the first sweep, remaining budget unused")

    # 3. A genuine quota stop suppresses the sweep entirely, even for a
    #    persona that happens to be one short.
    retried = drive(
        first_pass={"founder": "short", "chanakya": "quota",
                    "gita": "complete", "reflection": "complete"},
        on_disk={"founder": 999, "chanakya": 10, "gita": 600, "reflection": 1000},
    )
    if retried:
        failures.append(f"quota stop: expected no sweep, got {retried}")
    else:
        print("  ok  quota stop suppresses the sweep")

    # 5. A pass that gains nothing stops the loop immediately. Without this,
    #    a genuinely stuck persona burns every remaining sweep proving it.
    retried = drive(
        first_pass={"founder": "short", "chanakya": "short",
                    "gita": "complete", "reflection": "complete"},
        on_disk={"founder": 998, "chanakya": 598, "gita": 600, "reflection": 1000},
        second_pass_gain=0,          # retry closes nothing
    )
    if sorted(retried) != ["chanakya", "founder"]:
        failures.append(
            f"no-progress: expected one pass over both then stop, got {retried}"
        )
    else:
        print("  ok  a sweep that gains nothing stops the loop")

    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print("all queue-sweep assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
