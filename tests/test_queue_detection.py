"""watch_and_publish must notice when the queue dies.

Regression test for the 2026-08-12 23:20 blind spot: queue_running() asked
tasklist whether any python.exe existed. The watcher IS a python.exe, so the
answer was always yes, and the "queue stopped early" branch could never run.
The queue died at 23:20 and the watcher polled on for 40 minutes reporting
nothing wrong.

The bug is invisible to a live test — the broken version and the fixed version
both say True while the queue is up. Only the negative case separates them, so
that is what this drives.

    python tests/test_queue_detection.py
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "watch_and_publish", PROJECT / "scripts" / "watch_and_publish.py"
)
wp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wp)


class FakeCompleted:
    def __init__(self, stdout):
        self.stdout = stdout
        self.returncode = 0


def with_processes(stdout, raises=False):
    """Run queue_running() against a controlled process listing."""
    original = subprocess.run

    def fake(*args, **kwargs):
        if raises:
            raise OSError("powershell unavailable")
        return FakeCompleted(stdout)

    wp.subprocess.run = fake
    try:
        return wp.queue_running()
    finally:
        wp.subprocess.run = original


PY = r'"C:\Python313\python.exe"'

CASES = [
    (
        "queue alive",
        f"{PY} -u ml/scripts/run_queue.py --all\n"
        f"{PY} -u ml/scripts/generate_founder_data.py\n"
        f"{PY} -u scripts/watch_and_publish.py\n",
        True,
    ),
    (
        # THE CASE THAT WAS BROKEN. Watcher and backend still up, queue gone.
        # The old "any python.exe" check returned True here and hid the failure.
        "queue dead, watcher and backend still up",
        f"{PY} -u scripts/watch_and_publish.py\n"
        f"{PY} -m uvicorn main:app --host 127.0.0.1 --port 8091\n",
        False,
    ),
    (
        "a lone generator with no queue still counts as working",
        f"{PY} -u ml/scripts/generate_reflection_data.py\n",
        True,
    ),
    ("nothing running at all", "", False),
    ("only unrelated python", f"{PY} -m http.server\n", False),
]


def main() -> int:
    failures = []

    for name, listing, expected in CASES:
        got = with_processes(listing)
        if got != expected:
            failures.append(f"{name}: expected {expected}, got {got}")
        else:
            print(f"  ok  {name:48} -> {got}")

    # If the probe itself fails we must assume the queue is alive: delaying a
    # publish is recoverable, publishing a partial dataset into a multi-hour
    # GPU run is not.
    got = with_processes("", raises=True)
    if got is not True:
        failures.append(f"probe failure: expected conservative True, got {got}")
    else:
        print(f"  ok  {'probe itself fails -> assume alive':48} -> {got}")

    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print("all queue-detection assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
