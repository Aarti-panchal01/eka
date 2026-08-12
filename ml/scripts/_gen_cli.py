"""Shared command line for the four persona generators.

Each generate_<mode>_data.py declares only what is genuinely mode-specific —
topic quotas, the PersonaSpec, and the extra prompt instruction — and calls
run() here. Four copies of "score the file, print a plan, run the generator"
is precisely the duplication that drifts out of sync the first time one of them
gets a fix.

    python ml/scripts/generate_<mode>_data.py                # generate / resume
    python ml/scripts/generate_<mode>_data.py --report-only  # re-score, no API calls
    python ml/scripts/generate_<mode>_data.py --plan         # estimate, no API calls
"""

import argparse
import asyncio
import re
from typing import Dict

from _gen_async import DEFAULT_MODEL, PAIRS_PER_CALL, generate_dataset_async
from _gen_common import DATASETS_DIR, load_existing
from _gen_quality import (
    PersonaSpec,
    print_report,
    quality_report,
    revalidate_existing,
    write_report,
)

# Assumed reject rate for the --plan request estimate only. Real runs report
# their true rejection_rate in the quality report.
_ASSUMED_REJECT_RATE = 0.10
# Observed sustainable throughput on llama-3.3-70b-versatile with batch=5 and
# concurrency=3, after 429s and regeneration overhead. Used for ETA only.
_PAIRS_PER_HOUR = 600


def report_only(mode: str, output: str, report: str, topic_counts, spec) -> int:
    """Re-score what is on disk against the current spec. No API calls."""
    rows = load_existing(DATASETS_DIR / output)
    if not rows:
        print(f"No data at {DATASETS_DIR / output} — nothing to score.")
        return 1
    kept, dropped, _ = revalidate_existing(rows, spec)
    print(
        f"\n{len(rows)} pairs on disk: {len(kept)} pass the current spec, "
        f"{len(dropped)} fail"
    )
    if dropped:
        print("\n  most common failures:")
        counts: Dict[str, int] = {}
        for row in dropped:
            for reason in row["_reasons"]:
                # Normalise numbers out so "53 words, below the 100-word
                # minimum" and "61 words, below the ..." group together —
                # otherwise every distinct word count is its own row and the
                # summary is unreadable.
                key = re.sub(r"\b\d+(\.\d+)?\b", "N", reason.split("(")[0].strip())
                counts[key] = counts.get(key, 0) + 1
        for reason, count in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {count:>4}  {reason}")
    built = quality_report(kept, spec, topic_counts)
    write_report(DATASETS_DIR / report, built)
    print_report(built)
    return 0 if built["verdict"] == "ok_to_train" else 2


def plan(mode: str, output: str, topic_counts, spec) -> int:
    """Print the request/time estimate without calling the API."""
    rows, _, _ = revalidate_existing(load_existing(DATASETS_DIR / output), spec)
    target = sum(topic_counts.values())
    need = max(0, target - len(rows))
    batches = -(-need // PAIRS_PER_CALL)  # ceil
    regens = int(batches * _ASSUMED_REJECT_RATE * PAIRS_PER_CALL / PAIRS_PER_CALL) or 0
    regens = max(regens, int(batches * _ASSUMED_REJECT_RATE))
    cap = "1,000" if "70b" in DEFAULT_MODEL else "14,400"
    print(f"\n  mode           {mode}")
    print(f"  target         {target} pairs across {len(topic_counts)} topics")
    print(f"  usable on disk {len(rows)}")
    print(f"  to generate    {need}")
    print(f"  eka words      {spec.eka_words[0]}-{spec.eka_words[1]}")
    print(f"  user words     {spec.user_words[0]}-{spec.user_words[1]}")
    print(f"  batch size     {PAIRS_PER_CALL}")
    print(f"  batch calls    ~{batches}")
    print(f"  + regens       ~{regens} (assuming a {_ASSUMED_REJECT_RATE:.0%} reject rate)")
    print(f"  total requests ~{batches + regens}  (free tier: {cap}/day)")
    print(f"  model          {DEFAULT_MODEL}")
    print(f"  est. runtime   ~{need / _PAIRS_PER_HOUR:.1f}h at ~{_PAIRS_PER_HOUR} pairs/hr\n")
    return 0


def run(
    mode: str,
    output: str,
    report: str,
    topic_counts: Dict[str, int],
    spec: PersonaSpec,
    extra: str = "",
) -> int:
    parser = argparse.ArgumentParser(description=f"Generate Eka {mode} data")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="re-score existing data against the current spec, no API calls",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print the request/time estimate and exit, no API calls",
    )
    args = parser.parse_args()

    if args.report_only:
        return report_only(mode, output, report, topic_counts, spec)
    if args.plan:
        return plan(mode, output, topic_counts, spec)

    print(
        f"EKA data generation — {mode} "
        f"({sum(topic_counts.values())} pairs, quality-gated)\n"
    )
    built = asyncio.run(
        generate_dataset_async(
            mode=mode,
            topic_counts=topic_counts,
            output_name=output,
            spec=spec,
            extra_instruction=extra,
            report_name=report,
        )
    )
    if built.get("verdict") != "ok_to_train":
        print("⚠ This dataset is NOT cleared for training — see blocking issues above.")
        return 2
    return 0
