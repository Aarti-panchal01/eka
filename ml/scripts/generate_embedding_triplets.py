"""Build (anchor, positive, negative) triplets for the embedding fine-tune.

    python ml/scripts/generate_embedding_triplets.py [--target 6000]

Positives come from Groq paraphrases (one cheap call each, 8B model).
Negatives are free: a random user message from a *different* persona mode,
which is exactly the confusion we want the retriever to stop making.

Round 2+ augmentation pairs an anchor with the paraphrase of a different
message from its own mode — a harder positive that teaches topical rather
than lexical similarity.

Resume-safe: re-run after a crash and it continues from the existing file.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

from _gen_common import (
    DATASETS_DIR,
    FAST_MODEL,
    SLEEP_BETWEEN_CALLS,
    get_groq_client,
    paraphrase,
)

OUTPUT = DATASETS_DIR / "embedding_triplets.jsonl"
SOURCES = {
    "founder": "founder_dataset.json",
    "chanakya": "chanakya_dataset.json",
    "gita": "gita_dataset.json",
    "reflection": "reflection_dataset.json",
}
random.seed(42)


def load_pools() -> Dict[str, List[str]]:
    """mode -> list of user messages."""
    pools: Dict[str, List[str]] = {}
    for mode, filename in SOURCES.items():
        path = DATASETS_DIR / filename
        if not path.exists():
            print(f"! missing {path.name} — skipping {mode}")
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        messages = [r["user"].strip() for r in rows if r.get("user")]
        pools[mode] = messages
        print(f"  {mode:<11} {len(messages)} user messages")
    if not pools:
        raise SystemExit(
            "No datasets found in ml/datasets/.\n"
            "Run the generate_*_data.py scripts first."
        )
    return pools


def load_done() -> List[dict]:
    if not OUTPUT.exists():
        return []
    rows = []
    for line in OUTPUT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def pick_negative(pools: Dict[str, List[str]], own_mode: str) -> str:
    """A user message from any other mode — no API call needed."""
    other = [m for m in pools if m != own_mode and pools[m]]
    if not other:
        return random.choice(pools[own_mode])
    return random.choice(pools[random.choice(other)])


def main(target: int) -> None:
    print("EKA embedding triplets\n")
    pools = load_pools()

    anchors: List[tuple] = []
    for mode, messages in pools.items():
        anchors.extend((mode, msg) for msg in messages)
    random.shuffle(anchors)
    print(f"\n  {len(anchors)} total anchors, target {target} triplets")

    done = load_done()
    seen_anchors = {row["anchor"] for row in done}
    print(f"  {len(done)} triplets already on disk — resuming\n")

    client = get_groq_client()
    written = len(done)
    handle = OUTPUT.open("a", encoding="utf-8")

    try:
        round_no = 0
        while written < target:
            round_no += 1
            made_this_round = 0

            for mode, anchor in anchors:
                if written >= target:
                    break
                # Round 1: every anchor once. Later rounds: augmentation only.
                if round_no == 1 and anchor in seen_anchors:
                    continue

                if round_no == 1:
                    positive = paraphrase(client, anchor, model=FAST_MODEL)
                    time.sleep(SLEEP_BETWEEN_CALLS)
                else:
                    # Harder positive: a different message from the same mode.
                    same_mode = [m for m in pools[mode] if m != anchor]
                    positive = random.choice(same_mode) if same_mode else None

                if not positive:
                    continue

                record = {
                    "anchor": anchor,
                    "positive": positive,
                    "negative": pick_negative(pools, mode),
                    "mode": mode,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                seen_anchors.add(anchor)
                written += 1
                made_this_round += 1

                if written % 20 == 0:
                    print(f"✓ triplets: {written}/{target}")

            if made_this_round == 0:
                print("! no new triplets producible — stopping early")
                break
            print(f"  -- round {round_no} done, {written}/{target} --")
    finally:
        handle.close()

    print(f"\n✅ {written} triplets -> {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=6000)
    main(parser.parse_args().target)
