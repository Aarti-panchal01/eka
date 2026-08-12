"""A trailing comma must not throw away a whole batch of pairs.

Found 2026-08-13 02:55, and only because the unparseable-reply log had just
started printing both ends of the failing text. Every sampled failure ended
the same way:

    ...What is the one move available to you right now?", } ]

A trailing comma before the closing brace. That is invalid strict JSON, so
json.loads rejects the entire array, and the balanced-object fallback re-parses
the same text and fails identically — so one stray comma discarded all five
pairs in a call. It accounted for most of the ~40% of generation calls that
were being thrown away (184 unparseable against ~256 successful batches), and
none of the replies were truncated: they ran 1,900-5,800 chars and closed
cleanly with `]` and a fence.

The repair has to be string-aware. These are long natural-language responses
full of commas, and a regex like `,(\\s*[}\\]])` would rewrite the ones inside
them.

    python tests/test_extract_pairs.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml" / "scripts"))

from _gen_async import _drop_trailing_commas, extract_pairs  # noqa: E402

# Verbatim shape from ml/datasets/queue_run.log.
REAL_FAILURE = """```json
[
  { "user": "My name is Arjun Mehta. I run a 12-person textile export firm in Surat, and last quarter our largest client walked.",
    "eka_response": "You are describing the loss as if it were an accident. The alternative fails. What is the one move available to you right now?", },
  { "user": "Rajiv Mehta, 34, Mumbai. My startup's lead investor pulled out, and I found out from a journalist.",
    "eka_response": "Being told by a stranger is its own injury, separate from the money. Which one are you actually angry about?", }
]
```"""

CLEAN = """[
  {"user": "a real question here", "eka_response": "a real answer here"}
]"""


def main() -> int:
    failures = []

    # 1. The batch that used to be discarded entirely.
    pairs = extract_pairs(REAL_FAILURE, 5)
    if len(pairs) != 2:
        failures.append(f"real failure: expected 2 pairs recovered, got {len(pairs)}")
    else:
        print(f"  ok  trailing-comma batch recovered: {len(pairs)} pairs")

    # 2. Commas inside the prose must survive — that is the whole risk of the fix.
    if pairs and "Surat, and last quarter" not in pairs[0]["user"]:
        failures.append("a comma inside a user message was rewritten")
    elif pairs:
        print("  ok  commas inside prose left intact")

    # 3. Valid JSON must be untouched.
    if len(extract_pairs(CLEAN, 5)) != 1:
        failures.append("clean JSON no longer parses")
    else:
        print("  ok  well-formed replies unaffected")

    # 4. The transformation itself, including a string that CONTAINS "},".
    cases = [
        ('{"a": "x, y", "b": "p, q"}', '{"a": "x, y", "b": "p, q"}'),
        ('[{"a": 1,}]', '[{"a": 1}]'),
        ('[{"a": "ends with },"},]', '[{"a": "ends with },"}]'),
        ('[{"a": 1} , ]', '[{"a": 1}  ]'),
        ('{"a": "no change"}', '{"a": "no change"}'),
    ]
    for src, expected in cases:
        got = _drop_trailing_commas(src)
        if got != expected:
            failures.append(f"_drop_trailing_commas({src!r}) -> {got!r}, want {expected!r}")
    if not any("_drop_trailing_commas" in f for f in failures):
        print(f"  ok  string-aware across {len(cases)} shapes")

    # 5. Whatever it returns must be loadable — the point is valid JSON, not
    #    merely different text.
    repaired = _drop_trailing_commas(REAL_FAILURE[REAL_FAILURE.find("["):REAL_FAILURE.rfind("]") + 1])
    try:
        json.loads(repaired)
        print("  ok  repaired text is valid JSON")
    except json.JSONDecodeError as exc:
        failures.append(f"repaired text still not valid JSON: {exc}")

    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print("all extract_pairs assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
