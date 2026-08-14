#!/usr/bin/env python3
"""Measure whether any of this actually helps.

    python ml/eval/eval_harness.py                 # all available configs
    python ml/eval/eval_harness.py --n 20          # smaller/cheaper run
    python ml/eval/eval_harness.py --only gold,prompted

Writes ml/eval/results/ablation_table.{json,md}.

WHY THE PROMPTS COME FROM founder_val.jsonl, NOT "the last 50 rows"
-------------------------------------------------------------------
preprocess.py splits with `random.shuffle` under a fixed seed and stratifies by
topic, so the last 50 rows of founder_dataset.json are scattered across TRAIN.
Evaluating on them would be testing on training data and every number would be
inflated. founder_val.jsonl is the split that was actually held out, so that is
what this reads.

WHAT THE ROWS MEAN
------------------
  gold       the dataset's own responses. Not a model — this is the target the
             fine-tune is aiming at, and it calibrates the whole scale. If a
             config scores near gold, it has learned the format.
  prompted   base Qwen via Groq with the persona prompt. No retrieval.
  rag        the deployed backend: retrieval + LightGBM ranker + generation.
  finetuned  the LoRA adapter. Skipped until one exists.

Persona adherence is deliberately mechanical — four binary checks, no LLM
judge. A judge would be more nuanced and would also need its own validation
before any number it produced meant anything.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
VAL_SPLIT = ROOT / "ml" / "data" / "splits" / "founder_val.jsonl"
PERSONA_PROMPT = ROOT / "backend" / "prompts" / "founder.txt"

API = os.environ.get("EKA_API", "https://eka-backend-doau.onrender.com")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_GEN_MODEL", "llama-3.3-70b-versatile")

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------- scoring
# Frameworks a founder-mode reply is supposed to reach for. Matched on word
# boundaries so "CAC" does not fire inside another word.
FRAMEWORKS = re.compile(
    r"\b(lean startup|product[- ]market fit|pmf|burn(?: rate)?|runway|"
    r"first principles|unit economics|jobs[- ]to[- ]be[- ]done|jtbd|"
    r"cac|ltv|churn|moat|traction|cohort)\b",
    re.I,
)
# Hedges. "might consider" is the giveaway phrase for a model refusing to
# commit, which is the exact failure the founder persona exists to avoid.
HEDGES = re.compile(
    r"\b(maybe|perhaps|might consider|you could try|it depends|possibly|"
    r"i'?m not sure|potentially)\b",
    re.I,
)
WORD_FLOOR, WORD_CEIL = 150, 250


def score(text: str) -> Dict:
    body = (text or "").strip()
    words = len(body.split())
    checks = {
        "ends_with_question": int(body.endswith("?")),
        "uses_framework": int(bool(FRAMEWORKS.search(body))),
        "no_hedging": int(not HEDGES.search(body)),
        "word_count_in_band": int(WORD_FLOOR <= words <= WORD_CEIL),
    }
    checks["persona_score"] = sum(checks.values())
    checks["words"] = words
    return checks


# ------------------------------------------------------------------ data
_TURN = re.compile(
    r"<\|im_start\|>user\s*(.*?)<\|im_end\|>.*?"
    r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>",
    re.S,
)


def load_heldout(n: int) -> List[Dict]:
    if not VAL_SPLIT.exists():
        raise SystemExit(
            f"{VAL_SPLIT} missing. Run: python ml/scripts/preprocess.py"
        )
    rows = []
    for line in VAL_SPLIT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = _TURN.search(json.loads(line)["text"])
        if match:
            rows.append(
                {"user": match.group(1).strip(), "gold": match.group(2).strip()}
            )
        if len(rows) >= n:
            break
    if not rows:
        raise SystemExit("Could not parse any turns out of the val split.")
    return rows


# --------------------------------------------------------------- runners
def run_gold(item: Dict) -> Dict:
    """No model call — the reference answer, as an upper bound."""
    return {"text": item["gold"], "latency_ms": 0, "memories": None}


def run_prompted(item: Dict) -> Dict:
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    persona = PERSONA_PROMPT.read_text(encoding="utf-8").strip()
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": persona},
            {"role": "user", "content": item["user"]},
        ],
        "temperature": 0.7,
        "max_tokens": 700,
    }
    req = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "eka-eval/1.0",
        },
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode())
    return {
        "text": body["choices"][0]["message"]["content"],
        "latency_ms": int((time.time() - start) * 1000),
        "memories": None,
    }


def run_rag(item: Dict, user_id: str) -> Dict:
    """The deployed pipeline: retrieval + ranker + generation."""
    payload = {
        "message": item["user"],
        "user_id": user_id,
        "session_id": None,  # fresh session per prompt: no history leakage
        "mode": "founder",
        "language": "en-IN",
    }
    req = urllib.request.Request(
        f"{API}/chat/send",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    return {
        "text": body.get("response", ""),
        "latency_ms": int((time.time() - start) * 1000),
        "memories": len(body.get("retrieved_memories") or []),
    }


# ------------------------------------------------------------------ main
CONFIGS = {
    "gold": ("Training data (gold)", "reference upper bound, not a model"),
    "prompted": ("Base + prompt, no RAG", "Groq, persona prompt only"),
    "rag": ("RAG + ranker (deployed)", "vector search + LightGBM rerank"),
    "finetuned": ("Fine-tuned + RAG + ranker", "LoRA adapter"),
}


def evaluate(name: str, items: List[Dict], user_id: str) -> Optional[Dict]:
    runner = {
        "gold": run_gold,
        "prompted": run_prompted,
        "rag": lambda i: run_rag(i, user_id),
    }.get(name)
    if runner is None:
        print(f"  {name}: SKIPPED — no adapter trained yet")
        return None

    rows, failures = [], 0
    for index, item in enumerate(items, 1):
        try:
            out = runner(item)
        except Exception as exc:
            failures += 1
            print(f"    [{index}/{len(items)}] failed: {type(exc).__name__}: {exc}")
            continue
        rows.append({**score(out["text"]), **{
            "latency_ms": out["latency_ms"], "memories": out["memories"]}})
        if index % 10 == 0:
            print(f"    [{index}/{len(items)}]")

    if not rows:
        return None

    mean = lambda k: round(statistics.mean(r[k] for r in rows), 4)  # noqa: E731
    return {
        "config": name,
        "label": CONFIGS[name][0],
        "n": len(rows),
        "failures": failures,
        "persona_adherence": round(mean("persona_score") / 4, 4),
        "persona_score_mean": mean("persona_score"),
        "ends_with_question": mean("ends_with_question"),
        "uses_framework": mean("uses_framework"),
        "no_hedging": mean("no_hedging"),
        "word_count_in_band": mean("word_count_in_band"),
        "words_mean": round(mean("words"), 1),
        "latency_ms_median": int(statistics.median(r["latency_ms"] for r in rows)),
        # Whether those memories were RELEVANT needs labels, which is what the
        # implicit-feedback logging in chat.py starts collecting. Until then
        # this is retrieval volume, not precision, and is reported as such.
        "memories_retrieved_mean": (
            round(statistics.mean(r["memories"] for r in rows), 2)
            if rows[0]["memories"] is not None
            else None
        ),
        "memory_precision_at_3": None,
    }


def to_markdown(results: List[Dict], n: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = [
        "# Evaluation results",
        "",
        f"_Generated {stamp} · n={n} held-out founder prompts from "
        "`founder_val.jsonl` (never seen in training)._",
        "",
        "## Persona adherence",
        "",
        "Four mechanical checks per response, averaged. `gold` is the dataset's "
        "own answers — the target, not a model.",
        "",
        "| Configuration | Persona score | Ends w/ question | Uses framework | No hedging | Words in band | Mean words |",
        "|---|---|---|---|---|---|---|",
    ]
    pct = lambda v: f"{v * 100:.0f}%"  # noqa: E731
    for r in results:
        out.append(
            f"| {r['label']} | **{r['persona_adherence'] * 100:.0f}%** "
            f"({r['persona_score_mean']:.2f}/4) | {pct(r['ends_with_question'])} "
            f"| {pct(r['uses_framework'])} | {pct(r['no_hedging'])} "
            f"| {pct(r['word_count_in_band'])} | {r['words_mean']:.0f} |"
        )
    for name, (label, _) in CONFIGS.items():
        if not any(r["config"] == name for r in results):
            out.append(f"| {label} | _pending_ | — | — | — | — | — |")

    out += [
        "",
        "## RAG ablation",
        "",
        "| Configuration | Memories retrieved | Precision@3 | Median latency |",
        "|---|---|---|---|",
    ]
    for r in results:
        mem = r["memories_retrieved_mean"]
        out.append(
            f"| {r['label']} | {'—' if mem is None else mem} | "
            f"{'_needs labels_' if r['memory_precision_at_3'] is None else r['memory_precision_at_3']} | "
            f"{r['latency_ms_median']} ms |"
        )

    out += [
        "",
        "### Method",
        "",
        "- **Prompts** — held-out validation split, never trained on.",
        "- **Scoring** — four binary checks: ends with `?`; mentions a startup "
        "framework; contains no hedge phrase; 150-250 words. Deliberately "
        "mechanical, so it is reproducible and needs no judge model.",
        "- **Precision@3** is unfilled on purpose. Counting retrieved memories "
        "is not the same as knowing they were useful, and inventing a number "
        "there would be worse than leaving it empty.",
        "",
        "Reproduce: `python ml/eval/eval_harness.py`",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--only", default="", help="comma-separated config names")
    args = parser.parse_args()

    wanted = [c.strip() for c in args.only.split(",") if c.strip()] or list(CONFIGS)
    items = load_heldout(args.n)
    user_id = f"eval-{int(time.time())}"
    print(f"{len(items)} held-out prompts · configs: {', '.join(wanted)}\n")

    results = []
    for name in wanted:
        if name not in CONFIGS:
            print(f"  unknown config: {name}")
            continue
        print(f"  {name} …")
        row = evaluate(name, items, user_id)
        if row:
            results.append(row)
            print(
                f"    persona {row['persona_adherence'] * 100:.0f}% "
                f"· {row['latency_ms_median']}ms median"
            )

    if not results:
        print("\nNothing ran. Check GROQ_API_KEY / backend reachability.")
        return 1

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ablation_table.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n": len(items),
                "source": "founder_val.jsonl (held out)",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (RESULTS / "ablation_table.md").write_text(
        to_markdown(results, len(items)), encoding="utf-8"
    )
    print(f"\nwrote {RESULTS / 'ablation_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
