"""Shared machinery for every Eka data-generation script.

Handles the boring-but-critical parts once: .env loading, the Groq client,
rate limiting, resume-safe checkpointing, robust JSON extraction from model
output, and progress reporting.

Only the four persona generators + the triplet generator import this. The
Kaggle training notebooks stay fully self-contained on purpose.
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Windows consoles default to cp1252, which cannot encode ✓/✅/⏳/💾.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ----------------------------------------------------------------- paths
SCRIPTS_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = ML_DIR.parent
DATASETS_DIR = ML_DIR / "datasets"
PROMPTS_DIR = PROJECT_ROOT / "backend" / "prompts"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- .env load
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    # dotenv is optional; real env vars still work.
    pass

# PACING IS TOKEN-BOUND, NOT REQUEST-BOUND.
#
# The requests/minute ceiling is not what throttles this script. Checked live
# against the API, llama-3.3-70b-versatile returns:
#     x-ratelimit-limit-requests = 1000     (per day)
#     x-ratelimit-limit-tokens   = 12000    (per MINUTE)  <-- the real ceiling
#
# One pair costs roughly 2500 tokens (a ~1200-token persona system prompt, the
# user prompt, and up to max_tokens=800 out). So the token budget allows about
# 12000/2500 = 4.8 calls per minute — call it ~12.5s between calls.
#
# An earlier version paced at 2.5s/call on the assumption that 30 RPM was the
# limit. That asks for ~24 calls/min = ~60000 TPM against a 12000 TPM ceiling,
# i.e. 5x over budget, so essentially every call 429'd and throughput collapsed
# to single digits per hour. Do not lower GEN_SLEEP below ~12s for a 70B model
# without raising EST_TOKENS_PER_CALL to match; going "faster" here makes the
# run strictly slower.
EST_TOKENS_PER_CALL = int(os.environ.get("GEN_EST_TOKENS", "2500"))
TOKENS_PER_MINUTE = int(os.environ.get("GEN_TPM", "12000"))
# 0.85 leaves headroom for prompt-length variance; brushing the ceiling on
# every call trades a little throughput for a lot of backoff.
_paced = 60.0 / max(1.0, (TOKENS_PER_MINUTE * 0.85) / EST_TOKENS_PER_CALL)
SLEEP_BETWEEN_CALLS = float(os.environ.get("GEN_SLEEP", f"{_paced:.1f}"))

# Backoff on 429. These are the fallback for when the API does not tell us how
# long to wait — _retry_after_seconds() below prefers the server's own
# retry-after header, which is usually only a few seconds because the token
# bucket refills every minute. Blind 4-minute sleeps against a 60s window are
# what turned a ~6 hour run into a ~6 day one.
RATE_LIMIT_BACKOFFS = (15, 30, 60)
RATE_LIMIT_MAX_WAIT = float(os.environ.get("GEN_MAX_BACKOFF", "90"))


def _retry_after_seconds(exc) -> Optional[float]:
    """Pull the server's own wait hint off a 429, if it gave one.

    Groq sends `retry-after` (seconds) and `x-ratelimit-reset-tokens` (e.g.
    "1.5s", "205ms"). Either beats guessing. Returns None if neither is
    readable, in which case the caller falls back to RATE_LIMIT_BACKOFFS.
    """
    headers = None
    for attr in ("response", "http_response"):
        candidate = getattr(exc, attr, None)
        if candidate is not None and getattr(candidate, "headers", None):
            headers = candidate.headers
            break
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None

    def _get(name):
        try:
            return headers.get(name)
        except Exception:
            return None

    raw = _get("retry-after") or _get("Retry-After")
    if raw:
        try:
            return min(float(str(raw).strip()), RATE_LIMIT_MAX_WAIT)
        except (TypeError, ValueError):
            pass

    reset = _get("x-ratelimit-reset-tokens") or _get("x-ratelimit-reset-requests")
    if reset:
        text = str(reset).strip().lower()
        try:
            if text.endswith("ms"):
                return min(float(text[:-2]) / 1000.0, RATE_LIMIT_MAX_WAIT)
            total, number = 0.0, ""
            for char in text:
                if char.isdigit() or char == ".":
                    number += char
                elif char in "hms" and number:
                    total += float(number) * {"h": 3600, "m": 60, "s": 1}[char]
                    number = ""
            if number:
                total += float(number)
            if total > 0:
                return min(total, RATE_LIMIT_MAX_WAIT)
        except (TypeError, ValueError):
            pass
    return None
# Save every 10 pairs, not 50. At ~5.5s per pair (70B generation + rate-limit
# sleep) a 50-pair interval means nothing reaches disk for the first ~4.5
# minutes, so an early crash loses everything and you get no feedback that it's
# working. Rewriting the whole JSON is atomic and costs ~1MB at 1000 pairs —
# cheap enough to do 100 times over a run.
SAVE_EVERY = 10
# llama-3.1-70b-versatile was decommissioned by Groq; 3.3 is the current 70B.
DEFAULT_GEN_MODEL = os.environ.get("GROQ_GEN_MODEL", "llama-3.3-70b-versatile")
FAST_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


def get_groq_client():
    """Build a Groq client or exit with a useful message."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "GROQ_API_KEY is not set.\n"
            f"  Add it to {PROJECT_ROOT / '.env'} or export it:\n"
            "    export GROQ_API_KEY=gsk_...\n"
        )
    try:
        from groq import Groq
    except ImportError:
        sys.exit("groq SDK missing. Run:  pip install groq python-dotenv")
    return Groq(api_key=api_key)


def load_persona(mode: str) -> str:
    """Read backend/prompts/<mode>.txt — the persona the model must imitate."""
    path = PROMPTS_DIR / f"{mode}.txt"
    if not path.exists():
        sys.exit(f"Persona prompt missing: {path}")
    return path.read_text(encoding="utf-8").strip()


# --------------------------------------------------------- json handling
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw: str) -> Optional[dict]:
    """Pull one JSON object out of an LLM response.

    Models wrap JSON in prose, fences, or both. Try the cheap paths first.
    """
    if not raw:
        return None
    text = raw.strip()
    # Strip ```json ... ``` fences.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last resort: models sometimes emit raw newlines inside strings.
        try:
            return json.loads(candidate.replace("\n", " "))
        except json.JSONDecodeError:
            return None


# ------------------------------------------------------------- file I/O
def load_existing(path: Path) -> List[dict]:
    """Resume support: read whatever we generated in previous runs."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        print(f"! {path.name} is not a JSON list — starting fresh")
    except json.JSONDecodeError:
        print(f"! {path.name} is corrupt — starting fresh (old file kept as .bak)")
        path.replace(path.with_suffix(".json.bak"))
    return []


def save_dataset(path: Path, rows: List[dict]) -> None:
    """Atomic write so a Ctrl-C mid-save can never corrupt the dataset."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------ topic plan
def build_topic_queue(topic_counts: Dict[str, int], done: List[dict]) -> List[str]:
    """Return the remaining topics to generate, accounting for what exists.

    Interleaves topics so an interrupted run still yields a balanced dataset
    rather than 400 pairs about one topic and nothing about the rest.
    """
    have: Dict[str, int] = {}
    for row in done:
        tags = row.get("tags") or []
        topic = tags[0] if tags else row.get("topic", "")
        have[topic] = have.get(topic, 0) + 1

    remaining: Dict[str, int] = {}
    for topic, target in topic_counts.items():
        need = target - have.get(topic, 0)
        if need > 0:
            remaining[topic] = need

    queue: List[str] = []
    while remaining:
        for topic in list(remaining.keys()):
            queue.append(topic)
            remaining[topic] -= 1
            if remaining[topic] <= 0:
                del remaining[topic]
    return queue


# --------------------------------------------------------- the main loop
def generate_dataset(
    mode: str,
    topic_counts: Dict[str, int],
    output_name: str,
    user_words: Tuple[int, int] = (50, 100),
    eka_words: Tuple[int, int] = (100, 180),
    model: str = DEFAULT_GEN_MODEL,
    extra_instruction: str = "",
) -> None:
    """Generate a persona conversation dataset, resume-safe.

    Each API call produces exactly one {user, eka_response} pair, which keeps
    JSON parsing reliable and makes every failure cost one pair, not fifty.
    """
    total = sum(topic_counts.values())
    out_path = DATASETS_DIR / output_name
    persona = load_persona(mode)
    client = get_groq_client()

    rows = load_existing(out_path)
    if rows:
        print(f"→ resuming {mode}: {len(rows)}/{total} pairs already generated")
    queue = build_topic_queue(topic_counts, rows)
    if not queue:
        print(f"✅ {mode}: already complete ({len(rows)}/{total}) — nothing to do")
        return

    print(f"→ {mode}: generating {len(queue)} more pairs with {model}")
    print(f"→ ETA ~{len(queue) * SLEEP_BETWEEN_CALLS / 60:.0f} min at {SLEEP_BETWEEN_CALLS}s/call\n")

    since_save = 0
    for topic in queue:
        # The length floor and the one-question rule are stated as hard
        # constraints and repeated at the end, because a first pass at this
        # produced responses averaging 88 words against a 100-180 target and
        # frequently ending in two or three stacked questions. The fine-tune
        # copies whatever the data does, so drift here becomes drift in Eka.
        midpoint = (eka_words[0] + eka_words[1]) // 2
        user_prompt = (
            f"Topic: {topic}\n"
            f"Generate ONE realistic conversation where someone talks to Eka "
            f"about {topic}.\n\n"
            f"USER MESSAGE ({user_words[0]}-{user_words[1]} words): specific and "
            f"emotional. A real person with a name, real numbers, a real place — "
            f"never a generic question.\n\n"
            f"EKA RESPONSE ({eka_words[0]}-{eka_words[1]} words, aim for about "
            f"{midpoint}): must match the persona in the system prompt exactly.\n"
            f"  - Do NOT come in under {eka_words[0]} words. Develop the thought "
            f"fully instead of stopping early.\n"
            f"  - End with EXACTLY ONE question. Not two, not three, not a "
            f"question followed by another question. One.\n"
            f"  - No bullet points or numbered lists. Continuous prose.\n"
            f"{extra_instruction}\n"
            f'Return ONLY valid JSON: {{"user": "...", "eka_response": "..."}}'
        )

        pair = _one_call(client, model, persona, user_prompt, topic)
        if pair is None:
            continue

        pair["mode"] = mode
        pair["tags"] = [topic]
        rows.append(pair)
        since_save += 1

        print(f"✓ {mode}: {len(rows)}/{total} | topic: {topic}")

        if since_save >= SAVE_EVERY:
            save_dataset(out_path, rows)
            since_save = 0
            print(f"  💾 saved {len(rows)} pairs -> {out_path.name}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    save_dataset(out_path, rows)
    print(f"\n✅ {mode} complete: {len(rows)} pairs -> {out_path}")


def _one_call(
    client, model: str, system: str, user_prompt: str, topic: str
) -> Optional[dict]:
    """One generation attempt.

    Rate limits get the full exponential backoff and do NOT consume a content
    attempt — being throttled says nothing about whether the prompt works, so
    burning a retry on it would silently thin out the dataset.
    """
    rate_limit_hits = 0
    attempt = 0
    while attempt < 2:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.9,  # high — we want variety across 1000 pairs
                max_tokens=800,
            )
            raw = completion.choices[0].message.content
        except Exception as exc:
            message = str(exc).lower()
            if "rate" in message or "429" in message or "too many requests" in message:
                # Prefer the server's own hint. The token bucket refills every
                # minute, so this is usually a few seconds — far less than the
                # blind backoff, and it is the difference between ~150 pairs/hr
                # and ~7 pairs/hr.
                hinted = _retry_after_seconds(exc)
                if hinted is not None:
                    wait = max(hinted + 1.0, 5.0)  # +1s so we clear the window
                    source = "server hint"
                else:
                    wait = RATE_LIMIT_BACKOFFS[
                        min(rate_limit_hits, len(RATE_LIMIT_BACKOFFS) - 1)
                    ]
                    source = "backoff"
                rate_limit_hits += 1
                print(
                    f"  ⏳ rate limited (x{rate_limit_hits}) — sleeping "
                    f"{wait:.1f}s ({source}), then retrying '{topic}'"
                )
                time.sleep(wait)
                continue  # deliberately does not increment `attempt`
            attempt += 1
            if "decommission" in message or "does not exist" in message:
                sys.exit(
                    f"\nGroq rejected model '{model}'.\n"
                    "Groq rotates models. Check console.groq.com/docs/models and set\n"
                    "  GROQ_GEN_MODEL=<current 70b model>  in your .env\n"
                )
            print(f"  ! API error on '{topic}' (attempt {attempt}): {exc}")
            time.sleep(5)
            continue

        # The call went through, so this counts as a content attempt.
        attempt += 1

        parsed = extract_json(raw)
        if parsed and parsed.get("user") and parsed.get("eka_response"):
            return {
                "user": str(parsed["user"]).strip(),
                "eka_response": str(parsed["eka_response"]).strip(),
            }
        print(f"  ! JSON parse failed on '{topic}' (attempt {attempt})")
        time.sleep(1)

    print(f"  ✗ skipped one pair for '{topic}'")
    return None


def paraphrase(client, text: str, model: str = FAST_MODEL) -> Optional[str]:
    """Used by the triplet generator to build positives.

    Same never-crash-on-429 policy as _one_call: back off and retry rather than
    dropping the anchor, since 6000 triplets is a long unattended run.
    """
    for hit in range(len(RATE_LIMIT_BACKOFFS) + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Rewrite this in different words keeping exact meaning: "
                            f"{text}\nReturn ONLY the rewritten text, no explanation."
                        ),
                    }
                ],
                temperature=0.8,
                max_tokens=300,
            )
            out = (completion.choices[0].message.content or "").strip()
            return out.strip('"') or None
        except Exception as exc:
            message = str(exc).lower()
            if "rate" in message or "429" in message or "too many requests" in message:
                if hit >= len(RATE_LIMIT_BACKOFFS):
                    return None
                hinted = _retry_after_seconds(exc)
                wait = max(hinted + 1.0, 5.0) if hinted is not None else (
                    RATE_LIMIT_BACKOFFS[hit]
                )
                print(f"  ⏳ rate limited (x{hit + 1}) — sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            return None
    return None


def stats(rows: List[dict]) -> None:
    """Print a per-topic breakdown at the end of a run."""
    counts: Dict[str, int] = {}
    for row in rows:
        tags = row.get("tags") or ["?"]
        counts[tags[0]] = counts.get(tags[0], 0) + 1
    print("\n  topic breakdown:")
    for topic, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}  {topic}")


__all__ = [
    "DATASETS_DIR",
    "PROJECT_ROOT",
    "DEFAULT_GEN_MODEL",
    "FAST_MODEL",
    "SLEEP_BETWEEN_CALLS",
    "generate_dataset",
    "get_groq_client",
    "load_existing",
    "load_persona",
    "paraphrase",
    "save_dataset",
    "stats",
]
