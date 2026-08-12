"""Async, batched, quality-gated persona-data generator.

    from _gen_async import generate_dataset_async

Replaces the sequential engine in _gen_common.py. Reuses its IO/persona/resume
helpers; adds concurrency, batching, and the quality pipeline in _gen_quality.

WHY THIS IS FASTER, in descending order of how much it actually matters
----------------------------------------------------------------------
1. BATCHING (the real win). One call returns PAIRS_PER_CALL pairs. The
   ~1200-token persona system prompt is the dominant fixed cost of a call and
   is sent ONCE per batch instead of once per pair:

       unbatched:  5 x (1200 persona + 450 prompt + 700 out) = ~11,750 tok / 5 pairs
       batched(5):  1200 persona + 450 prompt + 2,500 out     =  ~4,150 tok / 5 pairs

   ~2,350 -> ~830 tokens per pair. Against a per-minute token ceiling that is
   a ~2.8x throughput gain, and it is the only change here worth more than a
   few percent.

2. CONCURRENCY. Overlaps network and model latency so the token bucket is not
   idle waiting on round trips. Worth 10-25%.

   It is NOT a multiplier. Concurrency cannot raise a token budget, only drain
   it sooner. TokenBudget below enforces the ceiling across all workers, which
   is what stops N concurrent workers from collectively overshooting and
   collapsing into a 429 storm.

3. SERVER-DIRECTED BACKOFF. On 429, sleep for max(retry-after, RETRY_FLOOR).
   The floor keeps a short burst blip cheap; honouring the header is what
   handles a genuinely drained bucket, where a fixed short sleep would re-429
   in a hot loop and burn the daily request cap for nothing.

MEASURED GROQ FREE-TIER LIMITS (checked live 2026-08-12, not from docs)
----------------------------------------------------------------------
                             tokens/min      requests/day
    llama-3.3-70b-versatile     12,000            1,000
    llama-3.1-8b-instant         6,000           14,400
    llama-3.1-70b-versatile     DECOMMISSIONED (HTTP 400)

The binding constraint is TOKENS PER MINUTE. Groq does not advertise a
per-minute request cap on these models; it caps requests per DAY. Any plan
built on "N requests per minute" is planning against the wrong number. Note
that 8b-instant has HALF the token ceiling of 70b — switching to it to go
faster makes a batched run ~2x SLOWER. Its 14,400/day request budget is the
only reason to reach for it (see the request-cap warning below).

REQUESTS ARE THE SECOND CEILING, AND WITH VALIDATION THEY CAN BIND FIRST
-----------------------------------------------------------------------
3,200 pairs / 5 per call = 640 batch calls. Regeneration adds more. At a
realistic 10% rejection rate, regenerating one pair per call would add ~320
requests for a total of ~960 against a hard 1,000/day — one retry-heavy run
exhausts the day.

So regeneration is BATCHED BY FAILURE REASON: pairs that failed the same way
are regenerated together, five at a time, with a prompt that names that exact
deficiency. Same corrective signal, ~5x fewer requests (~770 total). Set
GEN_REGEN_INDIVIDUAL=1 to force strict one-pair-per-call instead, and watch
the request count.

Env knobs (no code edit needed to retune):

    GROQ_GEN_MODEL         model id          (default llama-3.3-70b-versatile)
    GEN_BATCH              pairs per call    (default 5)
    GEN_CONCURRENCY        in-flight calls   (default 3)
    GEN_TPM                token/min override (default: per-model table)
    GEN_RETRY_FLOOR        min 429 sleep, s  (default 8)
    GEN_INTER_BATCH_SLEEP  s between batches (default 1)
    GEN_REGEN_INDIVIDUAL   1 = one pair/call (default 0 = batch by reason)
    GEN_MAX_PASSES         shortfall passes  (default 3)
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from _gen_common import (
    DATASETS_DIR,
    build_topic_queue,
    extract_json,
    load_existing,
    load_persona,
    save_dataset,
)
from _gen_judge import AdviceJudge
from _gen_providers import ProviderRotator
from _gen_quality import (
    DiversityIndex,
    PersonaSpec,
    print_report,
    quality_report,
    revalidate_existing,
    topic_of,
    validate_pair,
    write_report,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_MODEL = os.environ.get("GROQ_GEN_MODEL", "llama-3.3-70b-versatile")
PAIRS_PER_CALL = int(os.environ.get("GEN_BATCH", "5"))
CONCURRENCY = int(os.environ.get("GEN_CONCURRENCY", "3"))
MAX_PASSES = int(os.environ.get("GEN_MAX_PASSES", "3"))
REGEN_INDIVIDUAL = os.environ.get("GEN_REGEN_INDIVIDUAL", "0") == "1"

_TPM_BY_MODEL = {
    "llama-3.3-70b-versatile": 12_000,
    "llama-3.1-8b-instant": 6_000,
}

RETRY_FLOOR = float(os.environ.get("GEN_RETRY_FLOOR", "8.0"))
RETRY_MAX = float(os.environ.get("GEN_RETRY_MAX", "90.0"))
INTER_BATCH_SLEEP = float(os.environ.get("GEN_INTER_BATCH_SLEEP", "1.0"))
MAX_RETRIES_PER_BATCH = int(os.environ.get("GEN_BATCH_RETRIES", "6"))

SAVE_EVERY = 10

# Token accounting for budget reservation. Reservations use the max_tokens
# upper bound and are then refunded down to real usage via settle(), so these
# only need to be in the right ballpark.
_PERSONA_TOKENS_EST = 1_200
_PROMPT_TOKENS_EST = 450
_OUT_TOKENS_PER_PAIR = 700  # covers a 250-word response + 200-word user + JSON


def tpm_for(model: str) -> int:
    override = os.environ.get("GEN_TPM")
    if override:
        return int(override)
    return _TPM_BY_MODEL.get(model, 6_000)


# TokenBudget used to live here. Per-minute pacing now belongs to
# _gen_providers.TokenPacer, one instance PER PROVIDER, because each provider
# has its own ceiling (Groq 12k tok/min, Cerebras ~60k) and a single shared
# budget would pace every provider to the slowest one. Keeping two
# implementations of a sliding-window limiter was the faster way to let them
# drift apart; TokenPacer carries the same fixes (sliding not tumbling window,
# never sleeping under the lock, ticket-based settle, non-re-armable escape).

# _retry_after_seconds lived here. Retry-after parsing now belongs to
# _gen_providers._parse_retry_after, which also classifies a 429 as a
# transient spike vs a DAILY quota wall — the distinction that decides
# whether to back off for seconds or park the provider for the run.


def extract_pairs(raw: str, want: int) -> List[dict]:
    """Pull as many valid {user, eka_response} objects as possible out of a reply.

    Batching's one real cost is JSON fragility: ask for five objects and a
    truncated or chatty reply can cost all five. So a malformed reply is never
    treated as zero pairs if anything in it is salvageable — parse the array
    when possible, else scan for individual balanced objects (which is what
    rescues a reply truncated mid-way through its last element).

    Shortfalls are re-queued by the caller, so a partial batch costs a fraction
    of a call rather than a whole one.
    """
    if not raw:
        return []
    text = raw.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    def _shaped(obj) -> bool:
        return (
            isinstance(obj, dict)
            and isinstance(obj.get("user"), str)
            and isinstance(obj.get("eka_response"), str)
            and obj["user"].strip()
            and obj["eka_response"].strip()
        )

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                good = [o for o in parsed if _shaped(o)]
                if good:
                    return good[:want]
        except json.JSONDecodeError:
            pass

    found: List[dict] = []
    depth, obj_start, in_string, escaped = 0, None, False, False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                candidate = extract_json(text[obj_start : i + 1])
                if _shaped(candidate):
                    found.append(candidate)
                    if len(found) >= want:
                        break
                obj_start = None
    return found


def build_prompt(
    topic: str, n: int, spec: PersonaSpec, extra_instruction: str
) -> str:
    """Batched generation prompt.

    Constraints are stated as hard rules and restated at the end because an
    earlier unguarded pass produced responses averaging 88 words against a
    100-180 target and routinely ended in three stacked questions. A fine-tune
    copies whatever the data does, so drift here becomes drift in Eka.

    The "must differ from each other" clause is specific to batching: asked for
    five conversations in one breath, models hand back five paraphrases of one
    conversation, which destroys the diversity the high temperature is meant to
    buy — and then the diversity gate rejects four of them.
    """
    u_lo, u_hi = spec.user_words
    e_lo, e_hi = spec.eka_words
    e_mid = (e_lo + e_hi) // 2

    rules = [
        f'"user" ({u_lo}-{u_hi} words): specific and emotional. A real person '
        f"with a name, real numbers, a real place — never a generic question.",
        f'"eka_response" ({e_lo}-{e_hi} words, aim for {e_mid}): must match the '
        f"persona in the system prompt exactly.",
        f"Do NOT come in under {e_lo} words. Develop the thought fully instead "
        f"of stopping early.",
        "End with EXACTLY ONE question. Not two, not three, not a question "
        "followed by another question. One.",
        "No bullet points or numbered lists. Continuous prose.",
    ]
    if spec.markers:
        rules.append(
            "The response MUST naturally use at least one of these ideas: "
            + ", ".join(spec.markers)
            + "."
        )
    if spec.require_verse:
        rules.append(
            "The response MUST cite a specific Bhagavad Gita chapter and verse "
            "in the form 2.47, and the citation must fit the situation."
        )
    if spec.forbid_advice:
        rules.append(
            "CRITICAL: the response must NOT give advice. Never write 'you "
            "should', 'you need to', 'my advice', 'try to', or 'make sure'. "
            "Only reflect back and ask. The user must reach their own "
            "conclusion."
        )
    if spec.require_ends_with_question:
        rules.append("The response MUST end with a question mark.")

    numbered = "\n".join(f"  {i}. {r}" for i, r in enumerate(rules, 1))
    return (
        f"Topic: {topic}\n"
        f"Generate {n} DIFFERENT realistic conversations where someone talks to "
        f"Eka about {topic}.\n\n"
        f"Each conversation is one object with two fields:\n{numbered}\n"
        f"{extra_instruction}\n"
        f"CRITICAL: all {n} conversations must be genuinely different from each "
        f"other — different names, different cities, different numbers, "
        f"different specifics. Do not produce {n} variations of one situation.\n\n"
        f"Return ONLY a valid JSON array of exactly {n} objects, nothing else:\n"
        f'[{{"user": "...", "eka_response": "..."}}, ...]'
    )


def build_regen_prompt(
    topic: str,
    n: int,
    spec: PersonaSpec,
    reasons: Sequence[str],
    extra: str,
    avoid: str = "",
) -> str:
    """Stricter prompt naming exactly what the rejected attempt got wrong.

    This is the whole value of the regeneration step. A generic "try again"
    reproduces the same defect; naming the deficiency is what actually moves
    the model, which is why validate_pair() phrases its reasons as complete
    sentences suitable for pasting in here.
    """
    listed = "\n".join(f"  - {r}" for r in reasons)
    # A persona can supply its own corrective framing. Reflection does, because
    # "you gave advice" needs a blunter instruction than a generic defect list.
    preamble = spec.regen_hint or (
        "A previous attempt at this task was REJECTED by an automated quality "
        "check."
    )
    # A duplicate needs the offending text quoted back. Telling a model to "be
    # different" without saying different from what just regenerates the same
    # scenario, which burns the single retry for nothing.
    steer = ""
    if avoid:
        steer = (
            f"\n\nGenerate a completely different scenario for topic {topic} — "
            f"different situation, different person, different emotional context "
            f"than this, which already exists:\n"
            f'  "{avoid[:200]}"\n'
        )
    return (
        f"{preamble}\n\nThe specific problems were:\n{listed}{steer}\n\n"
        f"Fix every one of those problems. Then:\n\n"
        + build_prompt(topic, n, spec, extra)
    )


class FatalFlag:
    """A run-ending condition, carried rather than raised.

    Raising SystemExit inside a task does NOT surface at the `await` site:
    asyncio sets it on the task and re-raises it out of the event loop, so the
    awaiting coroutine sees CancelledError and every batch that had completed
    but not yet been consumed is discarded. Measured on the previous version:
    10 of 30 paid-for pairs lost, and the CLI's return value never assigned.
    Ctrl-C takes the identical path and is far more likely in practice.
    """

    def __init__(self) -> None:
        self.message: Optional[str] = None

    def set(self, message: str) -> None:
        if self.message is None:  # keep the first cause, not the last
            self.message = message

    def is_set(self) -> bool:
        return self.message is not None


async def _call(
    rotator: ProviderRotator,
    persona: str,
    prompt: str,
    n: int,
    label: str,
    fatal: FatalFlag,
) -> Tuple[List[dict], Optional[str]]:
    """One generation call. Returns (pairs, provider_name). Never raises.

    Per-provider rate limiting, rotation, and token pacing all live in the
    rotator now, so this function only retries for reasons the rotator cannot
    fix by switching providers: an unparseable reply. A 429 is not retried here
    because the rotator has already moved to a different provider, or has told
    us every one is out for the day.

    The provider name comes back so the caller can attribute gate outcomes and
    catch a provider whose quantisation costs it instruction-following — the
    failure mode measured on 8b, where 0 of 10 pairs cleared the length floor.
    """
    if fatal.is_set():
        return [], None
    max_tokens = 300 + n * _OUT_TOKENS_PER_PAIR
    estimate = _PERSONA_TOKENS_EST + _PROMPT_TOKENS_EST + max_tokens

    for attempt in range(MAX_RETRIES_PER_BATCH):
        if fatal.is_set():
            return [], None
        text, _tokens, provider = await rotator.complete(
            persona, prompt, max_tokens=max_tokens, temperature=0.9,
            estimate=estimate,
        )
        if text is None:
            # Every configured provider is rate limited or erroring. Stop the
            # run cleanly and keep what is on disk rather than spinning: with
            # daily quotas the wall is hours wide, not seconds.
            fatal.set(
                "Every configured provider failed or is out of quota for today.\n"
                "Progress is saved — re-run to resume. Check availability with:\n"
                "  python ml/scripts/_gen_providers.py --check\n"
                "Adding another provider key to .env raises the daily ceiling."
            )
            return [], None

        pairs = extract_pairs(text, n)
        if pairs:
            if INTER_BATCH_SLEEP:
                # Smooths token burn so a provider's window drains evenly
                # rather than in spikes that trigger avoidable 429s.
                await asyncio.sleep(INTER_BATCH_SLEEP)
            return pairs, provider
        # Say WHY, not just that it happened. These fire often — 184 times
        # against ~256 successful batches on 2026-08-12 — and the bare message
        # gave nothing to act on: a reply truncated by max_tokens, a chatty
        # preamble that hides the array, and a refusal all looked identical in
        # the log. extract_pairs already salvages fenced code, whole arrays and
        # individual balanced objects, so anything reaching here defeated all
        # three, and the shape of the text is the only clue left.
        preview = " ".join((text or "").split())
        # Unbalanced braces, not "does it end in punctuation" — a refusal and a
        # chatty preamble both end in a full stop and neither is truncated.
        # An unclosed object is the one shape that actually means the reply was
        # cut off, and it is the one that says raise max_tokens.
        looks_truncated = preview.count("{") > preview.count("}")
        print(
            f"  ! unparseable reply for {label} from {provider} "
            f"(attempt {attempt + 1}) — {len(text or '')} chars"
            f"{', looks TRUNCATED' if looks_truncated else ''}"
            f" | starts: {preview[:110]!r}"
            f" | ends: {preview[-70:]!r}"
        )

    print(f"  ✗ gave up on {label}")
    return [], None



def build_batch_queue(
    topic_counts: Dict[str, int], rows: List[dict], batch_size: int
) -> List[Tuple[str, int]]:
    """Group remaining per-topic need into (topic, count) batches.

    Emitted round-robin across topics rather than topic-by-topic, so an
    interrupted run leaves a balanced dataset instead of every pair for the
    first few topics and none for the rest. A batch stays single-topic because
    that is what lets one persona prompt cover all five pairs.
    """
    # A batch_size of 0 makes `remaining[topic] -= 0` loop forever.
    batch_size = max(1, batch_size)
    remaining: Dict[str, int] = {}
    for topic in build_topic_queue(topic_counts, rows):
        remaining[topic] = remaining.get(topic, 0) + 1

    batches: List[Tuple[str, int]] = []
    while remaining:
        for topic in list(remaining.keys()):
            take = min(batch_size, remaining[topic])
            batches.append((topic, take))
            remaining[topic] -= take
            if remaining[topic] <= 0:
                del remaining[topic]
    return batches


class Accumulator:
    """Validates arrivals, enforces quotas, and checkpoints.

    Validation runs here — in the single consumer — rather than inside the
    workers, so the diversity index and the per-topic counters are only ever
    touched from one place and cannot race.
    """

    def __init__(
        self,
        mode: str,
        spec: PersonaSpec,
        topic_counts: Dict[str, int],
        out_path: Path,
        rows: List[dict],
        diversity: DiversityIndex,
        rotator: Optional[ProviderRotator] = None,
    ) -> None:
        self.mode = mode
        self.spec = spec
        self.topic_counts = topic_counts
        self.out_path = out_path
        self.rows = rows
        self.diversity = diversity
        self.rotator = rotator
        self.since_save = 0
        self.accepted = 0
        self.rejected = 0
        self.regen_attempted = 0
        self.regen_succeeded = 0
        # Judge outcomes are tracked separately from `rejected` so the report can
        # distinguish "the regex caught it" from "the judge caught it" — the
        # latter is the number that says whether the judge is earning its cost.
        self.judge_rejected = 0
        self.judge_unavailable = 0
        # Quota ceiling: never exceed a topic's share by more than 10%.
        self.quota_cap = {t: int(q * 1.10) for t, q in topic_counts.items()}
        self.have: Dict[str, int] = {}
        for row in rows:
            t = topic_of(row)
            self.have[t] = self.have.get(t, 0) + 1

    def total(self) -> int:
        return len(self.rows)

    async def offer_async(
        self,
        topic: str,
        pairs: List[dict],
        judge: Optional[AdviceJudge] = None,
        provider: Optional[str] = None,
    ) -> List[Tuple[List[str], str]]:
        """Cheap gates, then the LLM judge, then accept. Returns rejects.

        Three-phase on purpose:

        1. The cheap gates run first so a pair that fails on word count never
           costs a judge call. At a realistic reject rate that is most of the
           savings available here.
        2. The judge runs on survivors only, concurrently across the batch.
        3. Acceptance re-checks the diversity gate against the index as it
           grows, which is what still catches two duplicates inside the SAME
           batch — phase 1 validated them both against the index before either
           was added, so neither saw the other.
        """
        rejects: List[Tuple[List[str], str]] = []
        survivors: List[dict] = []
        for pair in pairs:
            if self.have.get(topic, 0) >= self.quota_cap.get(topic, 10**9):
                continue
            reasons = validate_pair(pair, self.spec, self.diversity, topic)
            if reasons:
                self.rejected += 1
                rejects.append((reasons, str(pair.get("user", ""))))
                continue
            survivors.append(pair)

        # --- phase 2: LLM judge -------------------------------------------
        if judge is not None and survivors and self.spec.llm_judge_advice:
            verdicts = await judge.judge_many(
                [str(p.get("eka_response", "")) for p in survivors]
            )
            kept: List[dict] = []
            for pair, verdict in zip(survivors, verdicts):
                if verdict is True:
                    self.rejected += 1
                    self.judge_rejected += 1
                    rejects.append(
                        (
                            [
                                "an LLM judge found that the response gives "
                                "direct advice or tells the person what to do; "
                                "it must only reflect and ask"
                            ],
                            str(pair.get("user", "")),
                        )
                    )
                    continue
                if verdict is None:
                    # Fail-open, but counted — judge.unavailable drives
                    # judge_coverage, which blocks the report below 1.0 so a
                    # dataset cannot claim 0% advice over unjudged pairs.
                    self.judge_unavailable += 1
                kept.append(pair)
            survivors = kept

        # --- phase 3: accept ----------------------------------------------
        accepted_now = 0
        for pair in survivors:
            if self.have.get(topic, 0) >= self.quota_cap.get(topic, 10**9):
                continue
            # Re-check diversity only: everything else already passed and the
            # index has grown since phase 1.
            score = self.diversity.worst_similarity(
                str(pair.get("user", "")), topic
            )
            if score > self.spec.max_similarity:
                self.rejected += 1
                rejects.append(
                    (
                        [
                            f"the user message is {score:.2f} Jaccard-similar to "
                            f"another pair in this same batch (limit "
                            f"{self.spec.max_similarity:.2f}); it needs a "
                            f"different person, city, and numbers"
                        ],
                        str(pair.get("user", "")),
                    )
                )
                continue
            pair["mode"] = self.mode
            pair["tags"] = [topic]
            if provider:
                # Provenance, so a provider whose output systematically misses
                # the gates is identifiable after the fact rather than blended
                # invisibly into the dataset.
                pair["provider"] = provider
            self.rows.append(pair)
            self.diversity.add(str(pair.get("user", "")), topic)
            self.have[topic] = self.have.get(topic, 0) + 1
            self.accepted += 1
            self.since_save += 1
            accepted_now += 1

        if provider is not None and self.rotator is not None:
            self.rotator.note_outcome(provider, accepted_now, len(rejects))
        return rejects

    def maybe_save(self) -> None:
        if self.since_save >= SAVE_EVERY:
            save_dataset(self.out_path, self.rows)
            self.since_save = 0
            print(f"  💾 saved {len(self.rows)} pairs -> {self.out_path.name}")

    def flush(self) -> None:
        if self.since_save:
            save_dataset(self.out_path, self.rows)
            self.since_save = 0
            print(f"  💾 saved {len(self.rows)} pairs -> {self.out_path.name}")


def _reason_key(reasons: Sequence[str]) -> str:
    """Collapse a reason list to a groupable signature.

    Groups on the KIND of failure, not the exact numbers, so "42 words, below
    the 150-word minimum" and "88 words, below the 150-word minimum" batch
    together and share one corrective prompt.
    """
    kinds = []
    for reason in reasons:
        low = reason.lower()
        if "below the" in low and "minimum" in low:
            kinds.append("too_short")
        elif "above the" in low and "maximum" in low:
            kinds.append("too_long")
        elif "markers" in low:
            kinds.append("no_marker")
        elif "verse" in low:
            kinds.append("no_verse")
        elif "does not end with a question" in low:
            kinds.append("no_question")
        elif "stacked questions" in low:
            kinds.append("stacked_questions")
        elif "advice" in low:
            kinds.append("advice")
        elif "similar" in low:
            kinds.append("duplicate")
        else:
            kinds.append("other")
    return "|".join(sorted(set(kinds)))


async def generate_dataset_async(
    mode: str,
    topic_counts: Dict[str, int],
    output_name: str,
    spec: PersonaSpec,
    model: str = DEFAULT_MODEL,
    extra_instruction: str = "",
    report_name: Optional[str] = None,
) -> dict:
    """Generate a quality-gated persona dataset. Returns the quality report.

    Resume-safe: all state lives in the output JSON, so killing this at any
    point and re-running picks up from what is on disk.
    """
    try:
        rotator = ProviderRotator()
    except Exception as exc:
        sys.exit(f"could not build the provider rotator: {exc}")

    if not rotator.configured:
        sys.exit(
            "No provider API keys found. Add at least GROQ_API_KEY to .env.\n"
            "See ml/scripts/_gen_providers.py for the full list, then verify with:\n"
            "  python ml/scripts/_gen_providers.py --check"
        )

    total_target = sum(topic_counts.values())
    out_path = DATASETS_DIR / output_name
    report_path = DATASETS_DIR / (
        report_name or output_name.replace(".json", "") + "_quality_report.json"
    )
    persona = load_persona(mode)
    # Combined per-minute ceiling across providers that are still available.
    # This is the throughput estimate; the DAILY caps are what actually decide
    # whether a run finishes today, and they are only discoverable by hitting
    # them (Groq's 100k/day is not exposed in any header until it is exceeded).
    tpm = sum(p.tokens_per_minute for p in rotator.available) or 6_000

    # --- resume, re-validating what is already there ----------------------
    raw_rows = load_existing(out_path)
    rows, dropped, diversity = revalidate_existing(raw_rows, spec)
    if raw_rows:
        print(f"→ resuming {mode}: {len(raw_rows)} pairs on disk")
        if dropped:
            # Quality floors can change between runs. Inheriting pairs built
            # under a looser spec would count toward the total and drag the
            # reported averages below target while looking like progress.
            print(
                f"  ⚠ dropped {len(dropped)} that fail the current spec "
                f"(they will be regenerated)"
            )
            for row in dropped[:3]:
                print(f"      - {row['_reasons'][0]}")
            if len(dropped) > 3:
                print(f"      ... and {len(dropped) - 3} more")
            save_dataset(out_path, rows)

    batches = build_batch_queue(topic_counts, rows, PAIRS_PER_CALL)
    if not batches:
        print(f"✅ {mode}: already complete ({len(rows)}/{total_target})")
        report = quality_report(rows, spec, topic_counts)
        write_report(report_path, report)
        print_report(report)
        return report

    need = total_target - len(rows)
    est_per_pair = (
        _PERSONA_TOKENS_EST + _PROMPT_TOKENS_EST + 300 + PAIRS_PER_CALL * 500
    ) / PAIRS_PER_CALL
    pairs_per_hour = (tpm * 0.85 / est_per_pair) * 60

    print(f"\n→ {mode}: {need} pairs needed, {len(batches)} batches")
    print(f"→ batch {PAIRS_PER_CALL} | concurrency {CONCURRENCY}")
    print("→ providers:")
    print(rotator.describe())
    print(f"→ combined {tpm:,} tok/min | ~{est_per_pair:.0f} tok/pair amortised")
    print(
        f"→ ceiling ~{pairs_per_hour:.0f} pairs/hr -> ETA ~{need / pairs_per_hour:.1f}h "
        f"before validation overhead and daily caps"
    )
    print(f"→ ~{len(batches)} batch calls + regens\n")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    acc = Accumulator(mode, spec, topic_counts, out_path, rows, diversity, rotator)
    fatal = FatalFlag()
    live: List[asyncio.Task] = []  # so the finally can cancel cleanly

    # The judge only exists for personas that ask for it, so no other persona
    # pays for the extra call. It rotates over 8B models on their own quotas.
    judge: Optional[AdviceJudge] = None
    if spec.llm_judge_advice:
        judge = AdviceJudge()
        if judge.configured:
            print(f"→ advice judge active: {judge.rotator.configured[0].model}")
        else:
            print("→ ⚠ advice judge requested but no provider key — coverage will be 0")

    async def batch_job(topic: str, n: int, prompt: str, label: str):
        """Carry the topic through with the result.

        Results are consumed with as_completed() rather than zip() order, which
        matters for durability: with zip() a single slow first batch head-of-line
        blocked validation and checkpointing behind it. Measured — 20 of 20 API
        calls complete, yet the first checkpoint waited 300s on batch 0, leaving
        95 paid-for pairs in memory with zero durability. That pile is exactly
        what a crash or Ctrl-C discards. Returning (topic, pairs) keeps the
        mapping without relying on as_completed yielding the original task, which
        is only guaranteed on newer Pythons.
        """
        # The semaphore bounds how many generation calls are in flight; the
        # rotator's per-provider pacers bound tokens per minute.
        async with semaphore:
            pairs, provider = await _call(rotator, persona, prompt, n, label, fatal)
        return topic, pairs, provider

    try:
        for pass_no in range(1, MAX_PASSES + 1):
            if pass_no > 1:
                batches = build_batch_queue(topic_counts, acc.rows, PAIRS_PER_CALL)
                if not batches:
                    break
                short = sum(n for _, n in batches)
                print(f"\n→ pass {pass_no}: {short} pairs still short, retrying\n")

            before = acc.total()

            # ---- generation pass ----------------------------------------
            tasks = [
                asyncio.create_task(
                    batch_job(
                        topic, n,
                        build_prompt(topic, n, spec, extra_instruction),
                        f"'{topic}'",
                    )
                )
                for topic, n in batches
            ]
            live = list(tasks)
            # Reject reasons collected per topic so regeneration can name them.
            pending: Dict[Tuple[str, str], List[Tuple[List[str], str]]] = {}
            for future in asyncio.as_completed(tasks):
                topic, pairs, provider = await future
                rejects = await acc.offer_async(topic, pairs, judge, provider)
                if pairs:
                    print(
                        f"✓ {mode}: {acc.total()}/{total_target} "
                        f"(+{len(pairs) - len(rejects)} kept, {len(rejects)} rejected)"
                        f" | {topic}"
                    )
                for reasons, user_text in rejects:
                    key = (topic, _reason_key(reasons))
                    pending.setdefault(key, []).append((reasons, user_text))
                acc.maybe_save()
            acc.flush()

            # ---- regeneration pass: ONE attempt per rejected pair --------
            if pending:
                group_size = 1 if REGEN_INDIVIDUAL else PAIRS_PER_CALL
                regen_jobs: List[Tuple[str, int, List[str], str]] = []
                for (topic, _kind), items in pending.items():
                    reasons = items[0][0]  # representative reason list
                    # Only quote text back when the failure was a duplicate;
                    # for a length or marker failure the offending text is
                    # irrelevant noise in the prompt.
                    avoid = next(
                        (
                            text
                            for reason_list, text in items
                            if any("similar" in r.lower() for r in reason_list)
                        ),
                        "",
                    )
                    remaining = len(items)
                    while remaining > 0:
                        take = min(group_size, remaining)
                        regen_jobs.append((topic, take, reasons, avoid))
                        remaining -= take

                total_regen = sum(n for _, n, _, _ in regen_jobs)
                acc.regen_attempted += total_regen
                print(
                    f"\n→ regenerating {total_regen} rejected pair(s) in "
                    f"{len(regen_jobs)} call(s), naming the specific defect\n"
                )
                regen_tasks = [
                    asyncio.create_task(
                        batch_job(
                            topic, n,
                            build_regen_prompt(
                                topic, n, spec, reasons, extra_instruction, avoid
                            ),
                            f"regen '{topic}'",
                        )
                    )
                    for topic, n, reasons, avoid in regen_jobs
                ]
                live = list(regen_tasks)
                for future in asyncio.as_completed(regen_tasks):
                    topic, pairs, provider = await future
                    before_regen = acc.total()
                    rejects = await acc.offer_async(topic, pairs, judge, provider)
                    gained = acc.total() - before_regen
                    acc.regen_succeeded += gained
                    if pairs:
                        print(
                            f"↻ {mode}: {acc.total()}/{total_target} "
                            f"(+{gained} recovered, {len(rejects)} still failing) "
                            f"| {topic}"
                        )
                    acc.maybe_save()
                acc.flush()

            if fatal.is_set():
                break
            if acc.total() == before:
                print("→ a full pass added nothing — stopping rather than looping")
                break
    finally:
        # Cancel anything still in flight BEFORE saving, so no task mutates
        # acc.rows while it is being written out.
        for task in live:
            if not task.done():
                task.cancel()
        if live:
            await asyncio.gather(*live, return_exceptions=True)
        acc.flush()
        save_dataset(out_path, acc.rows)
        # BaseException (CancelledError) is deliberately not swallowed here;
        # the saves above have already happened.
        try:
            await rotator.aclose()
        except Exception:
            pass
        if judge is not None:
            try:
                await judge.aclose()
            except Exception:
                pass

    # Raised here, on solid ground, rather than from inside a task — see
    # FatalFlag for why a SystemExit in a task loses completed work.
    if fatal.is_set():
        raise SystemExit(f"\n{fatal.message}\n")

    report = quality_report(
        acc.rows, spec, topic_counts,
        generated=acc.accepted,
        rejected=acc.rejected,
        regen_attempted=acc.regen_attempted,
        regen_succeeded=acc.regen_succeeded,
    )
    report["api_requests_used"] = sum(
        v["ok_calls"] for v in rotator.stats().values()
    )
    report["provider_stats"] = rotator.stats()
    if judge is not None:
        report["judge"] = judge.stats()
        report["judge_coverage"] = judge.stats()["coverage"]
        report["judge_advice_rejections"] = acc.judge_rejected
        report["judge_unavailable"] = acc.judge_unavailable
    write_report(report_path, report)
    print_report(report)
    print(f"  dataset -> {out_path}")
    print(f"  report  -> {report_path}")
    print(f"  api requests used this run: {report['api_requests_used']}")
    rotator.print_stats()
    if judge is not None:
        js = judge.stats()
        print(f"  advice judge: {js['judged']} judged, "
              f"{js['advice_found']} flagged as advice, "
              f"{js['unavailable']} unavailable "
              f"(coverage {js['coverage']:.0%})")
    return report
