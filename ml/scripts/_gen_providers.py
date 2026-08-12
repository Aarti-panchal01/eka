"""Multi-provider rotation, so generation can outlive any one free-tier quota.

    from _gen_providers import ProviderRotator
    rotator = ProviderRotator()          # reads .env, skips unconfigured
    text, usage, provider = await rotator.complete(system, user, max_tokens)

WHY THIS EXISTS
---------------
llama-3.3-70b-versatile on Groq's free tier has a 100,000 tokens-per-DAY cap.
Measured need for the full four-persona set is ~2.4M tokens, i.e. ~24 days on
Groq alone. Several providers serve the same Llama 3.3 70B weights with
independent daily quotas, so rotating across them multiplies the daily ceiling
without paying anyone.

Every provider here speaks the OpenAI /chat/completions wire format, so this is
one httpx client with different base_urls rather than four vendor SDKs — no new
dependencies, and direct access to the response headers the retry-after parsing
needs.

QUALITY IS NOT ASSUMED IDENTICAL ACROSS PROVIDERS
-------------------------------------------------
The same weights do NOT guarantee the same output. Providers differ in
quantisation (fp8 vs bf16), chat template details, and sampling defaults, and
custom-silicon providers (Cerebras, SambaNova) make their own precision
tradeoffs. That is not hypothetical: llama-3.1-8b-instant was measured producing
85-98 word responses against a 170-word floor — 0 of 10 pairs passed. A provider
whose quantisation costs it instruction-following could fail the same way.

So this module does not take quality on faith. Every accepted pair is tagged
with the provider that produced it, and stats() reports per-provider accept and
reject counts. If one provider's output systematically misses the gates, it
shows up as a low accept rate rather than as a quietly mixed-quality dataset.
The gates themselves are unchanged and provider-agnostic — nothing enters a
dataset without clearing all five.

MODEL IDS DRIFT
---------------
Every id below is env-overridable. Vendors rename and retire models without
much notice, so a wrong default should be a one-line .env fix rather than a code
change. When a provider rejects an id, the error is surfaced verbatim.

SETUP
-----
Add whichever keys you have to .env; unconfigured providers are skipped with a
notice, and the rotator works fine with only one:

    GROQ_API_KEY=...          # already present
    CEREBRAS_API_KEY=...      # cloud.cerebras.ai      (free signup)
    SAMBANOVA_API_KEY=...     # cloud.sambanova.ai     (free signup)
    TOGETHER_API_KEY=...      # together.ai            ($1 credit, see below)
    NOVITA_API_KEY=...        # novita.ai              (free tier + credits)

Optional id overrides: GROQ_MODEL_ID, CEREBRAS_MODEL_ID, SAMBANOVA_MODEL_ID,
TOGETHER_MODEL_ID, NOVITA_MODEL_ID.

Verify before committing to a long run:

    python ml/scripts/_gen_providers.py --check

NOTE ON TOGETHER: its $1 is a CREDIT, not a daily free tier. It depletes
permanently (~1.7M tokens at Llama-3.3-70B rates) and does not reset tomorrow.
The rotator treats it like any other provider; just do not expect it back.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), ".env"))
except ImportError:  # pragma: no cover
    pass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class RateLimited(Exception):
    """A provider refused for quota reasons. Carries the server's wait hint."""

    def __init__(self, message: str, retry_after: Optional[float], daily: bool):
        super().__init__(message)
        self.retry_after = retry_after
        # True when the message names a per-DAY quota. A daily wall means this
        # provider is finished until tomorrow — rotate away and do not come
        # back this run, rather than retrying into it.
        self.daily = daily


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    api_key_env: str
    # Measured or documented per-minute token ceiling, used for local pacing.
    # Conservative defaults: pacing too slow costs throughput, pacing too fast
    # costs 429s and (on Groq) real daily quota.
    tokens_per_minute: int = 6_000
    # Per-DAY token ceiling. 0 means unknown. This is the limit that actually
    # decides whether a run finishes: Groq's is 100,000, which is ~133 pairs,
    # and it is not exposed in any response header until you exceed it. Tracked
    # locally so a provider can be skipped BEFORE it 429s.
    daily_token_limit: int = 0
    # Per-DAY REQUEST ceiling, which on some free tiers binds long before tokens
    # do. Google's free tier for gemini-2.5-flash is 20 requests/day
    # ("quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
    # "quotaValue": "20") — at 5 pairs per call that is ~100 pairs/day, nothing
    # like the ~1,600/day its 1M token budget implies. Tracking tokens alone
    # made Google look 16x more capable than it is; this is the same
    # requests-vs-tokens mistake that made Groq look usable earlier.
    daily_request_limit: int = 0
    # Extra request pieces some providers need. Kept as data rather than
    # branching in complete(), so adding a provider never means editing the
    # call path.
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_body: Dict[str, object] = field(default_factory=dict)
    notes: str = ""

    api_key: str = field(default="", repr=False)
    exhausted_until: float = 0.0  # monotonic; set when a daily wall is hit
    tokens_today: int = 0
    requests_today: int = 0
    ok_calls: int = 0
    rate_limited: int = 0
    errors: int = 0
    tokens_used: int = 0
    pairs_accepted: int = 0
    pairs_rejected: int = 0

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def daily_exhausted(self) -> bool:
        if self.daily_token_limit and self.tokens_today >= self.daily_token_limit:
            return True
        if self.daily_request_limit and self.requests_today >= self.daily_request_limit:
            return True
        return False

    def available(self) -> bool:
        return (
            self.configured
            and time.monotonic() >= self.exhausted_until
            and not self.daily_exhausted()
        )


# Ordered by preference. Groq first because its output is the measured baseline
# every gate threshold was calibrated against.
def default_providers() -> List[Provider]:
    """The four configured providers, in rotation order.

    STATUS AS MEASURED 2026-08-12. Every entry was verified with a REAL 5-pair
    generation batch through the actual quality gates, not a "reply OK" probe —
    because a probe proves only that the key authenticates. Two providers passed
    a probe and then failed a real call (SambaNova with HTTP 402, Gemini with a
    truncated 0/5), so probes are not evidence.

      groq        llama-3.3-70b-versatile. Working. 100k tokens/DAY is the real
                  ceiling (~133 pairs/day); 12k/min and 1000 req/day never bind
                  first.
      openrouter  nemotron-3-ultra-550b-a55b:free. Verified 5/5.
      google      gemini-2.5-flash. Verified 5/5 at 622 tokens/pair. 1M
                  tokens/day, which is what makes the whole plan viable.

    SambaNova was removed, not disabled: a 40-token probe returned 200 but a
    real 2.4k-token call returned HTTP 402 PAYMENT_METHOD_REQUIRED with
    balance_units: 0. Its free tier no longer serves generation.

    A NOTE ON MIXING TEACHERS — this is the real cost of the rotation. These are
    three different model families: Llama 3.3 70B, Nemotron 3 Ultra 550B, and
    Gemini 2.5 Flash. Every gate threshold here was calibrated against Groq's
    Llama output, and since Groq caps at ~133 pairs/day while Google allows
    ~1,600, the overwhelming majority of pairs will be Gemini's. All three
    passed the gates cleanly, so this is a provenance question rather than a
    quality one: a LoRA imitates its teacher, and this dataset will have three,
    dominated by one that is not the baseline. Every pair carries a `provider`
    field and the quality report breaks accept rates down per provider, so the
    mix stays auditable — and a single-teacher subset stays recoverable by
    filtering on that field.
    """
    return [
        Provider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            model=os.environ.get("GROQ_MODEL_ID", "llama-3.3-70b-versatile"),
            api_key_env="GROQ_API_KEY",
            tokens_per_minute=12_000,
            daily_token_limit=100_000,
            notes="measured: 100k tok/day, 1000 req/day",
        ),
        Provider(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            # NOT Llama. OpenRouter withdrew free llama-3.3-70b entirely (the
            # :free slug 404s with "The paid version is available now"), and of
            # its 19 remaining free models with >=32k context none are Llama.
            # Nemotron 3 Ultra was chosen by measurement, not by size: on a real
            # 5-pair batch it returned 5/5 accepted, user 142-155w and Eka
            # 195-211w, at 769 tokens/pair. The two runners-up both failed —
            # nemotron-3-super-120b truncated at max_tokens for 0/5, and
            # gemma-4-31b was 429 rate-limited upstream.
            model=os.environ.get(
                "OPENROUTER_MODEL_ID", "nvidia/nemotron-3-ultra-550b-a55b:free"
            ),
            api_key_env="OPENROUTER_API_KEY",
            tokens_per_minute=8_000,
            daily_token_limit=100_000,
            extra_headers={
                "HTTP-Referer": "https://github.com/amijackofalltrades/eka",
                "X-Title": "Eka",
            },
            notes="nemotron-3-ultra-550b, verified 5/5 on a real batch",
        ),
        Provider(
            name="mistral",
            base_url="https://api.mistral.ai/v1",
            model=os.environ.get("MISTRAL_MODEL_ID", "mistral-large-latest"),
            api_key_env="MISTRAL_API_KEY",
            tokens_per_minute=int(os.environ.get("MISTRAL_TPM", "10000")),
            # Mistral's free tier meters per MONTH (~1B tokens), not per day, so
            # a daily cap is a self-imposed pacing guard rather than their limit.
            # Their binding free-tier constraint is ~1 request/second.
            daily_token_limit=int(os.environ.get("MISTRAL_DAILY_TOKENS", "500000")),
            notes="free tier ~1B tokens/month; ~1 req/sec",
        ),
        Provider(
            name="github",
            # RETIRED — kept configured so it self-reports rather than being
            # silently forgotten. Measured 2026-08-12: the Azure-hosted endpoint
            # returns 404, and models.github.ai returns HTTP 410
            # "github_models_retirement_brownout — GitHub Models is temporarily
            # unavailable as part of a scheduled retirement brownout". No token
            # fixes this; the service is going away. Override the base URL via
            # GITHUB_MODELS_BASE_URL if GitHub reverses course.
            base_url=os.environ.get(
                "GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference"
            ),
            model=os.environ.get("GITHUB_MODEL_ID", "meta/Meta-Llama-3.3-70B-Instruct"),
            api_key_env="GITHUB_TOKEN",
            tokens_per_minute=8_000,
            daily_token_limit=150_000,
            notes="RETIRED (HTTP 410 retirement brownout) — expect failure",
        ),
        Provider(
            name="google",
            # Google's OpenAI-compatible endpoint, so this needs no extra SDK
            # and shares the single call path with every other provider.
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model=os.environ.get("GOOGLE_MODEL_ID", "gemini-2.5-flash"),
            api_key_env="GOOGLE_API_KEY",
            tokens_per_minute=30_000,
            # Token budget is generous but never binds — requests do. Measured
            # from the 429 body, not from the published token figure.
            daily_token_limit=1_000_000,
            daily_request_limit=int(os.environ.get("GOOGLE_DAILY_REQUESTS", "20")),
            # reasoning_effort=none IS LOAD BEARING. gemini-2.5-flash is a
            # thinking model: left on, a 5-pair call spent 1,844 tokens on
            # internal reasoning, leaving 552 for output, hit max_tokens and
            # truncated mid-JSON -> 0 usable pairs from a 3,169-token call.
            # With it off: thinking=0, 5/5 pairs accepted, 622 tokens/pair.
            # gemini-2.0-flash is retired (404); 2.5-flash is the current one.
            extra_body={"reasoning_effort": "none"},
            notes="1M tok/day — the provider that makes this plan work",
        ),
    ]


def judge_providers() -> List[Provider]:
    """Small-model rotation for the advice judge.

    An 8B model cannot GENERATE to the quality spec — measured, it produced
    85-98 word responses against a 170-word floor and 0 of 10 pairs passed. But
    binary classification of text it is shown is a much easier task than
    producing that text, and it costs roughly a fifth as many tokens per call.
    So the judge runs on 8B while generation stays on 70B.

    These are separate Provider instances from default_providers() on purpose:
    each has its own TokenPacer, so judging cannot eat the generation model's
    per-minute budget. On Groq they are also separate daily quotas entirely.
    """
    return [
        Provider(
            name="groq-8b",
            base_url="https://api.groq.com/openai/v1",
            model=os.environ.get("GROQ_JUDGE_MODEL_ID", "llama-3.1-8b-instant"),
            api_key_env="GROQ_API_KEY",
            tokens_per_minute=6_000,
            notes="14,400 req/day",
        ),
        Provider(
            name="cerebras-8b",
            base_url="https://api.cerebras.ai/v1",
            model=os.environ.get("CEREBRAS_JUDGE_MODEL_ID", "llama3.1-8b"),
            api_key_env="CEREBRAS_API_KEY",
            tokens_per_minute=60_000,
        ),
        Provider(
            name="sambanova-8b",
            base_url="https://api.sambanova.ai/v1",
            model=os.environ.get(
                "SAMBANOVA_JUDGE_MODEL_ID", "Meta-Llama-3.1-8B-Instruct"
            ),
            api_key_env="SAMBANOVA_API_KEY",
            tokens_per_minute=10_000,
        ),
        Provider(
            name="together-8b",
            base_url="https://api.together.xyz/v1",
            model=os.environ.get(
                "TOGETHER_JUDGE_MODEL_ID",
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            ),
            api_key_env="TOGETHER_API_KEY",
            tokens_per_minute=20_000,
            notes="$1 credit, depletes permanently",
        ),
    ]


def _parse_retry_after(headers, body: str) -> Tuple[Optional[float], bool]:
    """Extract (seconds, is_daily) from a 429."""
    seconds: Optional[float] = None
    raw = None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        raw = None
    if raw:
        try:
            seconds = float(str(raw).strip())
        except (TypeError, ValueError):
            seconds = None

    low = (body or "").lower()
    # The daily/transient distinction decides whether a provider is parked for
    # the rest of the run, so it has to be precise.
    #
    # AN EARLIER VERSION MATCHED "quota exceeded" AND COST A WHOLE RUN. Gemini's
    # per-MINUTE throttle message reads "Quota exceeded for quota metric
    # 'GenerateContent requests per minute'" — that substring made a routine
    # 60-second throttle look like a daily wall, so Google was parked for 24h
    # after spending 49k of its 1,000,000 daily tokens. Both other providers were
    # genuinely out, so the run stopped with 95% of the day's real capacity
    # unused. "insufficient" and "credit" were similarly overbroad.
    #
    # So: require explicit per-day language, and treat any mention of a SHORTER
    # window as proof it is not a daily limit. A missed daily wall is cheap (the
    # provider 429s again and the retry-after cap handles it); a false daily wall
    # costs the entire remaining run.
    per_day = re.search(r"per[\s_-]?day|\btpd\b|daily|per[\s_-]?24\s?h", low)
    shorter_window = re.search(
        r"per[\s_-]?(minute|min\b|second|sec\b|hour|hr\b)|\btpm\b|\brpm\b", low
    )
    # A depleted balance will not reset today either, so it parks — but it is a
    # distinct condition from a daily quota and must not be inferred from the
    # word "quota" alone.
    balance = re.search(
        r"insufficient|payment[\s_-]?(method|required)|balance|out of credit", low
    )
    daily = bool(balance) or (bool(per_day) and not shorter_window)
    if seconds is None:
        # Groq also embeds "Please try again in 56m38.1s".
        # NOTE: `re` is imported at module level. A function-local `import re`
        # here made `re` local to this whole function, so the classifier above
        # raised UnboundLocalError before it ever ran.
        match = re.search(
            r"try again in\s+(?:(\d+)m)?\s*([\d.]+)?s", low
        )
        if match:
            minutes = float(match.group(1) or 0)
            secs = float(match.group(2) or 0)
            seconds = minutes * 60 + secs
    return seconds, daily


class TokenPacer:
    """Per-provider sliding tokens-per-minute window.

    Deliberately a sliding window over expiring charges rather than a counter
    reset on a boundary: a fixed window lets any real 60s span straddling the
    boundary carry two windows' worth. That was measured on an earlier version
    of the generator's budget at 14,400 tokens against a 12,000 cap.

    Never sleeps while holding the lock, so a settle can always land.
    """

    def __init__(self, tokens_per_minute: int, headroom: float = 0.85) -> None:
        self.budget = max(1.0, tokens_per_minute * headroom)
        self._events: List[Tuple[float, float]] = []
        self._reserved = 0.0
        self._lock = asyncio.Lock()

    def _in_window(self) -> float:
        now = time.monotonic()
        self._events = [(t, v) for t, v in self._events if t > now]
        return sum(v for _, v in self._events) + self._reserved

    async def reserve(self, estimate: float) -> dict:
        while True:
            async with self._lock:
                used = self._in_window()
                if used + estimate <= self.budget or used == 0.0:
                    self._reserved += estimate
                    return {"est": estimate}
                delay = (
                    min(t for t, _ in self._events) - time.monotonic()
                    if self._events
                    else 0.5
                )
            await asyncio.sleep(max(0.05, min(delay, 60.0)) + 0.01)

    async def settle(self, ticket: dict, actual: Optional[float]) -> None:
        async with self._lock:
            self._reserved -= ticket["est"]
            charge = ticket["est"] if actual is None else float(actual)
            self._events.append((time.monotonic() + 60.0, charge))

    async def penalise(self, seconds: float) -> None:
        """Back the whole provider off, not just the caller that saw the 429."""
        async with self._lock:
            self._events.append((time.monotonic() + max(0.0, seconds), self.budget))


class ProviderRotator:
    """Tries providers in order, rotating past any that is rate limited.

    Sticks with a working provider rather than round-robining every call: a
    provider that is answering is the cheapest one to keep using, and rotating
    per-call would spread output across quantisations for no benefit. Rotation
    happens on failure only.

    A provider that reports a DAILY wall is parked for the rest of the run.
    Retrying into a 56-minute wall is how the previous single-provider version
    burned request quota to no effect.
    """

    def __init__(
        self,
        providers: Optional[List[Provider]] = None,
        timeout: float = 120.0,
        max_rotations: int = 0,
    ) -> None:
        self.providers = providers if providers is not None else default_providers()
        for provider in self.providers:
            provider.api_key = os.environ.get(provider.api_key_env, "").strip()
        self.timeout = timeout
        self._index = 0
        self._pacers: Dict[str, TokenPacer] = {
            p.name: TokenPacer(p.tokens_per_minute) for p in self.providers
        }
        self._client = httpx.AsyncClient(timeout=timeout)
        self._lock = asyncio.Lock()
        self._usage_dirty = False
        self._load_usage()
        # Total attempts per call, not rotations. With a single configured
        # provider there is nowhere to rotate TO, so a bound of len(configured)
        # made one transient 429 report "every provider is exhausted" and
        # abandon the call — which is exactly the common case while only
        # GROQ_API_KEY is set. Allow real retries, and sleep out a transient
        # limit when no alternative provider is available.
        self.max_rotations = max_rotations or max(4, len(self.configured) * 2)

    # ------------------------------------------------- daily usage tracking
    # Persisted to disk and keyed by local date, so the count survives
    # restarts. This matters because today's Groq quota was consumed by an
    # EARLIER process: a counter that lived only in memory would start every run
    # believing it had a fresh 100k and would rediscover the wall by 429ing into
    # it. Local tracking is a supplement to the 429 handling, never a
    # replacement — the provider's own accounting is authoritative and can
    # include usage from anything else sharing the key.
    def _usage_path(self):
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets",
            ".provider_usage.json",
        )

    def _load_usage(self) -> None:
        today = time.strftime("%Y-%m-%d")
        try:
            with open(self._usage_path(), "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return
        # Anything not from today is stale — that IS the midnight reset.
        for provider in self.providers:
            entry = (data.get(today) or {}).get(provider.name, 0)
            if isinstance(entry, dict):
                provider.tokens_today = int(entry.get("tokens", 0))
                provider.requests_today = int(entry.get("requests", 0))
            else:  # legacy: bare token count
                provider.tokens_today = int(entry or 0)

    def _save_usage(self) -> None:
        if not self._usage_dirty:
            return
        today = time.strftime("%Y-%m-%d")
        path = self._usage_path()
        try:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                data = {}
            # Keep only today, so the file cannot grow without bound.
            data = {today: data.get(today, {})}
            for provider in self.providers:
                if provider.configured:
                    data[today][provider.name] = {
                        "tokens": provider.tokens_today,
                        "requests": provider.requests_today,
                    }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            self._usage_dirty = False
        except Exception as exc:
            print(f"  ! could not persist provider usage: {exc}")

    # ---------------------------------------------------------------- views
    @property
    def configured(self) -> List[Provider]:
        return [p for p in self.providers if p.configured]

    @property
    def available(self) -> List[Provider]:
        return [p for p in self.providers if p.available()]

    def describe(self) -> str:
        lines = []
        for p in self.providers:
            if not p.configured:
                lines.append(f"  - {p.name:10} SKIPPED (no {p.api_key_env})")
            else:
                state = "ready" if p.available() else "exhausted today"
                parts = []
                if p.daily_request_limit:
                    parts.append(f"{p.requests_today}/{p.daily_request_limit} req")
                if p.daily_token_limit:
                    parts.append(f"{p.tokens_today:,}/{p.daily_token_limit:,} tok")
                daily = " · ".join(parts) if parts else f"{p.tokens_today:,} tok"
                daily += " today"
                lines.append(
                    f"  - {p.name:11} {state:16} {p.model[:44]:44} {daily}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------- rotation
    async def _next_available(self) -> Optional[Provider]:
        async with self._lock:
            count = len(self.providers)
            for offset in range(count):
                candidate = self.providers[(self._index + offset) % count]
                if candidate.available():
                    self._index = (self._index + offset) % count
                    return candidate
            return None

    async def _rotate(self) -> None:
        async with self._lock:
            self._index = (self._index + 1) % len(self.providers)

    # ---------------------------------------------------------------- call
    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.9,
        estimate: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Return (text, total_tokens, provider_name). text is None on failure.

        Never raises for provider-level problems — the caller's retry/validation
        machinery handles an empty result. Raises only if nothing is configured,
        which is a setup error worth stopping for.
        """
        if not self.configured:
            raise RuntimeError(
                "No provider API keys found. Add at least GROQ_API_KEY to .env "
                "(see the module docstring for the full list)."
            )

        estimate = estimate if estimate is not None else max_tokens + 2_000
        attempts = 0
        while attempts < self.max_rotations:
            provider = await self._next_available()
            if provider is None:
                print("  ✗ every configured provider is exhausted for today")
                return None, None, None

            pacer = self._pacers[provider.name]
            ticket = await pacer.reserve(estimate)
            settled = False
            # One try/finally around the whole attempt: every reserve() must be
            # settled or the reservation stays in flight forever and eventually
            # deadlocks the pacer. A bare `finally: pass` would leak the ticket
            # on CancelledError, which is a BaseException and skips `except
            # Exception` entirely — and Ctrl-C is the likeliest way this code
            # ever gets cancelled.
            try:
                headers = {
                    "Authorization": f"Bearer {provider.api_key}",
                    "Content-Type": "application/json",
                }
                headers.update(provider.extra_headers)
                body = {
                    "model": provider.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                # Provider-specific knobs as data, not branches. Google's
                # reasoning_effort=none goes here and is load bearing: with
                # thinking on, a 5-pair call burned 1,844 tokens reasoning,
                # truncated, and yielded 0 usable pairs.
                body.update(provider.extra_body)
                try:
                    response = await self._client.post(
                        f"{provider.base_url}/chat/completions",
                        headers=headers,
                        json=body,
                    )
                except Exception as exc:
                    provider.errors += 1
                    print(f"  ! {provider.name} transport error: {exc}")
                    await self._rotate()
                    attempts += 1
                    continue

                body = response.text
                if response.status_code == 429:
                    provider.rate_limited += 1
                    seconds, daily = _parse_retry_after(response.headers, body)
                    if daily or (seconds or 0) > 300:
                        # Park it for the run. A daily wall is not a spike, and
                        # retrying into one burns request quota to no effect.
                        provider.exhausted_until = time.monotonic() + 24 * 3600
                        print(
                            f"  ⇄ {provider.name} hit a DAILY quota wall — "
                            f"parking it and rotating"
                        )
                    else:
                        wait = max(2.0, min(seconds or 2.0, 60.0))
                        await pacer.penalise(wait)
                        # If there is somewhere else to go, go there immediately.
                        # If this is the only provider left, waiting is the only
                        # thing that can help — rotating back onto itself and
                        # giving up would abandon a call over a 2-second blip.
                        alternatives = [
                            p for p in self.available if p.name != provider.name
                        ]
                        if alternatives:
                            print(
                                f"  ⇄ {provider.name} rate limited, rotating "
                                f"(hint {wait:.0f}s)"
                            )
                        else:
                            print(
                                f"  ⏳ {provider.name} rate limited and it is the "
                                f"only one available — waiting {wait:.0f}s"
                            )
                            await asyncio.sleep(wait)
                    await self._rotate()
                    attempts += 1
                    continue

                if response.status_code >= 400:
                    provider.errors += 1
                    # Surfaced verbatim: a renamed or retired model id is the
                    # single most likely cause and the message names it.
                    print(
                        f"  ! {provider.name} HTTP {response.status_code}: "
                        f"{body[:220]}"
                    )
                    # A deterministic 4xx will never fix itself mid-run: the
                    # payload shape is constant, so a bad/retired model id (404,
                    # 410), a bad key (401), or a forbidden resource (403) fails
                    # identically on every retry. Left unparked, GitHub Models'
                    # HTTP 410 retirement brownout cost one wasted round-trip on
                    # EVERY call for the whole run — the only provider actually
                    # burning time, since genuinely parked providers are skipped
                    # by available() without any network call at all.
                    if response.status_code in (400, 401, 403, 404, 410):
                        provider.exhausted_until = time.monotonic() + 24 * 3600
                        print(
                            f"  ⇄ {provider.name} returned a permanent "
                            f"{response.status_code} — parking it for this run"
                        )
                    await self._rotate()
                    attempts += 1
                    continue

                try:
                    payload = response.json()
                    text = payload["choices"][0]["message"]["content"]
                    total = (payload.get("usage") or {}).get("total_tokens")
                except Exception as exc:
                    provider.errors += 1
                    print(f"  ! {provider.name} unreadable response: {exc}")
                    await self._rotate()
                    attempts += 1
                    continue

                await pacer.settle(ticket, total)
                settled = True
                provider.ok_calls += 1
                provider.tokens_used += int(total or 0)
                provider.tokens_today += int(total or 0)
                provider.requests_today += 1
                self._usage_dirty = True
                if provider.daily_exhausted():
                    print(
                        f"  ⇄ {provider.name} reached its tracked daily limit "
                        f"({provider.tokens_today:,}/{provider.daily_token_limit:,}) "
                        f"— rotating"
                    )
                    self._save_usage()
                    await self._rotate()
                elif provider.ok_calls % 10 == 0:
                    # Persist periodically rather than every call: the file is
                    # tiny but a long run makes hundreds of writes otherwise.
                    self._save_usage()
                return text, total, provider.name
            finally:
                if not settled:
                    # Nothing was consumed on any failure path, so refund fully.
                    await pacer.settle(ticket, 0.0)

        print("  ✗ all providers failed this call")
        return None, None, None

    # --------------------------------------------------------------- stats
    def note_outcome(self, provider_name: Optional[str], accepted: int, rejected: int) -> None:
        """Record gate outcomes per provider.

        This is what makes a bad provider visible. Same weights do not mean same
        output; a provider whose quantisation costs it instruction-following
        shows up here as a low accept rate instead of quietly degrading the
        dataset.
        """
        for provider in self.providers:
            if provider.name == provider_name:
                provider.pairs_accepted += accepted
                provider.pairs_rejected += rejected
                return

    def stats(self) -> dict:
        out = {}
        for p in self.providers:
            if not p.configured:
                continue
            total = p.pairs_accepted + p.pairs_rejected
            out[p.name] = {
                "model": p.model,
                "ok_calls": p.ok_calls,
                "rate_limited": p.rate_limited,
                "errors": p.errors,
                "tokens_used": p.tokens_used,
                "pairs_accepted": p.pairs_accepted,
                "pairs_rejected": p.pairs_rejected,
                "accept_rate": round(p.pairs_accepted / total, 4) if total else None,
                "exhausted": not p.available(),
            }
        return out

    def print_stats(self) -> None:
        rows = self.stats()
        if not rows:
            return
        print("\n  per-provider outcomes:")
        for name, s in rows.items():
            rate = "n/a" if s["accept_rate"] is None else f"{s['accept_rate']:.0%}"
            print(
                f"    {name:10} calls={s['ok_calls']:4} tok={s['tokens_used']:>8,} "
                f"accepted={s['pairs_accepted']:4} rejected={s['pairs_rejected']:4} "
                f"accept={rate:>5} {'EXHAUSTED' if s['exhausted'] else ''}"
            )

    async def aclose(self) -> None:
        self._save_usage()
        await self._client.aclose()


# =========================================================== --check probe
async def _check() -> int:
    """Probe every configured provider with a real, tiny call."""
    rotator = ProviderRotator()
    print("\nConfigured providers:")
    print(rotator.describe())

    missing = [p for p in rotator.providers if not p.configured]
    if missing:
        print("\nTo add a provider, put its key in .env:")
        for p in missing:
            print(f"  {p.api_key_env:22} -> {p.notes}")

    if not rotator.configured:
        print("\nNothing configured. Add at least GROQ_API_KEY to .env.")
        await rotator.aclose()
        return 1

    print("\nProbing each with one real call (~40 tokens each):")
    daily_total = 0
    for provider in rotator.configured:
        single = ProviderRotator(providers=[provider], max_rotations=1)
        started = time.monotonic()
        text, total, name = await single.complete(
            "You are a terse assistant.",
            "Reply with exactly: OK",
            max_tokens=8,
            temperature=0.0,
            estimate=200,
        )
        elapsed = time.monotonic() - started
        if text is None:
            print(f"  ✗ {provider.name:10} FAILED (see message above)")
        else:
            print(
                f"  ✓ {provider.name:10} {elapsed:5.2f}s  "
                f"{total or '?'} tokens  model={provider.model}  "
                f"reply={text.strip()[:20]!r}"
            )
            daily_total += 1
        await single.aclose()

    print(f"\n{daily_total}/{len(rotator.configured)} configured providers answered.")
    if daily_total:
        print(
            "Run the generators normally — they rotate automatically as each "
            "provider's daily quota runs out."
        )
    await rotator.aclose()
    return 0 if daily_total else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Eka generation providers")
    parser.add_argument(
        "--check", action="store_true", help="probe every configured provider"
    )
    args = parser.parse_args()
    if args.check:
        return asyncio.run(_check())
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
