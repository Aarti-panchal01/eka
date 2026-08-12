"""Transient throttling must not be reported as a daily quota wall.

Regression test for the 2026-08-12 23:20 stall: a founder run died at 844/1000
after a burst of `rate limited, rotating (hint 2s)` lines. The daily-wall
branch never fired — every 429 was a two-second throttle — but the rotation
budget drained, complete() returned None, and the caller reads None as "every
provider is out of quota for today" and stops the whole run. A --check a minute
later had all four Mistral keys answering with 213k of 6,000,000 tokens spent.

    python tests/test_throttle_waves.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml" / "scripts"))

os.environ.setdefault("FAKE_KEY_A", "test-key-a")
os.environ.setdefault("FAKE_KEY_B", "test-key-b")

from _gen_providers import Provider, ProviderRotator  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, text="", payload=None, headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeClient:
    """Returns transient 429s for the first `throttled_calls`, then succeeds."""

    def __init__(self, throttled_calls):
        self.throttled_calls = throttled_calls
        self.calls = 0

    async def post(self, url, headers=None, json=None):
        self.calls += 1
        if self.calls <= self.throttled_calls:
            # Mistral's wording, with a hint measured in SECONDS. No "per day",
            # no "daily" — the classifier must read this as transient.
            return FakeResponse(
                429,
                text="Requests rate limit exceeded. Please try again in 2.0s",
                headers={"retry-after": "2"},
            )
        return FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"total_tokens": 20},
            },
        )

    async def aclose(self):
        pass


class CountingStdout:
    """Count wave announcements.

    An earlier version of this test patched asyncio.sleep and counted every
    call, which swept up the TokenPacer's own internal sleeps and reported
    7.7 million "waves". The wave message is the thing under test, so count
    that instead of trying to guess which sleeps were ours.
    """

    WAVE_MARKER = "throttled — waiting"

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.waves = []

    def write(self, text):
        if self.WAVE_MARKER in text:
            self.waves.append(text.strip())
        return self.wrapped.write(text)

    def flush(self):
        self.wrapped.flush()


def build_rotator(throttled_calls):
    providers = [
        Provider(
            name=f"fake-{n}",
            base_url="https://example.invalid/v1",
            model="fake-model",
            api_key_env=env,
            tokens_per_minute=1_000_000,
            daily_token_limit=6_000_000,
        )
        for n, env in ((1, "FAKE_KEY_A"), (2, "FAKE_KEY_B"))
    ]
    rotator = ProviderRotator(providers=providers)
    rotator._client = FakeClient(throttled_calls)
    return rotator


def run(name, throttled_calls):
    """Run one complete() with real waits collapsed to zero.

    The waves sleep 5s, 10s, 15s... for real; a test that actually waited
    105 seconds would not get run. Patch asyncio.sleep to return immediately
    but keep counting the wave messages, which is what the assertions use.
    """
    rotator = build_rotator(throttled_calls)
    original = asyncio.sleep

    async def instant(seconds, *a, **k):
        return await original(0)

    counter = CountingStdout(sys.stdout)
    asyncio.sleep = instant
    sys.stdout = counter
    try:
        text, tokens, provider = asyncio.new_event_loop().run_until_complete(
            rotator.complete("sys", "user", max_tokens=100)
        )
    finally:
        asyncio.sleep = original
        sys.stdout = counter.wrapped
    return text, provider, counter.waves, rotator._client.calls


def main():
    failures = []

    # 1. The exact production shape: enough consecutive transient 429s to burn
    #    the rotation budget (max_rotations = max(4, configured*2) = 4), then
    #    the throttle clears. Before the fix this returned None and killed the
    #    run. It must now wait a wave and go on to succeed.
    text, provider, sleeps, calls = run("burst", throttled_calls=6)
    if text is None:
        failures.append(
            f"burst: gave up on a transient throttle (calls={calls}, waves={len(sleeps)})"
        )
    elif not sleeps:
        failures.append("burst: succeeded but never waited — budget was not the limit")
    else:
        print(f"  ok  transient burst recovered after {len(sleeps)} wave(s), "
              f"{calls} calls, served by {provider}")

    # 2. A throttle that never clears must still terminate, and only after
    #    genuinely waiting it out — not instantly.
    text, provider, sleeps, calls = run("permanent", throttled_calls=10_000)
    if text is not None:
        failures.append("permanent: returned text from an always-429 provider")
    elif len(sleeps) < 2:
        failures.append(
            f"permanent: gave up after only {len(sleeps)} wave(s) — too eager"
        )
    else:
        print(f"  ok  unrelenting throttle gave up after {len(sleeps)} waves "
              f"calls={calls}")

    # 3. Waves must be bounded — a stuck provider cannot spin forever.
    if len(sleeps) > 6:
        failures.append(f"permanent: {len(sleeps)} waves exceeds max_throttle_waves")

    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print("all throttle-wave assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
