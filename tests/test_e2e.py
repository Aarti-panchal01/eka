"""End-to-end smoke test for the Eka backend — run against a LIVE server.

Prerequisites
-------------
1. The backend must actually be running and reachable. Locally:

       make serve-local
       # or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload   (from backend/)

2. `.env` should be filled in (DATABASE_URL at minimum; GROQ_API_KEY or a local
   Ollama for chat to work at all — /health will fail loudly otherwise).
3. `httpx` must be installed (it already is — it's a backend dependency).

Usage
-----
    python tests/test_e2e.py
    EKA_BASE_URL=https://eka.onrender.com python tests/test_e2e.py

This is a standalone script, NOT a pytest suite — no pytest import, no
fixtures, no test discovery magic. It runs 7 tests in order against
EKA_BASE_URL (default http://localhost:8000), using a fresh random user_id
each run so nothing collides with previous runs or other test users.

Degraded-mode note
-------------------
Eka is deliberately designed to keep working (in a reduced way) when Qdrant,
Sarvam, or the fine-tuned persona models aren't available — the backend falls
back to heuristics/base models rather than erroring. Tests 3 (memory search),
5 (voice), and 7 (daily/weekly insight summaries) depend on exactly those
external services, so they self-downgrade to WARNINGS instead of hard
failures when the relevant service is reported absent. That means a green
run (7/7) on a minimally-configured box (no Qdrant, no Sarvam key, no
deployed LoRA) is still a meaningful signal that the core API contract works
— check the warnings list printed at the end to see what was skipped.
"""

import io
import json
import os
import sys
import time
import uuid
import wave
from datetime import datetime, timezone

# Windows consoles default to cp1252, which cannot encode ✅/❌/⚠️.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import httpx
except ImportError:
    print("httpx is required: pip install httpx")
    sys.exit(1)


BASE_URL = os.environ.get("EKA_BASE_URL", "http://localhost:8000").rstrip("/")
USER_ID = str(uuid.uuid4())

# Render free-tier instances cold-start from a full stop, which can take
# well over a minute before the first response — give the very first call
# a generous timeout so a sleeping dyno doesn't look like a failed test.
HEALTH_TIMEOUT = 120.0
# Chat goes through the full RAG pipeline (embed -> Qdrant search -> rerank
# -> LLM generation) and the LLM call itself can be slow on a cold Groq/HF
# Space/Ollama backend, so give it more room than a plain CRUD call.
CHAT_TIMEOUT = 90.0
DEFAULT_TIMEOUT = 30.0

# Shared state across tests: session_id, memory_id, goal_id, health flags, etc.
ctx = {}
warnings = []


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"   ⚠️  WARNING: {msg}")


# ------------------------------------------------------------- test runner
_tests = []


def test(name):
    def decorator(fn):
        _tests.append((name, fn))
        return fn

    return decorator


class TestFailure(AssertionError):
    pass


def run_all():
    client = httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT)
    ctx["client"] = client

    results = []
    suite_start = time.perf_counter()

    for name, fn in _tests:
        start = time.perf_counter()
        try:
            fn(client)
            elapsed = time.perf_counter() - start
            print(f"✅ PASS  {name}  ({elapsed:.2f}s)")
            results.append((name, True, elapsed, None))
        except Exception as exc:  # noqa: BLE001 - we want to catch everything
            elapsed = time.perf_counter() - start
            reason = f"{type(exc).__name__}: {exc}"
            print(f"❌ FAIL  {name}  ({elapsed:.2f}s)")
            print(f"     {reason}")
            results.append((name, False, elapsed, reason))

    total_elapsed = time.perf_counter() - suite_start
    passed = sum(1 for _, ok, _, _ in results if ok)

    print()
    if warnings:
        print(f"Warnings collected ({len(warnings)}):")
        for w in warnings:
            print(f"   ⚠️  {w}")
        print()

    print(f"{passed}/{len(results)} tests passed in {total_elapsed:.1f}s")

    client.close()
    sys.exit(0 if passed == len(results) else 1)


# ------------------------------------------------------------------ helpers
def _wav_bytes_1s_silence(framerate=22050) -> bytes:
    """A 1-second, 22050Hz, mono, 16-bit silent WAV built with stdlib `wave`."""
    buf = io.BytesIO()
    n_frames = framerate  # 1 second
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


# ============================================================== test 1
@test("health check")
def test_health(client: httpx.Client):
    resp = client.get("/health", timeout=HEALTH_TIMEOUT)
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    llm_ok = bool(data.get("ollama") or data.get("groq") or data.get("hf_space"))
    assert llm_ok, (
        "no LLM backend is up (ollama/groq/hf_space all false) — "
        "nothing can generate a reply, this must fail loudly"
    )
    assert data.get("database") is True, "database is not reachable — check DATABASE_URL"

    qdrant_ok = bool(data.get("qdrant"))
    ctx["qdrant_ok"] = qdrant_ok
    if not qdrant_ok:
        warn("qdrant is DOWN — memory search/retrieval will run degraded for the rest of this suite")

    print(f"     status={data.get('status')}  llm_mode={data.get('llm_mode')}")
    print(
        f"     ollama={data.get('ollama')} groq={data.get('groq')} "
        f"hf_space={data.get('hf_space')} qdrant={data.get('qdrant')} "
        f"database={data.get('database')}"
    )
    degraded_info = data.get("details", {})
    if degraded_info:
        print(f"     degraded/details: {json.dumps(degraded_info, default=str)[:400]}")

    ctx["health"] = data


# ============================================================== test 2
@test("session + 5 messages")
def test_session_messages(client: httpx.Client):
    resp = client.post(
        "/chat/sessions", json={"user_id": USER_ID, "mode": "founder"}, timeout=DEFAULT_TIMEOUT
    )
    assert resp.status_code in (200, 201), f"create session failed: {resp.status_code} {resp.text}"
    session = resp.json()
    session_id = session["id"]
    ctx["session_id"] = session_id
    print(f"     session_id={session_id}")

    # These must each exceed settings.AUTO_MEMORY_MIN_CHARS (150) or
    # rag_service._worth_remembering() rejects them, no auto-memory is ever
    # created, and the retrieved_memories assertion below fails for the wrong
    # reason — it would be reporting "retrieval is broken" when in fact there
    # was simply nothing to retrieve. Keep them long and first-person.
    messages = [
        "I've been thinking seriously about quitting my job to work on my startup "
        "full time. I've been at this company for four years and the stability is "
        "comfortable, but I feel like I'm wasting the best years I have to take a "
        "real risk on something of my own.",
        "My biggest fear is running out of money within six months and having to "
        "crawl back to a salaried job. I keep running the numbers late at night and "
        "I can't make them work past two quarters without either real revenue "
        "growth or raising outside money.",
        "I have about 200,000 INR saved up right now, which is roughly six months "
        "of personal runway if I keep my expenses exactly where they are and don't "
        "travel or take on anything unexpected during that window.",
        "I already have three paying customers at 2000 rupees a month each, which "
        "isn't much money, but they found me without any marketing spend at all "
        "and two of them have already renewed once without me having to ask.",
        "My co-founder wants to raise funding now while we have some traction, but "
        "I think we should stay lean and push for profitability first. We've been "
        "arguing about this for weeks and it's starting to affect how we work "
        "together day to day.",
    ]

    last_reply = None
    for i, msg in enumerate(messages, start=1):
        resp = client.post(
            "/chat/send",
            json={
                "message": msg,
                "user_id": USER_ID,
                "session_id": session_id,
                "mode": "founder",
            },
            timeout=CHAT_TIMEOUT,
        )
        assert resp.status_code == 200, f"message {i} failed: {resp.status_code} {resp.text}"
        data = resp.json()
        reply = data.get("response", "")
        assert reply, f"message {i}: empty response"
        assert len(reply) > 50, f"message {i}: response too short ({len(reply)} chars): {reply!r}"

        print(
            f"     msg{i}: complexity={data.get('complexity')} "
            f"latency_ms={data.get('latency_ms')} llm_backend={data.get('llm_backend')}"
        )
        last_reply = data

    # For message 5, check retrieved_memories.
    retrieved = last_reply.get("retrieved_memories")
    assert isinstance(retrieved, list), "retrieved_memories is not a list"

    if not retrieved:
        if ctx.get("qdrant_ok"):
            # Qdrant was reported healthy in test 1, so by this point (5
            # messages in) we should have gotten at least one memory back —
            # a fully empty list here means retrieval itself is broken, not
            # just "no data yet", so this is a real assertion failure.
            raise TestFailure(
                "retrieved_memories is empty on message 5 even though /health "
                "reported qdrant=true — memory retrieval appears broken"
            )
        else:
            warn(
                "retrieved_memories is empty on message 5 — memory retrieval is degraded "
                "(Qdrant is down per /health, or embeddings fell back to the hashed tier). "
                "This is expected in a minimally-configured setup, not a hard failure."
            )
    else:
        print(f"     msg5 retrieved_memories: {len(retrieved)} item(s)")

    ctx["message_count"] = len(messages)


# ============================================================== test 3
@test("memory ops")
def test_memory_ops(client: httpx.Client):
    resp = client.post(
        "/memory",
        json={
            "user_id": USER_ID,
            "title": "Background",
            "content": "User is a B.Tech student working on AI",
            "source": "manual",
            "importance": 8,
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert resp.status_code in (200, 201), f"create memory failed: {resp.status_code} {resp.text}"
    memory = resp.json()
    memory_id = memory["id"]
    ctx["memory_id"] = memory_id
    print(f"     memory_id={memory_id}")

    qdrant_ok = ctx.get("qdrant_ok")

    resp = client.post(
        "/memory/search",
        json={"text": "student background", "user_id": USER_ID, "limit": 10},
        timeout=DEFAULT_TIMEOUT,
    )
    assert resp.status_code == 200, f"memory search failed: {resp.status_code} {resp.text}"
    results = resp.json()
    top_ids = [r["id"] for r in results[:3]]
    if memory_id in top_ids:
        print("     created memory found in top 3 search results")
    else:
        if qdrant_ok:
            raise TestFailure(
                f"memory {memory_id} not in top 3 search results ({top_ids}) "
                "even though qdrant is up"
            )
        warn(
            f"memory {memory_id} not found in top 3 search results ({top_ids}) — "
            "qdrant is down per /health, skipping this check"
        )

    # priority update
    resp = client.put(
        f"/memory/{memory_id}/priority",
        params={"user_id": USER_ID},
        json={"priority": "high"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert resp.status_code == 200, f"priority update failed: {resp.status_code} {resp.text}"
    updated = resp.json()
    assert updated.get("user_priority") == "high", f"user_priority not updated: {updated}"

    # delete
    resp = client.delete(
        f"/memory/{memory_id}", params={"user_id": USER_ID}, timeout=DEFAULT_TIMEOUT
    )
    assert resp.status_code == 200, f"delete failed: {resp.status_code} {resp.text}"
    assert resp.json().get("deleted") is True, f"delete response unexpected: {resp.json()}"

    # search again -> gone
    resp = client.post(
        "/memory/search",
        json={"text": "student background", "user_id": USER_ID, "limit": 10},
        timeout=DEFAULT_TIMEOUT,
    )
    assert resp.status_code == 200, f"post-delete search failed: {resp.status_code} {resp.text}"
    ids_after = [r["id"] for r in resp.json()]
    if memory_id in ids_after:
        if qdrant_ok:
            raise TestFailure(f"deleted memory {memory_id} still appears in search results")
        warn(
            f"deleted memory {memory_id} still appears in search results, but qdrant is "
            "down per /health so the search index may just be stale/heuristic — not failing"
        )
    else:
        print("     deleted memory confirmed gone from search")


# ============================================================== test 4
@test("mode switching (chanakya persona)")
def test_mode_switching(client: httpx.Client):
    resp = client.post(
        "/chat/sessions", json={"user_id": USER_ID, "mode": "chanakya"}, timeout=DEFAULT_TIMEOUT
    )
    assert resp.status_code in (200, 201), f"create session failed: {resp.status_code} {resp.text}"
    session_id = resp.json()["id"]
    ctx["chanakya_session_id"] = session_id

    resp = client.post(
        "/chat/send",
        json={
            "message": "My business partner is hiding revenue from me",
            "user_id": USER_ID,
            "session_id": session_id,
            "mode": "chanakya",
        },
        timeout=CHAT_TIMEOUT,
    )
    assert resp.status_code == 200, f"chanakya send failed: {resp.status_code} {resp.text}"
    data = resp.json()
    reply = data.get("response", "")
    assert reply, "chanakya reply is empty"

    # Keyword check is a soft signal, not a hard requirement: LLM_MODE=groq
    # serves the BASE Llama model, not the Chanakya-fine-tuned LoRA, so the
    # persona voice (strategic/political framing) may simply not be deployed
    # yet. An empty reply is still a real failure; a generic-sounding reply
    # is just a warning that the fine-tune isn't live.
    keywords = [
        "evidence", "move", "timing", "strategic", "leverage", "power",
        "position", "alliance", "information", "act", "control",
    ]
    lower_reply = reply.lower()
    if not any(kw in lower_reply for kw in keywords):
        print(f"     reply: {reply}")
        warn(
            "chanakya reply doesn't contain any strategic-persona keyword "
            f"({keywords}) — the persona LoRA is probably not deployed yet "
            "(LLM_MODE=groq serves BASE Llama, not the fine-tune)"
        )
    else:
        print("     chanakya persona keyword found in reply")


# ============================================================== test 5
@test("voice stt/tts")
def test_voice(client: httpx.Client):
    wav_bytes = _wav_bytes_1s_silence()

    resp = client.post(
        "/voice/stt",
        files={"file": ("silence.wav", wav_bytes, "audio/wav")},
        timeout=DEFAULT_TIMEOUT,
    )
    if resp.status_code == 200:
        data = resp.json()
        assert "text" in data, f"stt 200 response missing 'text' key: {data}"
        print(f"     stt: 200 OK (backend={data.get('backend')!r}, text={data.get('text')!r})")
    elif resp.status_code == 503:
        print("     stt: 503 (no ASR backend configured/available) — acceptable")
    else:
        raise TestFailure(f"stt returned unexpected status {resp.status_code}: {resp.text}")

    resp = client.post(
        "/voice/tts",
        json={"text": "Hello I am Eka", "mode": "founder"},
        timeout=DEFAULT_TIMEOUT,
    )
    if resp.status_code == 200:
        content_type = resp.headers.get("content-type", "")
        assert "audio" in content_type, f"tts 200 but content-type={content_type!r}"
        assert len(resp.content) > 1000, f"tts audio body too small: {len(resp.content)} bytes"
        print(f"     tts: 200 OK ({len(resp.content)} bytes, content-type={content_type})")
    elif resp.status_code == 503:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = resp.text
        assert "unavailable" in detail.lower(), f"tts 503 detail doesn't mention unavailable: {detail!r}"
        print(f"     tts: 503 (unavailable — {detail}) — acceptable")
    else:
        raise TestFailure(f"tts returned unexpected status {resp.status_code}: {resp.text}")


# ============================================================== test 6
@test("goals crud + auto-complete")
def test_goals(client: httpx.Client):
    resp = client.post(
        "/goals",
        json={
            "user_id": USER_ID,
            "goal_name": "Ship MVP",
            "target_value": 1,
            "unit": "completion",
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert resp.status_code in (200, 201), f"create goal failed: {resp.status_code} {resp.text}"
    goal = resp.json()
    goal_id = goal["id"]
    ctx["goal_id"] = goal_id
    print(f"     goal_id={goal_id}")

    resp = client.get("/goals", params={"user_id": USER_ID}, timeout=DEFAULT_TIMEOUT)
    assert resp.status_code == 200, f"list goals failed: {resp.status_code} {resp.text}"
    goal_ids = [g["id"] for g in resp.json()]
    assert goal_id in goal_ids, f"created goal {goal_id} not in list {goal_ids}"

    resp = client.put(
        f"/goals/{goal_id}/progress", json={"current_value": 0.5}, timeout=DEFAULT_TIMEOUT
    )
    assert resp.status_code == 200, f"progress update failed: {resp.status_code} {resp.text}"
    data = resp.json()
    assert data.get("current_value") == 0.5, f"current_value not 0.5: {data.get('current_value')}"
    assert data.get("status") == "active", f"expected still active at 0.5, got {data.get('status')}"

    resp = client.put(
        f"/goals/{goal_id}/progress", json={"current_value": 1.0}, timeout=DEFAULT_TIMEOUT
    )
    assert resp.status_code == 200, f"progress update to 1.0 failed: {resp.status_code} {resp.text}"
    data = resp.json()
    assert data.get("current_value") == 1.0, f"current_value not 1.0: {data.get('current_value')}"
    assert data.get("status") == "completed", (
        f"goal did not auto-complete at target_value — status={data.get('status')!r}"
    )
    print("     goal auto-completed at target_value as expected")

    resp = client.delete(f"/goals/{goal_id}", timeout=DEFAULT_TIMEOUT)
    assert resp.status_code == 200, f"delete goal failed: {resp.status_code} {resp.text}"
    assert resp.json().get("deleted") is True, f"delete response unexpected: {resp.json()}"

    resp = client.get("/goals", params={"user_id": USER_ID}, timeout=DEFAULT_TIMEOUT)
    assert resp.status_code == 200, f"list goals after delete failed: {resp.status_code} {resp.text}"
    goal_ids_after = [g["id"] for g in resp.json()]
    assert goal_id not in goal_ids_after, f"deleted goal {goal_id} still present: {goal_ids_after}"


# ============================================================== test 7
@test("daily + weekly insights")
def test_insights(client: httpx.Client):
    today = datetime.now(timezone.utc).date().isoformat()

    resp = client.get(
        f"/insights/daily/{today}", params={"user_id": USER_ID}, timeout=CHAT_TIMEOUT
    )
    assert resp.status_code == 200, f"daily insight failed: {resp.status_code} {resp.text}"
    data = resp.json()

    score = data.get("alignment_score")
    assert isinstance(score, (int, float)), f"alignment_score not numeric: {score!r}"
    assert 0 <= score <= 100, f"alignment_score out of [0,100]: {score}"

    message_count = data.get("message_count", 0)
    assert message_count >= 5, (
        f"message_count is {message_count}, expected >= 5 (test 2 posted 5 messages today)"
    )

    summary = data.get("summary")
    if summary:
        print(f"     summary: {summary[:200]}")
    else:
        warn(
            "daily insight summary is empty — the summarizer model may not exist yet "
            "and the LLM fallback path may have produced nothing"
        )

    print(f"     alignment_score={score}  message_count={message_count}")

    resp = client.get("/insights/weekly", params={"user_id": USER_ID}, timeout=CHAT_TIMEOUT)
    assert resp.status_code == 200, f"weekly insight failed: {resp.status_code} {resp.text}"
    weekly = resp.json()
    days_with_data = weekly.get("days_with_data", 0)
    assert days_with_data >= 1, f"days_with_data is {days_with_data}, expected >= 1"
    print(f"     weekly: days_with_data={days_with_data}  total_messages={weekly.get('total_messages')}")


if __name__ == "__main__":
    print(f"Eka E2E suite -> {BASE_URL}")
    print(f"user_id={USER_ID}")
    print()
    run_all()
