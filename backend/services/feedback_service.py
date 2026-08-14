"""Implicit relevance labels for the ranker, harvested from real use.

The ranker is currently trained on synthetic data from `make_dataset()` — data
built to be separable, then learned. NDCG@3 0.9466 against that is not evidence
about production behaviour.

This collects the real thing. After each reply we hold which memories were
retrieved. When the NEXT message in that session arrives, we ask a cheap
question: did the user stay on the topic those memories were about? If yes,
they were probably useful; if the subject changed immediately, probably not.

That is a weak label, not ground truth — a user can change subject for reasons
that have nothing to do with retrieval quality. It is still vastly better than
synthetic labels, and at a few hundred rows it is enough to retrain against.

WHERE THE ROWS GO
-----------------
stdout, as one JSON object per line prefixed `RANKER_FEEDBACK`. Render's disk
is ephemeral — a JSONL file here does not survive a redeploy — but stdout is
captured and retrievable. A local file is also written when a writable path
exists, which is what you get in development.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Words too common to signal that two messages are about the same thing.
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in",
    "on", "for", "with", "at", "by", "from", "up", "about", "into", "over",
    "after", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "should", "could", "can",
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your", "our",
    "this", "that", "these", "those", "what", "how", "why", "when", "who",
    "not", "no", "yes", "just", "get", "got", "like", "know", "think",
}

_LOG_PATH = Path(__file__).resolve().parent.parent / "ranker_feedback.jsonl"

# session_id -> the reply we are waiting to judge
_PENDING: Dict[str, Dict] = {}
_MAX_PENDING = 500  # bounded: this is a cache, not a store


def _tokens(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z0-9']{3,}", (text or "").lower())
        if w not in _STOP
    }


def _overlap(a: str, b: str) -> float:
    """Jaccard over content words. Cheap, and good enough for a weak label."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def record_retrieval(
    session_id: str,
    user_id: str,
    message: str,
    mode: str,
    complexity: str,
    memories: List[Dict],
) -> None:
    """Hold this turn's retrieval until the next message judges it."""
    if not session_id:
        return
    _PENDING[session_id] = {
        "session_id": session_id,
        "user_id": user_id,
        "mode": mode,
        "complexity": complexity,
        "message": message[:400],
        "memory_ids": [m.get("id") for m in memories if m.get("id")],
        # Kept so the follow-up can be compared against what was retrieved,
        # not merely against the previous message.
        "memory_text": " ".join(
            (m.get("content") or "")[:300] for m in memories
        )[:1200],
        "n_memories": len(memories),
    }
    if len(_PENDING) > _MAX_PENDING:
        _PENDING.pop(next(iter(_PENDING)))


def resolve(session_id: str, next_message: str) -> Optional[Dict]:
    """Judge the previous turn's retrieval using the message that followed."""
    pending = _PENDING.pop(session_id, None)
    if not pending:
        return None

    # Two signals, deliberately separate: staying on the same TOPIC, and
    # referencing what was actually RETRIEVED. The second is the stronger
    # evidence that a memory earned its place in the prompt.
    topic_overlap = _overlap(pending["message"], next_message)
    memory_overlap = _overlap(pending["memory_text"], next_message)
    continued = topic_overlap >= 0.08 or memory_overlap >= 0.06

    row = {
        **{k: v for k, v in pending.items() if k != "memory_text"},
        "next_message": (next_message or "")[:400],
        "topic_overlap": round(topic_overlap, 4),
        "memory_overlap": round(memory_overlap, 4),
        "user_continued_thread": bool(continued),
    }

    logger.info("RANKER_FEEDBACK %s", json.dumps(row, ensure_ascii=False))
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # Read-only or ephemeral filesystem — stdout above is the durable path.
        pass
    return row
