"""A thin safety gate.

Two separate jobs, deliberately kept apart:

  is_safe()      — is this text abusive/toxic enough to refuse? Fails OPEN
                   (returns safe) when the classifier is unreachable, because
                   silently blocking a distressed user's message is worse than
                   letting a rude one through.

  crisis_check() — does this text suggest the person may be in danger? Runs on
                   an explicit keyword list, never on a remote model, so it
                   cannot be disabled by a network failure. Fails CLOSED.

The crisis path does not censor Eka's reply. It attaches resource information,
because a therapy-shaped chatbot that responds to a suicide disclosure with only
a reflective question is not a neutral outcome.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import httpx

from config import settings

logger = logging.getLogger(__name__)

TOXICITY_THRESHOLD = 0.85

# Narrow and explicit. False positives here have a real cost (an unwanted
# crisis banner), so this list stays literal rather than clever.
_CRISIS_PATTERNS = (
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\btake my own life\b",
    r"\bsuicidal?\b",
    r"\bwant to die\b",
    r"\bbetter off dead\b",
    r"\bno reason to live\b",
    r"\bnot worth living\b",
    r"\bhurt myself\b",
    r"\bharm myself\b",
    r"\bcut myself\b",
)
_CRISIS_RE = re.compile("|".join(_CRISIS_PATTERNS), re.IGNORECASE)

CRISIS_RESOURCES = (
    "If you're in danger right now, please reach a person who can help: "
    "Tele-MANAS (India) 14416 or 1-800-891-4416, available 24/7 in multiple "
    "languages. AASRA: +91-9820466726. If you're outside India, "
    "findahelpline.com lists a number for your country."
)


class SafetyService:
    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------- toxicity
    async def is_safe(self, text: str) -> Tuple[bool, float]:
        """(safe, confidence). Fails open — an unreachable model means safe."""
        text = (text or "").strip()
        if not text:
            return True, 1.0
        if not settings.HF_TOKEN:
            return True, 0.0

        url = f"{settings.HF_INFERENCE_BASE}/{settings.TOXICITY_MODEL}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {settings.HF_TOKEN}"},
                    json={"inputs": text[:2000], "options": {"wait_for_model": False}},
                )
            if response.status_code != 200:
                self._available = False
                return True, 0.0

            scores = self._flatten_scores(response.json())
            self._available = True
            if not scores:
                return True, 0.0

            toxic = max(
                (
                    score
                    for label, score in scores.items()
                    if "toxic" in label or "hate" in label or "threat" in label
                ),
                default=0.0,
            )
            return toxic < TOXICITY_THRESHOLD, toxic

        except Exception as exc:
            self._last_error = str(exc)
            self._available = False
            logger.debug("Toxicity check unavailable: %s", exc)
            return True, 0.0

    @staticmethod
    def _flatten_scores(body) -> Dict[str, float]:
        if isinstance(body, dict) and "error" in body:
            return {}
        rows = body[0] if isinstance(body, list) and body and isinstance(body[0], list) else body
        if not isinstance(rows, list):
            return {}
        out = {}
        for row in rows:
            if isinstance(row, dict) and "label" in row:
                out[str(row["label"]).lower()] = float(row.get("score", 0.0))
        return out

    # ---------------------------------------------------------------- crisis
    @staticmethod
    def crisis_check(text: str) -> bool:
        """Keyword-only, so no network failure can switch this off."""
        return bool(_CRISIS_RE.search(text or ""))

    @staticmethod
    def crisis_addendum() -> str:
        return CRISIS_RESOURCES

    def annotate(self, user_message: str, reply: str) -> str:
        """Append crisis resources to a reply when the message warrants it."""
        if not self.crisis_check(user_message):
            return reply
        if "14416" in reply or "findahelpline" in reply:
            return reply
        return f"{reply.rstrip()}\n\n---\n{CRISIS_RESOURCES}"

    # ---------------------------------------------------------------- health
    async def health(self) -> Dict:
        return {
            "toxicity_model": settings.TOXICITY_MODEL,
            "toxicity_available": self._available,
            "crisis_patterns": len(_CRISIS_PATTERNS),
            "last_error": self._last_error,
        }


safety_service = SafetyService()
