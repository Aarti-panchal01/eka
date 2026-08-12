"""Complexity routing — decides how much context Eka retrieves per message.

Three tiers, tried in order:
  1. A remote complexity microservice, if COMPLEXITY_SERVE_URL is set.
  2. The fine-tuned DistilBERT loaded in-process (lazy, ~255MB resident).
  3. A word-count + structure heuristic. No dependencies, always available.

Tier 3 is genuinely usable — it's the same length signal the training data was
generated from — so Render's 512MB instance can skip torch entirely.
"""

import asyncio
import logging
import re
from typing import Dict, Optional, Tuple

import httpx

from config import COMPLEXITY_LABELS, settings

logger = logging.getLogger(__name__)

# Words that signal the user is doing pattern work on themselves, not just
# asking a question. These push a message up a tier regardless of length.
_DEPTH_MARKERS = (
    "i always", "i keep", "every time", "i realize", "i realise", "i notice",
    "pattern", "since childhood", "my father", "my mother", "growing up",
    "it's connected", "connected to", "i've never told", "i can't stop",
    "why do i", "same thing happened", "again and again", "deep down",
)
_COMPARISON_MARKERS = ("balance", "versus", " vs ", "trade off", "trade-off",
                       "on one hand", "torn between", "both")


class ComplexityService:
    def __init__(self) -> None:
        self._pipeline: Optional[object] = None
        self._load_attempted = False
        self._lock = asyncio.Lock()
        self._remote_ok: Optional[bool] = None
        self._last_tier = "heuristic"

    async def classify(self, text: str) -> Tuple[str, float]:
        """Return (label, confidence). label ∈ simple|normal|complex|deep."""
        text = (text or "").strip()
        if not text:
            return "simple", 1.0

        if settings.COMPLEXITY_SERVE_URL and self._remote_ok is not False:
            result = await self._remote(text)
            if result:
                self._last_tier = "remote"
                return result

        if settings.ENABLE_LOCAL_CLASSIFIERS:
            result = await self._local(text)
            if result:
                self._last_tier = "local"
                return result

        self._last_tier = "heuristic"
        return self.heuristic(text)

    # ------------------------------------------------------------- remote
    async def _remote(self, text: str) -> Optional[Tuple[str, float]]:
        try:
            async with httpx.AsyncClient(timeout=settings.SERVICE_TIMEOUT) as client:
                response = await client.post(
                    f"{settings.COMPLEXITY_SERVE_URL}/classify", json={"text": text}
                )
            if response.status_code != 200:
                return None
            body = response.json()
            label = body.get("label")
            if label in COMPLEXITY_LABELS:
                self._remote_ok = True
                return label, float(body.get("confidence", 0.0))
        except Exception as exc:
            if self._remote_ok is not False:
                logger.warning("Complexity service unreachable (%s) — using local/heuristic", exc)
            self._remote_ok = False
        return None

    # -------------------------------------------------------------- local
    async def _local(self, text: str) -> Optional[Tuple[str, float]]:
        pipe = await self._get_pipeline()
        if pipe is None:
            return None
        try:
            def run():
                return pipe(text[:512], truncation=True, max_length=256)

            output = await asyncio.to_thread(run)
            if isinstance(output, list) and output:
                top = output[0]
                if isinstance(top, list):  # return_all_scores shape
                    top = max(top, key=lambda d: d["score"])
                label = str(top["label"]).lower()
                # A model saved without id2label emits LABEL_0..LABEL_3.
                if label.startswith("label_"):
                    index = int(label.split("_")[-1])
                    label = COMPLEXITY_LABELS[index] if index < len(COMPLEXITY_LABELS) else "normal"
                if label in COMPLEXITY_LABELS:
                    return label, float(top["score"])
        except Exception as exc:
            logger.warning("Local complexity inference failed: %s", exc)
        return None

    async def _get_pipeline(self):
        if self._load_attempted:
            return self._pipeline

        async with self._lock:
            if self._load_attempted:
                return self._pipeline
            self._load_attempted = True

            if not settings.HF_USERNAME:
                logger.info("HF_USERNAME unset — complexity model can't be located")
                return None
            try:
                from transformers import pipeline

                model_id = settings.complexity_model_id
                logger.info("Loading complexity model %s (~255MB)...", model_id)
                self._pipeline = await asyncio.to_thread(
                    pipeline,
                    "text-classification",
                    model=model_id,
                    token=settings.HF_TOKEN or None,
                    device=-1,  # CPU
                )
                logger.info("Complexity model ready")
            except ImportError:
                logger.info(
                    "transformers not installed — complexity falls back to the "
                    "heuristic (expected on Render free tier)"
                )
                self._pipeline = None
            except Exception as exc:
                logger.warning(
                    "Complexity model %s unavailable (%s) — using heuristic. "
                    "This is normal before training finishes.",
                    settings.complexity_model_id, exc,
                )
                self._pipeline = None
        return self._pipeline

    # ---------------------------------------------------------- heuristic
    @staticmethod
    def heuristic(text: str) -> Tuple[str, float]:
        """Length + structure. Mirrors how the training data was generated."""
        words = re.findall(r"\S+", text)
        count = len(words)
        lowered = text.lower()

        depth_hits = sum(1 for marker in _DEPTH_MARKERS if marker in lowered)
        comparison_hits = sum(1 for marker in _COMPARISON_MARKERS if marker in lowered)
        sentences = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
        question_marks = text.count("?")

        if count < 8:
            label, confidence = "simple", 0.75
        elif count < 25:
            label, confidence = "normal", 0.65
        elif count < 50:
            label, confidence = "complex", 0.65
        else:
            label, confidence = "deep", 0.75

        # Autobiographical pattern-work is "deep" even when stated briefly.
        if depth_hits >= 2 and count >= 15:
            label, confidence = "deep", 0.70
        elif depth_hits >= 1 and label in ("simple", "normal") and count >= 15:
            label, confidence = "complex", 0.60

        # Multi-part questions need more context than their length suggests.
        if label == "normal" and (comparison_hits or sentences >= 3 or question_marks >= 2):
            label, confidence = "complex", 0.60

        return label, confidence

    # ------------------------------------------------------------- health
    async def health(self) -> Dict:
        return {
            "tier": self._last_tier,
            "remote_url": settings.COMPLEXITY_SERVE_URL or None,
            "remote_ok": self._remote_ok,
            "local_loaded": self._pipeline is not None,
            "model": settings.complexity_model_id if settings.HF_USERNAME else None,
        }

    async def available(self) -> bool:
        """True if a trained model (not the heuristic) is answering."""
        if settings.COMPLEXITY_SERVE_URL and self._remote_ok:
            return True
        return self._pipeline is not None


complexity_service = ComplexityService()
