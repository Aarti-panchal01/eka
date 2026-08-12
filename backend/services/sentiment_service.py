"""6-class emotion classification: positive, neutral, negative, reflective,
anxious, motivated.

Shared by rag_service (tags each exchange) and insight_service (derives the
day's mood trend). Same three-tier pattern as complexity: remote service ->
in-process model -> lexicon heuristic.

The heuristic here is weaker than the complexity one — emotion isn't a function
of length — so treat "neutral" from the heuristic tier as "unknown" rather than
as a real signal.
"""

import asyncio
import logging
import re
from typing import Dict, Optional, Tuple

import httpx

from config import SENTIMENT_LABELS, settings

logger = logging.getLogger(__name__)

# Deliberately small and high-precision. Broad lexicons produce confident
# nonsense, which is worse than admitting neutrality.
_LEXICON = {
    "anxious": (
        "afraid", "anxious", "anxiety", "scared", "terrified", "nervous",
        "worried", "worry", "panic", "dread", "fear", "uneasy", "insecure",
        "what if", "can't sleep", "overwhelmed", "spiralling", "spiraling",
    ),
    "negative": (
        "angry", "furious", "hate", "resent", "bitter", "betrayed", "sad",
        "depressed", "miserable", "hopeless", "exhausted", "drained",
        "broken", "failed", "failure", "guilty", "ashamed", "lonely",
        "grief", "hurt", "disappointed", "frustrated", "stuck", "quit",
    ),
    "motivated": (
        "determined", "ready", "committed", "let's do", "lets do", "i will",
        "going to build", "excited to", "can't wait", "motivated", "driven",
        "focused", "all in", "double down", "ship it", "want to",
    ),
    "positive": (
        "happy", "grateful", "thankful", "proud", "glad", "relieved",
        "great", "amazing", "wonderful", "love", "joy", "hopeful",
        "optimistic", "won", "landed", "closed the deal", "it worked",
    ),
    "reflective": (
        "i realize", "i realise", "i notice", "i wonder", "thinking about",
        "looking back", "maybe i", "pattern", "makes me think", "understand why",
        "confused", "not sure why", "figured out", "it occurred to me",
    ),
}


class SentimentService:
    def __init__(self) -> None:
        self._pipeline: Optional[object] = None
        self._load_attempted = False
        self._lock = asyncio.Lock()
        self._remote_ok: Optional[bool] = None
        self._last_tier = "heuristic"

    async def classify(self, text: str) -> Tuple[str, float]:
        text = (text or "").strip()
        if not text:
            return "neutral", 0.0

        if settings.SENTIMENT_SERVE_URL and self._remote_ok is not False:
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

    async def classify_label(self, text: str) -> str:
        label, _confidence = await self.classify(text)
        return label

    # ------------------------------------------------------------- remote
    async def _remote(self, text: str) -> Optional[Tuple[str, float]]:
        try:
            async with httpx.AsyncClient(timeout=settings.SERVICE_TIMEOUT) as client:
                response = await client.post(
                    f"{settings.SENTIMENT_SERVE_URL}/sentiment", json={"text": text}
                )
            if response.status_code != 200:
                return None
            body = response.json()
            label = body.get("label")
            if label in SENTIMENT_LABELS:
                self._remote_ok = True
                return label, float(body.get("confidence", 0.0))
        except Exception as exc:
            if self._remote_ok is not False:
                logger.warning("Sentiment service unreachable (%s)", exc)
            self._remote_ok = False
        return None

    # -------------------------------------------------------------- local
    async def _local(self, text: str) -> Optional[Tuple[str, float]]:
        pipe = await self._get_pipeline()
        if pipe is None:
            return None
        try:
            def run():
                return pipe(text[:512], truncation=True, max_length=128)

            output = await asyncio.to_thread(run)
            if isinstance(output, list) and output:
                top = output[0]
                if isinstance(top, list):
                    top = max(top, key=lambda d: d["score"])
                label = self._normalise(str(top["label"]))
                if label:
                    return label, float(top["score"])
        except Exception as exc:
            logger.warning("Local sentiment inference failed: %s", exc)
        return None

    @staticmethod
    def _normalise(raw: str) -> Optional[str]:
        """Map model output onto Eka's six labels."""
        label = raw.lower().strip()
        if label in SENTIMENT_LABELS:
            return label
        if label.startswith("label_"):
            try:
                index = int(label.split("_")[-1])
                return SENTIMENT_LABELS[index] if index < len(SENTIMENT_LABELS) else None
            except ValueError:
                return None
        # The base fallback model (distilroberta emotion) emits Ekman labels.
        mapping = {
            "joy": "positive", "love": "positive", "admiration": "positive",
            "optimism": "positive", "gratitude": "positive",
            "anger": "negative", "sadness": "negative", "disgust": "negative",
            "grief": "negative", "remorse": "negative",
            "fear": "anxious", "nervousness": "anxious",
            "surprise": "reflective", "confusion": "reflective",
            "realization": "reflective", "curiosity": "reflective",
            "desire": "motivated",
            "neutral": "neutral", "caring": "neutral",
        }
        return mapping.get(label)

    async def _get_pipeline(self):
        if self._load_attempted:
            return self._pipeline

        async with self._lock:
            if self._load_attempted:
                return self._pipeline
            self._load_attempted = True
            try:
                from transformers import pipeline

                model_id = (
                    settings.sentiment_model_id
                    if settings.HF_USERNAME
                    else settings.SENTIMENT_FALLBACK_MODEL
                )
                logger.info("Loading sentiment model %s (~310MB)...", model_id)
                try:
                    self._pipeline = await asyncio.to_thread(
                        pipeline,
                        "text-classification",
                        model=model_id,
                        token=settings.HF_TOKEN or None,
                        device=-1,
                    )
                except Exception:
                    logger.info(
                        "Fine-tuned sentiment model unavailable — using %s",
                        settings.SENTIMENT_FALLBACK_MODEL,
                    )
                    self._pipeline = await asyncio.to_thread(
                        pipeline,
                        "text-classification",
                        model=settings.SENTIMENT_FALLBACK_MODEL,
                        device=-1,
                    )
                logger.info("Sentiment model ready")
            except ImportError:
                logger.info("transformers not installed — sentiment uses the lexicon")
                self._pipeline = None
            except Exception as exc:
                logger.warning("Sentiment model failed to load: %s", exc)
                self._pipeline = None
        return self._pipeline

    # ---------------------------------------------------------- heuristic
    @staticmethod
    def heuristic(text: str) -> Tuple[str, float]:
        lowered = text.lower()
        scores: Dict[str, int] = {}
        for label, terms in _LEXICON.items():
            hits = sum(1 for term in terms if term in lowered)
            if hits:
                scores[label] = hits

        if not scores:
            return "neutral", 0.30

        # Negation flips the most common false positive: "I'm not happy".
        if re.search(r"\b(not|never|no longer|isn't|aren't|don't)\s+\w{0,12}\b", lowered):
            for label in ("positive", "motivated"):
                if label in scores:
                    scores[label] = max(0, scores[label] - 1)

        best = max(scores.items(), key=lambda kv: kv[1])
        if best[1] == 0:
            return "neutral", 0.30
        total = sum(scores.values()) or 1
        # Cap confidence at 0.65: a lexicon should never sound certain.
        return best[0], min(0.65, 0.35 + 0.30 * best[1] / total)

    # ------------------------------------------------------------- health
    async def health(self) -> Dict:
        return {
            "tier": self._last_tier,
            "remote_url": settings.SENTIMENT_SERVE_URL or None,
            "remote_ok": self._remote_ok,
            "local_loaded": self._pipeline is not None,
        }

    async def available(self) -> bool:
        if settings.SENTIMENT_SERVE_URL and self._remote_ok:
            return True
        return self._pipeline is not None


sentiment_service = SentimentService()
