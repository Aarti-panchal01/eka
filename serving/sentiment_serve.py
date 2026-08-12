"""Standalone sentiment classifier service (port 8007).

    python serving/sentiment_serve.py

OPTIONAL — see complexity_serve.py. Point the backend at it with
SENTIMENT_SERVE_URL=http://localhost:8007.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("sentiment")

LABELS = ("positive", "neutral", "negative", "reflective", "anxious", "motivated")

# Maps the base fallback model's Ekman labels onto Eka's six.
EKMAN_MAP = {
    "joy": "positive", "love": "positive", "admiration": "positive",
    "optimism": "positive", "gratitude": "positive", "approval": "positive",
    "anger": "negative", "sadness": "negative", "disgust": "negative",
    "grief": "negative", "remorse": "negative", "annoyance": "negative",
    "fear": "anxious", "nervousness": "anxious",
    "surprise": "reflective", "confusion": "reflective",
    "realization": "reflective", "curiosity": "reflective",
    "desire": "motivated",
    "neutral": "neutral", "caring": "neutral",
}

HF_USERNAME = os.environ.get("HF_USERNAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "") or None
MODEL_ID = os.environ.get(
    "SENTIMENT_MODEL_ID",
    f"{HF_USERNAME}/eka-sentiment" if HF_USERNAME else
    "j-hartmann/emotion-english-distilroberta-base",
)
PORT = int(os.environ.get("PORT", os.environ.get("SENTIMENT_PORT", 8007)))
EAGER_LOAD = os.environ.get("EAGER_LOAD", "0") == "1"

_pipeline = None
_load_attempted = False
_lock = asyncio.Lock()


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1)


class SentimentResponse(BaseModel):
    label: str
    confidence: float
    all_scores: Dict[str, float] = Field(default_factory=dict)
    tier: str


class BatchRequest(BaseModel):
    texts: List[str]


async def get_pipeline():
    global _pipeline, _load_attempted
    if _load_attempted:
        return _pipeline

    async with _lock:
        if _load_attempted:
            return _pipeline
        _load_attempted = True
        try:
            from transformers import pipeline

            logger.info("Loading %s ...", MODEL_ID)
            _pipeline = await asyncio.to_thread(
                pipeline,
                "text-classification",
                model=MODEL_ID,
                token=HF_TOKEN,
                device=-1,
                top_k=None,
            )
            logger.info("Model ready")
        except Exception as exc:
            logger.error("Model load failed: %s", exc)
            _pipeline = None
    return _pipeline


def normalise(raw: str) -> str:
    label = raw.lower().strip()
    if label in LABELS:
        return label
    if label.startswith("label_"):
        try:
            index = int(label.split("_")[-1])
            return LABELS[index] if index < len(LABELS) else "neutral"
        except ValueError:
            return "neutral"
    return EKMAN_MAP.get(label, "neutral")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if EAGER_LOAD:
        await get_pipeline()
    yield


app = FastAPI(title="Eka sentiment service", version="0.1.0", lifespan=lifespan)


@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(request: SentimentRequest) -> SentimentResponse:
    pipe = await get_pipeline()
    if pipe is None:
        return SentimentResponse(
            label="neutral", confidence=0.0, all_scores={}, tier="unavailable"
        )

    try:
        def run():
            return pipe(request.text[:512], truncation=True, max_length=128)

        output = await asyncio.to_thread(run)
        rows = output[0] if isinstance(output[0], list) else output

        # Several source labels can collapse to the same Eka label; sum them.
        scores: Dict[str, float] = {}
        for row in rows:
            label = normalise(str(row["label"]))
            scores[label] = scores.get(label, 0.0) + float(row["score"])
        top = max(scores.items(), key=lambda kv: kv[1])
        return SentimentResponse(
            label=top[0],
            confidence=min(1.0, top[1]),
            all_scores={k: round(v, 4) for k, v in scores.items()},
            tier="model",
        )
    except Exception as exc:
        logger.warning("Inference failed: %s", exc)
        return SentimentResponse(
            label="neutral", confidence=0.0, all_scores={}, tier="error"
        )


@app.post("/sentiment/batch", response_model=List[SentimentResponse])
async def sentiment_batch(request: BatchRequest):
    if len(request.texts) > 64:
        raise HTTPException(status_code=413, detail="Max 64 texts per batch")
    return [await sentiment(SentimentRequest(text=t)) for t in request.texts if t.strip()]


@app.get("/health")
async def health():
    return {
        "service": "sentiment",
        "model_id": MODEL_ID,
        "loaded": _pipeline is not None,
        "load_attempted": _load_attempted,
        "labels": list(LABELS),
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("sentiment service on :%d (model %s)", PORT, MODEL_ID)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
