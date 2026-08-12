"""Standalone complexity classifier service (port 8004).

    python serving/complexity_serve.py

OPTIONAL. The backend loads this model in-process by default, so you only need
this if you're splitting services across machines. If you do run it, point the
backend at it with COMPLEXITY_SERVE_URL=http://localhost:8004.

The model loads lazily on first request, behind a lock, so startup is instant
and two concurrent first-requests can't load it twice (which would double the
255MB footprint and OOM a 512MB box).
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("complexity")

LABELS = ("simple", "normal", "complex", "deep")
HF_USERNAME = os.environ.get("HF_USERNAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "") or None
MODEL_ID = os.environ.get(
    "COMPLEXITY_MODEL_ID",
    f"{HF_USERNAME}/eka-complexity" if HF_USERNAME else "",
)
PORT = int(os.environ.get("PORT", os.environ.get("COMPLEXITY_PORT", 8004)))
EAGER_LOAD = os.environ.get("EAGER_LOAD", "0") == "1"

_pipeline = None
_load_attempted = False
_lock = asyncio.Lock()


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ClassifyResponse(BaseModel):
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

        if not MODEL_ID:
            logger.warning("No model id — set HF_USERNAME or COMPLEXITY_MODEL_ID")
            return None
        try:
            from transformers import pipeline

            logger.info("Loading %s ...", MODEL_ID)
            _pipeline = await asyncio.to_thread(
                pipeline,
                "text-classification",
                model=MODEL_ID,
                token=HF_TOKEN,
                device=-1,
                top_k=None,  # return every class score
            )
            logger.info("Model ready")
        except Exception as exc:
            logger.error("Model load failed: %s", exc)
            _pipeline = None
    return _pipeline


def heuristic(text: str) -> ClassifyResponse:
    """Word-count fallback, identical in spirit to the backend's."""
    words = len(text.split())
    if words < 8:
        label, confidence = "simple", 0.75
    elif words < 25:
        label, confidence = "normal", 0.65
    elif words < 50:
        label, confidence = "complex", 0.65
    else:
        label, confidence = "deep", 0.75
    return ClassifyResponse(
        label=label, confidence=confidence, all_scores={}, tier="heuristic"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if EAGER_LOAD:
        await get_pipeline()
    yield


app = FastAPI(title="Eka complexity service", version="0.1.0", lifespan=lifespan)


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    pipe = await get_pipeline()
    if pipe is None:
        return heuristic(request.text)

    try:
        def run():
            return pipe(request.text[:512], truncation=True, max_length=256)

        output = await asyncio.to_thread(run)
        rows = output[0] if isinstance(output[0], list) else output
        scores = {}
        for row in rows:
            label = str(row["label"]).lower()
            if label.startswith("label_"):
                index = int(label.split("_")[-1])
                label = LABELS[index] if index < len(LABELS) else label
            scores[label] = float(row["score"])
        top = max(scores.items(), key=lambda kv: kv[1])
        return ClassifyResponse(
            label=top[0], confidence=top[1], all_scores=scores, tier="model"
        )
    except Exception as exc:
        logger.warning("Inference failed (%s) — using heuristic", exc)
        return heuristic(request.text)


@app.post("/classify/batch", response_model=List[ClassifyResponse])
async def classify_batch(request: BatchRequest):
    if len(request.texts) > 64:
        raise HTTPException(status_code=413, detail="Max 64 texts per batch")
    return [await classify(ClassifyRequest(text=t)) for t in request.texts if t.strip()]


@app.get("/health")
async def health():
    return {
        "service": "complexity",
        "model_id": MODEL_ID or None,
        "loaded": _pipeline is not None,
        "load_attempted": _load_attempted,
        "labels": list(LABELS),
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("complexity service on :%d (model %s)", PORT, MODEL_ID or "(none)")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
