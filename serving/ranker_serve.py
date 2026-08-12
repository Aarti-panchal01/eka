"""Standalone memory reranker service (port 8005).

    python serving/ranker_serve.py

Pure numpy + lightgbm — no torch, ~10MB resident. Cheap enough to run anywhere.
Point the backend at it with RANKER_SERVE_URL=http://localhost:8005.

Feature order is a contract with training/train_ranker_local.py:
    f0 cosine  f1 importance  f2 priority  f3 days_old
    f4 access  f5 topic_match f6 source    f7 length_norm
"""

import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ranker")

FEATURE_NAMES = [
    "cosine_similarity", "importance", "priority_weight", "days_old",
    "access_count", "topic_match", "source_type", "content_length_norm",
]
N_FEATURES = len(FEATURE_NAMES)
MAX_ROWS = 256

DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent / "ml" / "models" / "ranker" / "eka_ranker.txt"
)
MODEL_PATH = Path(os.environ.get("RANKER_MODEL_PATH", str(DEFAULT_PATH)))
PORT = int(os.environ.get("PORT", os.environ.get("RANKER_PORT", 8005)))

_booster = None


def load_booster():
    """Loaded once at import — the model is tiny, no need to be lazy."""
    global _booster
    if _booster is not None:
        return _booster
    if not MODEL_PATH.exists():
        logger.warning(
            "No model at %s — /rank will use the heuristic. "
            "Create it with: python training/train_ranker_local.py",
            MODEL_PATH,
        )
        return None
    try:
        import lightgbm as lgb

        _booster = lgb.Booster(model_file=str(MODEL_PATH))
        logger.info("Loaded ranker (%d trees) from %s", _booster.num_trees(), MODEL_PATH)
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        _booster = None
    return _booster


class RankRequest(BaseModel):
    features: List[List[float]] = Field(..., min_length=1)

    @field_validator("features")
    @classmethod
    def _check_shape(cls, rows):
        if len(rows) > MAX_ROWS:
            raise ValueError(f"at most {MAX_ROWS} rows per request")
        for index, row in enumerate(rows):
            if len(row) != N_FEATURES:
                raise ValueError(
                    f"row {index} has {len(row)} features, expected {N_FEATURES} "
                    f"({', '.join(FEATURE_NAMES)})"
                )
        return rows


class RankResponse(BaseModel):
    scores: List[float]
    tier: str


def heuristic_scores(rows: List[List[float]]) -> List[float]:
    """Mirrors the training scoring function, so ordering stays consistent."""
    out = []
    for cosine, importance, priority, days_old, access, topic, _source, _length in rows:
        recency = max(0.0, 1.0 - (days_old / 365.0))
        out.append(
            1.60 * cosine
            + 0.55 * (importance / 10.0)
            + 0.50 * (priority / 3.0)
            + 0.45 * recency
            + 0.40 * topic
            + 0.20 * min(1.0, access / 20.0)
        )
    return out


app = FastAPI(title="Eka ranker service", version="0.1.0")


@app.post("/rank", response_model=RankResponse)
async def rank(request: RankRequest) -> RankResponse:
    booster = load_booster()
    if booster is None:
        return RankResponse(scores=heuristic_scores(request.features), tier="heuristic")
    try:
        matrix = np.asarray(request.features, dtype=np.float32)
        scores = booster.predict(matrix)
        return RankResponse(scores=[float(s) for s in scores], tier="model")
    except Exception as exc:
        logger.warning("Inference failed (%s) — using heuristic", exc)
        return RankResponse(scores=heuristic_scores(request.features), tier="heuristic")


@app.get("/health")
async def health():
    booster = load_booster()
    return {
        "service": "ranker",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "loaded": booster is not None,
        "trees": booster.num_trees() if booster else 0,
        "feature_names": FEATURE_NAMES,
    }


@app.get("/features")
async def features():
    return {"order": FEATURE_NAMES, "count": N_FEATURES}


if __name__ == "__main__":
    import uvicorn

    load_booster()
    logger.info("ranker service on :%d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
