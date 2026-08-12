"""Memory reranking — LightGBM over 8 features, with a cosine fallback.

Vector search returns a candidate pool ordered by cosine similarity alone. This
service reorders that pool using signals cosine cannot see: how important the
memory is, whether the user deprioritised it, how stale it is, how often it has
actually been useful before.

Feature order is a contract with training/train_ranker_local.py. Do not reorder:

    f0 cosine_similarity  f1 importance   f2 priority_weight  f3 days_old
    f4 access_count       f5 topic_match  f6 source_type      f7 content_length_norm
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "cosine_similarity",
    "importance",
    "priority_weight",
    "days_old",
    "access_count",
    "topic_match",
    "source_type",
    "content_length_norm",
]
N_FEATURES = len(FEATURE_NAMES)

PRIORITY_WEIGHT = {"high": 3.0, "normal": 2.0, "low": 1.0, "excluded": 0.0}
SOURCE_TYPE = {"chat": 1.0, "upload": 2.0, "manual": 3.0, "reflection": 3.0}


class RankerService:
    def __init__(self) -> None:
        self._booster: Optional[object] = None
        self._load_attempted = False
        self._lock = asyncio.Lock()
        self._remote_ok: Optional[bool] = None
        self._last_tier = "heuristic"

    async def rank(self, features_list: List[List[float]]) -> List[float]:
        """Score each feature row. Higher is better. Never raises."""
        if not features_list:
            return []

        rows = [self._pad(row) for row in features_list]

        if settings.RANKER_SERVE_URL and self._remote_ok is not False:
            scores = await self._remote(rows)
            if scores is not None:
                self._last_tier = "remote"
                return scores

        scores = await self._local(rows)
        if scores is not None:
            self._last_tier = "model"
            return scores

        self._last_tier = "heuristic"
        return [self.heuristic_score(row) for row in rows]

    @staticmethod
    def _pad(row: List[float]) -> List[float]:
        row = [float(v) for v in row[:N_FEATURES]]
        return row + [0.0] * (N_FEATURES - len(row))

    # ------------------------------------------------------------- remote
    async def _remote(self, rows: List[List[float]]) -> Optional[List[float]]:
        try:
            async with httpx.AsyncClient(timeout=settings.SERVICE_TIMEOUT) as client:
                response = await client.post(
                    f"{settings.RANKER_SERVE_URL}/rank", json={"features": rows}
                )
            if response.status_code != 200:
                return None
            scores = response.json().get("scores")
            if isinstance(scores, list) and len(scores) == len(rows):
                self._remote_ok = True
                return [float(s) for s in scores]
        except Exception as exc:
            if self._remote_ok is not False:
                logger.warning("Ranker service unreachable (%s) — using local model", exc)
            self._remote_ok = False
        return None

    # -------------------------------------------------------------- local
    async def _local(self, rows: List[List[float]]) -> Optional[List[float]]:
        booster = await self._get_booster()
        if booster is None:
            return None
        try:
            import numpy as np

            matrix = np.asarray(rows, dtype=np.float32)
            scores = await asyncio.to_thread(booster.predict, matrix)
            return [float(s) for s in scores]
        except Exception as exc:
            logger.warning("Ranker inference failed: %s", exc)
            return None

    async def _get_booster(self):
        if self._load_attempted:
            return self._booster

        async with self._lock:
            if self._load_attempted:
                return self._booster
            self._load_attempted = True

            path = Path(settings.RANKER_MODEL_PATH)
            if not path.exists():
                logger.info(
                    "Ranker model not found at %s — using the cosine heuristic. "
                    "Create it with: python training/train_ranker_local.py",
                    path,
                )
                return None
            try:
                import lightgbm as lgb

                self._booster = await asyncio.to_thread(
                    lgb.Booster, model_file=str(path)
                )
                logger.info(
                    "Ranker loaded from %s (%d trees)",
                    path, self._booster.num_trees(),
                )
            except ImportError:
                logger.info("lightgbm not installed — ranker falls back to heuristic")
                self._booster = None
            except Exception as exc:
                logger.warning("Ranker failed to load: %s", exc)
                self._booster = None
        return self._booster

    # ---------------------------------------------------------- heuristic
    @staticmethod
    def heuristic_score(row: List[float]) -> float:
        """Cosine-dominant linear blend. Mirrors the training scoring function."""
        cosine, importance, priority, days_old, access, topic_match, _source, _length = row
        recency = max(0.0, 1.0 - (days_old / 365.0))
        return (
            1.60 * cosine
            + 0.55 * (importance / 10.0)
            + 0.50 * (priority / 3.0)
            + 0.45 * recency
            + 0.40 * topic_match
            + 0.20 * min(1.0, access / 20.0)
        )

    # ------------------------------------------------------ feature build
    @staticmethod
    def build_features(
        candidate: Dict,
        query_topic: Optional[str] = None,
        now=None,
    ) -> List[float]:
        """Turn a Qdrant hit into the 8-feature row the model expects."""
        from datetime import datetime, timezone

        now = now or datetime.now(timezone.utc)
        payload = candidate.get("payload") or {}

        cosine = float(candidate.get("score", 0.0) or 0.0)
        importance = float(payload.get("importance", 5) or 5)
        priority = PRIORITY_WEIGHT.get(str(payload.get("user_priority", "normal")), 2.0)

        days_old = 0.0
        created = payload.get("created_date")
        if created:
            try:
                if isinstance(created, (int, float)):
                    then = datetime.fromtimestamp(float(created), tz=timezone.utc)
                else:
                    then = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                if then.tzinfo is None:
                    then = then.replace(tzinfo=timezone.utc)
                days_old = max(0.0, min(365.0, (now - then).total_seconds() / 86400.0))
            except (ValueError, TypeError, OSError):
                days_old = 0.0

        access = float(min(50, int(payload.get("access_count", 0) or 0)))

        topic_match = 0.0
        memory_topic = (payload.get("topic") or "").strip().lower()
        if query_topic and memory_topic:
            query_topic = query_topic.strip().lower()
            topic_match = float(
                memory_topic == query_topic
                or query_topic in memory_topic
                or memory_topic in query_topic
            )

        source = SOURCE_TYPE.get(str(payload.get("source", "chat")), 1.0)

        content = payload.get("content") or ""
        # Normalised against a 2000-char reference; longer memories aren't
        # better, so this saturates rather than growing without bound.
        length_norm = min(1.0, len(content) / 2000.0)

        return [
            cosine, importance, priority, days_old,
            access, topic_match, source, length_norm,
        ]

    # ------------------------------------------------------------- health
    async def health(self) -> Dict:
        path = Path(settings.RANKER_MODEL_PATH)
        return {
            "tier": self._last_tier,
            "remote_url": settings.RANKER_SERVE_URL or None,
            "remote_ok": self._remote_ok,
            "model_loaded": self._booster is not None,
            "model_path": str(path),
            "model_exists": path.exists(),
        }

    async def available(self) -> bool:
        if settings.RANKER_SERVE_URL and self._remote_ok:
            return True
        return await self._get_booster() is not None


ranker_service = RankerService()
