"""Embeddings + Qdrant Cloud — the substrate of Eka's semantic memory.

Embedding strategy, in priority order:
  1. HF Inference API against the fine-tuned {HF_USERNAME}/eka-embeddings.
  2. HF Inference API against the base BAAI/bge-base-en-v1.5 (works before
     the fine-tune exists — same 768 dims, so vectors stay compatible).
  3. A local sentence-transformers model, if torch happens to be installed.

All three produce real 768-dim semantic vectors, so the Qdrant collection never
needs rebuilding when you graduate from one to the next.

There is deliberately NO synthetic fallback below tier 3. An earlier version
had a hashed bag-of-words tier so that nothing ever raised — but a vector that
isn't semantic silently poisons the memory index: writes and reads look
successful, `embedded=True` gets set, and Eka just quietly recalls the wrong
things forever. Failing visibly is the correct behaviour. embed() returns an
empty list, callers skip the vector write or search, `Memory.embedded` stays
False so reembed_orphans() can repair it later, and /health reports the tier as
"unavailable".
"""

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self._collection_ready = False
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._local_model: Optional[Any] = None
        self._local_tried = False
        self._lock = asyncio.Lock()
        self._active_tier = "unknown"
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------- qdrant
    @property
    def client(self):
        """Lazily built AsyncQdrantClient (import cost only when used)."""
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=settings.QDRANT_TIMEOUT,
            )
        return self._client

    async def ensure_collection(self) -> bool:
        """Create the memories collection if absent. Called on startup."""
        if self._collection_ready:
            return True
        try:
            from qdrant_client.models import (
                Distance,
                OptimizersConfigDiff,
                PayloadSchemaType,
                VectorParams,
            )

            existing = await self.client.get_collections()
            names = {c.name for c in existing.collections}

            if settings.QDRANT_COLLECTION not in names:
                await self.client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                    optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
                )
                logger.info("Created Qdrant collection '%s'", settings.QDRANT_COLLECTION)

                # Payload indexes make the user_id filter cheap. Without them
                # every search scans the whole collection.
                for field, schema in (
                    ("user_id", PayloadSchemaType.KEYWORD),
                    ("user_priority", PayloadSchemaType.KEYWORD),
                    ("topic", PayloadSchemaType.KEYWORD),
                ):
                    try:
                        await self.client.create_payload_index(
                            collection_name=settings.QDRANT_COLLECTION,
                            field_name=field,
                            field_schema=schema,
                        )
                    except Exception as exc:
                        logger.debug("payload index %s: %s", field, exc)
            else:
                logger.info("Qdrant collection '%s' present", settings.QDRANT_COLLECTION)

            self._collection_ready = True
            return True
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Qdrant unavailable (%s) — memory search disabled", exc)
            return False

    async def store_vector(
        self, point_id: str, vector: List[float], payload: Dict[str, Any]
    ) -> bool:
        try:
            from qdrant_client.models import PointStruct

            await self.client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
                wait=False,  # don't block the request on disk flush
            )
            return True
        except Exception as exc:
            logger.warning("store_vector failed for %s: %s", point_id, exc)
            return False

    async def search(
        self,
        vector: List[float],
        limit: int,
        user_id: str,
        exclude_priorities: Optional[List[str]] = None,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Vector search scoped to one user, excluding muted memories."""
        exclude_priorities = exclude_priorities or ["excluded"]
        try:
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                MatchAny,
                MatchValue,
            )

            must = [FieldCondition(key="user_id", match=MatchValue(value=str(user_id)))]
            if topic:
                must.append(FieldCondition(key="topic", match=MatchValue(value=topic)))

            query_filter = Filter(
                must=must,
                must_not=[
                    FieldCondition(
                        key="user_priority", match=MatchAny(any=exclude_priorities)
                    )
                ],
            )

            # qdrant-client renamed search() -> query_points() and DELETED the
            # old method in 1.12+. requirements.txt pins 1.9.1 (which has
            # search), but a looser install gets the new client and every search
            # fails with AttributeError — silently, since we catch broadly here.
            # Support both rather than depend on the pin holding.
            if hasattr(self.client, "query_points"):
                response = await self.client.query_points(
                    collection_name=settings.QDRANT_COLLECTION,
                    query=vector,
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True,
                )
                results = response.points
            else:
                results = await self.client.search(
                    collection_name=settings.QDRANT_COLLECTION,
                    query_vector=vector,
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True,
                )
            return [
                {"id": str(hit.id), "score": float(hit.score), "payload": hit.payload or {}}
                for hit in results
            ]
        except Exception as exc:
            logger.warning("Qdrant search failed: %s", exc)
            return []

    async def update_payload(self, point_id: str, updates: Dict[str, Any]) -> bool:
        """Merge fields into an existing payload without clobbering the rest."""
        try:
            await self.client.set_payload(
                collection_name=settings.QDRANT_COLLECTION,
                payload=updates,
                points=[point_id],
                wait=False,
            )
            return True
        except Exception as exc:
            logger.warning("update_payload failed for %s: %s", point_id, exc)
            return False

    async def batch_update_payload(
        self, point_ids: List[str], updates: Dict[str, Any]
    ) -> bool:
        if not point_ids:
            return True
        try:
            await self.client.set_payload(
                collection_name=settings.QDRANT_COLLECTION,
                payload=updates,
                points=point_ids,
                wait=False,
            )
            return True
        except Exception as exc:
            logger.warning("batch_update_payload failed: %s", exc)
            return False

    async def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        try:
            records = await self.client.retrieve(
                collection_name=settings.QDRANT_COLLECTION,
                ids=[point_id],
                with_payload=True,
            )
            if not records:
                return None
            return {"id": str(records[0].id), "payload": records[0].payload or {}}
        except Exception as exc:
            logger.warning("get_point failed for %s: %s", point_id, exc)
            return None

    async def delete_point(self, point_id: str) -> bool:
        try:
            from qdrant_client.models import PointIdsList

            await self.client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=PointIdsList(points=[point_id]),
                wait=True,  # deletes must be durable — the user asked for it
            )
            return True
        except Exception as exc:
            logger.warning("delete_point failed for %s: %s", point_id, exc)
            return False

    async def count_points(self, user_id: Optional[str] = None) -> int:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = None
            if user_id:
                query_filter = Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=str(user_id)))]
                )
            result = await self.client.count(
                collection_name=settings.QDRANT_COLLECTION,
                count_filter=query_filter,
                exact=False,
            )
            return int(result.count)
        except Exception as exc:
            logger.warning("count_points failed: %s", exc)
            return -1

    # ---------------------------------------------------------- embedding
    async def embed(self, text: str) -> List[float]:
        """Text -> 768-dim semantic vector, or [] if no backend is reachable.

        Never raises. An empty return means "no semantic vector available";
        callers must skip the write or search rather than store a fake one.
        """
        text = (text or "").strip()
        if not text:
            return []

        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        vector = await self._embed_uncached(text)
        if not vector:
            return []  # don't cache a failure — the next call should retry

        self._cache[key] = vector
        if len(self._cache) > settings.EMBED_CACHE_SIZE:
            self._cache.popitem(last=False)
        return vector

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Sequential on purpose — the HF free tier rate-limits parallel bursts."""
        return [await self.embed(text) for text in texts]

    async def _embed_uncached(self, text: str) -> List[float]:
        candidates = []
        if settings.HF_TOKEN and settings.HF_USERNAME:
            candidates.append((settings.embedding_model_id, "finetuned"))
        if settings.HF_TOKEN:
            candidates.append((settings.EMBEDDING_FALLBACK_MODEL, "hf_base"))

        for model_id, tier in candidates:
            vector = await self._hf_embed(model_id, text)
            if vector:
                self._active_tier = tier
                return self._fit_dim(vector)

        vector = await self._local_embed(text)
        if vector:
            self._active_tier = "local"
            return self._fit_dim(vector)

        # No synthetic fallback by design — see the module docstring. An empty
        # vector propagates as "no semantic search this turn", which is honest.
        self._active_tier = "unavailable"
        logger.error(
            "All embedding backends failed — returning no vector. Memory search "
            "is disabled until this recovers. Check HF_TOKEN has the 'Make calls "
            "to Inference Providers' permission, or install sentence-transformers "
            "for a local model."
        )
        return []

    async def _hf_embed(self, model_id: str, text: str) -> Optional[List[float]]:
        """HF Inference API, handling the 503 cold start."""
        url = f"{settings.HF_INFERENCE_BASE}/{model_id}"
        headers = {"Authorization": f"Bearer {settings.HF_TOKEN}"}
        payload = {
            "inputs": text,
            # feature-extraction models need this or a cold model 503s forever
            "options": {"wait_for_model": True, "use_cache": True},
        }

        for attempt in range(settings.HF_COLD_START_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, headers=headers, json=payload)

                if response.status_code == 503:
                    wait = settings.HF_COLD_START_WAIT
                    logger.info(
                        "%s is loading (503), retrying in %ss (%d/%d)",
                        model_id, wait, attempt + 1, settings.HF_COLD_START_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    continue

                if response.status_code == 404:
                    # Repo doesn't exist yet — normal before training finishes.
                    logger.debug("%s not found on the Hub", model_id)
                    return None

                if response.status_code in (401, 403):
                    # Overwhelmingly this is a token scope problem, not a bug.
                    # Say so once, loudly, with the fix.
                    logger.error(
                        "HF Inference refused the token (HTTP %s) for %s. "
                        "Regenerate it at huggingface.co/settings/tokens with "
                        "'Make calls to Inference Providers' enabled (plus read "
                        "+ write for model uploads). Trying a local model next; "
                        "if that is unavailable, memory search is disabled until "
                        "this is fixed. Response: %s",
                        response.status_code, model_id, response.text[:200],
                    )
                    return None

                if response.status_code != 200:
                    logger.debug("%s -> HTTP %s: %s",
                                 model_id, response.status_code, response.text[:200])
                    return None

                return self._flatten(response.json())

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                logger.debug("%s network error: %s", model_id, exc)
                return None
            except Exception as exc:
                logger.debug("%s unexpected error: %s", model_id, exc)
                return None
        return None

    @staticmethod
    def _flatten(data: Any) -> Optional[List[float]]:
        """HF returns [floats], [[floats]], or token-level [[[floats]]]."""
        if isinstance(data, dict):
            if "error" in data:
                return None
            data = data.get("embeddings") or data.get("vector")
        if not isinstance(data, list) or not data:
            return None
        if isinstance(data[0], (int, float)):
            return [float(v) for v in data]
        if isinstance(data[0], list) and data[0] and isinstance(data[0][0], (int, float)):
            # Token-level output: mean-pool it.
            columns = list(zip(*data))
            return [float(sum(col) / len(col)) for col in columns]
        if isinstance(data[0], list) and data[0] and isinstance(data[0][0], list):
            tokens = data[0]
            columns = list(zip(*tokens))
            return [float(sum(col) / len(col)) for col in columns]
        return None

    async def _local_embed(self, text: str) -> Optional[List[float]]:
        """sentence-transformers in-process, if torch is installed."""
        if self._local_tried and self._local_model is None:
            return None

        async with self._lock:
            if not self._local_tried:
                self._local_tried = True
                try:
                    from sentence_transformers import SentenceTransformer

                    model_id = (
                        settings.embedding_model_id
                        if settings.HF_USERNAME
                        else settings.EMBEDDING_FALLBACK_MODEL
                    )
                    logger.info("Loading local embedding model %s", model_id)
                    try:
                        self._local_model = SentenceTransformer(
                            model_id, token=settings.HF_TOKEN or None
                        )
                    except Exception:
                        self._local_model = SentenceTransformer(
                            settings.EMBEDDING_FALLBACK_MODEL
                        )
                except ImportError:
                    logger.debug("sentence-transformers not installed")
                    self._local_model = None
                except Exception as exc:
                    logger.warning("Local embedding model failed to load: %s", exc)
                    self._local_model = None

        if self._local_model is None:
            return None
        try:
            # encode() is blocking; keep the event loop free.
            vector = await asyncio.to_thread(
                self._local_model.encode, text, normalize_embeddings=True
            )
            return [float(v) for v in vector]
        except Exception as exc:
            logger.warning("Local embed failed: %s", exc)
            return None

    @staticmethod
    def _fit_dim(vector: List[float]) -> List[float]:
        """Guard against a model whose dim doesn't match the collection."""
        dim = settings.EMBEDDING_DIM
        if len(vector) == dim:
            return vector
        if len(vector) > dim:
            logger.warning("Embedding %d dims -> truncating to %d", len(vector), dim)
            return vector[:dim]
        logger.warning("Embedding %d dims -> padding to %d", len(vector), dim)
        return vector + [0.0] * (dim - len(vector))

    # ------------------------------------------------------------- health
    async def health(self) -> Dict[str, Any]:
        qdrant_ok = False
        points = -1
        try:
            await self.client.get_collections()
            qdrant_ok = True
            points = await self.count_points()
        except Exception as exc:
            self._last_error = str(exc)

        # "unknown" means nothing has been embedded yet this process, not that
        # anything is wrong. Probe once with a trivial string so /health reports
        # the tier that a real request would actually get. The result is cached,
        # so this costs one API call per process lifetime.
        if self._active_tier == "unknown":
            await self.embed("health check")

        return {
            "qdrant": qdrant_ok,
            "collection": settings.QDRANT_COLLECTION,
            "points": points,
            "embedding_tier": self._active_tier,
            "cache_size": len(self._cache),
            "degraded": self._active_tier == "unavailable",
        }

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None


embedding_service = EmbeddingService()
