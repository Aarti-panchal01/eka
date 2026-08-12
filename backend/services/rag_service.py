"""The RAG pipeline — what happens between a user's message and Eka's reply.

    1  classify complexity          ->  how much context to spend
    2  look up the retrieval budget
    3  load conversation history
    4  embed the query
    5  vector search a candidate pool
    6  rerank the pool, keep the top N
    7  load active goals
    8  build the prompt
    9  generate
    10 classify the reply's sentiment
    11 extract tags
    12 persist the exchange
    13 maybe distil a new memory from it
    14 bump access counts on what was used
    15 return

Every step from 1 to 7 and 10 to 14 is allowed to fail. Only steps 8, 9 and 15
are load-bearing: as long as the LLM answers, the user gets a reply. Whatever
degraded is reported in ChatResponse.degraded so the failure is visible rather
than silent.
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from config import PROMPTS_DIR, VALID_MODES, settings
from models.db_models import ChatMessage, GoalTracking
from models.schemas import ChatRequest, ChatResponse, RetrievedMemory
from services.complexity_service import complexity_service
from services.embedding_service import embedding_service
from services.llm_service import llm_service
from services.memory_service import memory_service
from services.ranker_service import ranker_service
from services.safety_service import safety_service
from services.sentiment_service import sentiment_service

logger = logging.getLogger(__name__)

# How much context each complexity tier is worth.
#   memories = how many retrieved memories reach the prompt
#   history  = how many past turns are replayed
#   pool     = how many candidates the reranker gets to choose from
CONTEXT_CONFIG = {
    "simple": {"memories": 1, "history": 2, "pool": 10},
    "normal": {"memories": 2, "history": 3, "pool": 15},
    "complex": {"memories": 3, "history": 5, "pool": 25},
    "deep": {"memories": 5, "history": 7, "pool": 40},
}

STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was",
    "one", "our", "out", "his", "has", "had", "him", "she", "they", "them",
    "this", "that", "with", "have", "from", "your", "what", "when", "where",
    "which", "would", "could", "should", "about", "there", "their", "been",
    "were", "will", "just", "like", "than", "then", "into", "over", "very",
    "much", "some", "more", "most", "only", "even", "also", "because", "while",
    "after", "before", "being", "does", "doing", "don't", "i'm", "i've", "it's",
    "that's", "really", "think", "know", "feel", "want", "need", "make", "made",
    "going", "still", "every", "always", "never", "thing", "things", "something",
    "anything", "myself", "yourself", "how", "why", "who", "get", "got", "let",
}


class RAGService:
    def __init__(self) -> None:
        self._persona_cache: Dict[str, str] = {}

    # =============================================================== main
    async def process_message(
        self, request: ChatRequest, db: AsyncSession
    ) -> ChatResponse:
        started = time.perf_counter()
        degraded: List[str] = []
        mode = request.mode if request.mode in VALID_MODES else "founder"

        # --- 0. session -------------------------------------------------
        session = await memory_service.get_or_create_session(
            db, request.user_id, request.session_id, mode
        )
        # Switching mode mid-session should stick.
        if session.mode != mode:
            session.mode = mode
            await db.commit()

        # --- 1. complexity ---------------------------------------------
        try:
            complexity, complexity_confidence = await complexity_service.classify(
                request.message
            )
            if not await complexity_service.available():
                degraded.append("complexity:heuristic")
        except Exception as exc:
            logger.warning("Complexity step failed: %s", exc)
            complexity, complexity_confidence = complexity_service.heuristic(
                request.message
            )
            degraded.append("complexity:heuristic")

        # --- 2. budget --------------------------------------------------
        config = CONTEXT_CONFIG.get(complexity, CONTEXT_CONFIG["normal"])

        # --- 3 & 4. history and query embedding, concurrently -----------
        # Independent: one hits Postgres, the other hits an embedding backend.
        history_task = asyncio.create_task(
            self._safe_history(db, session.id, config["history"])
        )
        embed_task = asyncio.create_task(self._safe_embed(request.message))
        history, (vector, embed_ok) = await asyncio.gather(history_task, embed_task)
        if not embed_ok:
            degraded.append("embedding")

        # --- 5. vector search -------------------------------------------
        candidates: List[Dict] = []
        if vector and any(vector):
            try:
                candidates = await embedding_service.search(
                    vector=vector, limit=config["pool"], user_id=request.user_id
                )
            except Exception as exc:
                logger.warning("Vector search failed: %s", exc)
                degraded.append("qdrant")
        if not candidates and embed_ok:
            # Not an error — a new user simply has no memories yet.
            logger.debug("No memory candidates for user %s", request.user_id)

        # --- 6. rerank ---------------------------------------------------
        query_topic = self._infer_topic(request.message)
        memories = await self._rerank(candidates, query_topic, config["memories"])
        if candidates and not await ranker_service.available():
            degraded.append("ranker:heuristic")

        # --- 7. goals ----------------------------------------------------
        try:
            goals = await memory_service.get_active_goals(db, request.user_id)
        except Exception as exc:
            logger.warning("Goal load failed: %s", exc)
            goals = []

        # --- 8. prompt ---------------------------------------------------
        persona = self.load_persona_prompt(mode)
        user_prompt = self.build_prompt(
            message=request.message,
            memories=memories,
            history=history,
            goals=goals,
        )

        # --- 9. generate -------------------------------------------------
        response_text = await llm_service.generate(
            prompt=user_prompt, mode=mode, system=persona
        )
        llm_backend = llm_service._last_backend

        # Reflection mode is instructed to never advise and only ask questions.
        # That is the right therapeutic stance and the wrong crisis response, so
        # resource information is appended outside the persona's control.
        # Keyword-based and local: no network failure can switch it off.
        response_text = safety_service.annotate(request.message, response_text)
        if llm_backend == "none":
            degraded.append("llm")
        elif settings.LLM_MODE != "groq" and llm_backend == "groq":
            degraded.append(f"llm:{settings.LLM_MODE}->groq")

        # --- 10 & 11. sentiment and tags ---------------------------------
        try:
            sentiment = await sentiment_service.classify_label(response_text)
            if not await sentiment_service.available():
                degraded.append("sentiment:heuristic")
        except Exception as exc:
            logger.warning("Sentiment step failed: %s", exc)
            sentiment = "neutral"
            degraded.append("sentiment:heuristic")

        tags = self.extract_tags(request.message)

        # --- 12. persist -------------------------------------------------
        latency_ms = int((time.perf_counter() - started) * 1000)
        message_id = None
        try:
            saved = await memory_service.save_message(
                db=db,
                session_id=session.id,
                user_message=request.message,
                eka_response=response_text,
                mode=mode,
                sentiment=sentiment,
                tags=tags,
                complexity=complexity,
                user_id=request.user_id,
                retrieved_memory_ids=[m.id for m in memories],
                latency_ms=latency_ms,
            )
            message_id = saved.id
        except Exception as exc:
            logger.error("Failed to persist message: %s", exc)
            degraded.append("persistence")

        # --- 13 & 14. background work ------------------------------------
        # The user already has their reply; don't make them wait for this.
        memory_created = False
        if self._worth_remembering(request.message):
            memory_created = await self._create_auto_memory(
                db, request.user_id, request.message, tags, query_topic, complexity
            )
        if memories:
            asyncio.create_task(
                self._safe_increment([m.id for m in memories])
            )

        # --- 15. respond --------------------------------------------------
        return ChatResponse(
            response=response_text,
            session_id=session.id,
            message_id=message_id,
            mode=mode,
            complexity=complexity,
            sentiment=sentiment,
            tags=tags,
            retrieved_memories=memories,
            memory_created=memory_created,
            latency_ms=int((time.perf_counter() - started) * 1000),
            llm_backend=llm_backend,
            degraded=sorted(set(degraded)),
        )

    # ====================================================== prompt build
    def load_persona_prompt(self, mode: str) -> str:
        """Read and cache backend/prompts/<mode>.txt."""
        if mode in self._persona_cache:
            return self._persona_cache[mode]

        path = PROMPTS_DIR / f"{mode}.txt"
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.error("Persona prompt missing: %s", path)
            text = (
                "You are Eka, a thoughtful long-term companion. Be direct, "
                "specific, and brief. End with one question."
            )
        self._persona_cache[mode] = text
        return text

    def build_prompt(
        self,
        message: str,
        memories: Optional[List[RetrievedMemory]] = None,
        history: Optional[List[ChatMessage]] = None,
        goals: Optional[List[GoalTracking]] = None,
    ) -> str:
        """Assemble the user-turn payload. The persona goes in as `system`."""
        sections: List[str] = []

        if memories:
            lines = []
            for index, memory in enumerate(memories, start=1):
                content = (memory.content or "").strip().replace("\n", " ")
                if len(content) > 400:
                    content = content[:400].rsplit(" ", 1)[0] + "..."
                title = (memory.title or "").strip()
                lines.append(f"{index}. {title + ': ' if title else ''}{content}")
            sections.append(
                "RELEVANT CONTEXT — things you already know about this person:\n"
                + "\n".join(lines)
            )

        if history:
            lines = []
            for turn in history:
                user_text = (turn.user_message or "").strip().replace("\n", " ")
                eka_text = (turn.eka_response or "").strip().replace("\n", " ")
                if len(user_text) > 300:
                    user_text = user_text[:300] + "..."
                if len(eka_text) > 300:
                    eka_text = eka_text[:300] + "..."
                lines.append(f"User: {user_text}\nYou: {eka_text}")
            sections.append("RECENT CONVERSATION:\n" + "\n\n".join(lines))

        if goals:
            lines = []
            for goal in goals[:5]:
                progress = ""
                if goal.target_value:
                    pct = min(100, (goal.current_value / goal.target_value) * 100)
                    progress = f" ({pct:.0f}% of {goal.target_value:g} {goal.unit})"
                lines.append(f"- {goal.goal_name}{progress}")
            sections.append("CURRENT GOALS:\n" + "\n".join(lines))

        sections.append(f"User: {message}")
        return "\n\n".join(sections)

    # =========================================================== helpers
    async def _safe_history(
        self, db: AsyncSession, session_id: str, limit: int
    ) -> List[ChatMessage]:
        if limit <= 0:
            return []
        try:
            return await memory_service.get_history(db, session_id, limit)
        except Exception as exc:
            logger.warning("History load failed: %s", exc)
            return []

    async def _safe_embed(self, text: str) -> Tuple[List[float], bool]:
        try:
            vector = await embedding_service.embed(text)
            # Empty vector == no backend reachable. There is no synthetic tier
            # any more, so "got a vector" is exactly "embeddings are healthy".
            return vector, bool(vector)
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return [], False

    async def _rerank(
        self, candidates: List[Dict], query_topic: Optional[str], keep: int
    ) -> List[RetrievedMemory]:
        """Score the candidate pool and return the top `keep` as memories."""
        if not candidates:
            return []

        try:
            features = [
                ranker_service.build_features(candidate, query_topic)
                for candidate in candidates
            ]
            scores = await ranker_service.rank(features)
        except Exception as exc:
            logger.warning("Rerank failed (%s) — falling back to cosine order", exc)
            scores = [candidate.get("score", 0.0) for candidate in candidates]

        if len(scores) != len(candidates):
            scores = [candidate.get("score", 0.0) for candidate in candidates]

        ordered = sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )[: max(0, keep)]

        results: List[RetrievedMemory] = []
        for candidate, score in ordered:
            payload = candidate.get("payload") or {}
            results.append(
                RetrievedMemory(
                    id=candidate["id"],
                    title=payload.get("title"),
                    content=payload.get("content"),
                    score=float(candidate.get("score", 0.0)),
                    rank_score=float(score),
                    topic=payload.get("topic"),
                    importance=payload.get("importance"),
                )
            )
        return results

    def extract_tags(self, text: str) -> List[str]:
        """Cheap keyword extraction — frequency over meaningful words."""
        words = re.findall(r"[a-zA-Z][a-zA-Z'\-]{3,}", (text or "").lower())
        counts: Dict[str, int] = {}
        for word in words:
            word = word.strip("'-")
            if len(word) < 4 or word in STOPWORDS:
                continue
            counts[word] = counts.get(word, 0) + 1

        # Frequency first, then earliest appearance — keeps the ordering stable.
        order = {word: index for index, word in enumerate(words)}
        ranked = sorted(
            counts.items(), key=lambda kv: (-kv[1], order.get(kv[0], 1e9))
        )
        return [word for word, _count in ranked[: settings.MAX_TAGS]]

    def _infer_topic(self, text: str) -> Optional[str]:
        """First tag doubles as the query topic for the ranker's f5 feature."""
        tags = self.extract_tags(text)
        return tags[0] if tags else None

    @staticmethod
    def _worth_remembering(message: str) -> bool:
        """Not every long message is worth a permanent memory."""
        text = (message or "").strip()
        if len(text) < settings.AUTO_MEMORY_MIN_CHARS:
            return False
        # A wall of text that's all question and no disclosure isn't a fact
        # about the user — it's a request. Don't archive it.
        if text.count("?") >= 3 and len(text.split()) < 40:
            return False
        # Needs some first-person content to be about the user at all.
        return bool(re.search(r"\b(i|my|me|we|our)\b", text.lower()))

    async def _create_auto_memory(
        self,
        db: AsyncSession,
        user_id: str,
        message: str,
        tags: List[str],
        topic: Optional[str],
        complexity: str,
    ) -> bool:
        try:
            title = message.strip().split("\n")[0][:80]
            if len(title) < len(message.strip().split("\n")[0]):
                title += "..."
            # Deeper disclosures matter more later; importance feeds the ranker.
            importance = {"simple": 3, "normal": 4, "complex": 6, "deep": 8}.get(
                complexity, 5
            )
            await memory_service.create_memory(
                db=db,
                user_id=user_id,
                title=title,
                content=message.strip(),
                source="chat",
                topic=topic,
                tags=tags,
                importance=importance,
                user_priority="normal",
            )
            return True
        except Exception as exc:
            logger.warning("Auto-memory creation failed: %s", exc)
            return False

    @staticmethod
    async def _safe_increment(memory_ids: List[str]) -> None:
        try:
            await memory_service.increment_memory_access(memory_ids)
        except Exception as exc:
            logger.debug("access_count increment failed: %s", exc)


rag_service = RAGService()
