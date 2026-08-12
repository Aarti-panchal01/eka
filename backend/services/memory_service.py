"""Memory, sessions, messages and goals — Postgres as the record of truth,
Qdrant as the search index.

The two stores can drift (a Qdrant write can fail while the Postgres commit
succeeds). Rather than pretend that's impossible, `Memory.embedded` records
whether the vector landed, so a repair pass can find and re-embed orphans.
Reads never depend on Qdrant being right: list/get always come from Postgres.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.db_models import (
    ChatMessage,
    ChatSession,
    DailyReflection,
    GoalTracking,
    Memory,
    User,
    UserPreferences,
    new_uuid,
    utcnow,
)
from services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


class MemoryService:
    # ---------------------------------------------------------------- users
    async def ensure_user(self, db: AsyncSession, user_id: str) -> User:
        """Get or create a user row.

        There is no auth yet — the frontend supplies a client-generated uuid.
        Every FK in the schema points at users.id, so this has to exist before
        anything else can be written.
        """
        user = await db.get(User, str(user_id))
        if user:
            return user
        user = User(id=str(user_id))
        db.add(user)
        try:
            await db.commit()
        except Exception:
            # Concurrent first request for the same user — one of them wins.
            await db.rollback()
            user = await db.get(User, str(user_id))
            if user:
                return user
            raise
        await db.refresh(user)
        logger.info("Created user %s", user_id)
        return user

    # -------------------------------------------------------------- memory
    async def create_memory(
        self,
        db: AsyncSession,
        user_id: str,
        title: str,
        content: str,
        source: str = "manual",
        topic: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: int = 5,
        user_priority: str = "normal",
    ) -> Memory:
        await self.ensure_user(db, user_id)

        memory = Memory(
            id=new_uuid(),
            user_id=str(user_id),
            title=title[:255],
            content=content,
            source=source,
            topic=topic,
            tags=tags or [],
            importance=max(1, min(10, int(importance))),
            user_priority=user_priority,
            embedded=False,
        )
        db.add(memory)
        await db.commit()
        await db.refresh(memory)

        # Embed the title too — it carries the strongest topical signal.
        embed_text = f"{title}. {content}" if title else content
        vector = await embedding_service.embed(embed_text)
        # An empty vector means no embedding backend was reachable. Never write a
        # placeholder point: it would be indistinguishable from a real one, and
        # `embedded` staying False is what lets reembed_orphans() find and fix it.
        if not vector:
            logger.error(
                "Memory %s saved but NOT indexed — no embedding backend. "
                "Run reembed_orphans() once embeddings recover.", memory.id,
            )
            return memory
        stored = await embedding_service.store_vector(
            point_id=memory.id,
            vector=vector,
            payload=self._payload(memory),
        )
        if stored:
            memory.embedded = True
            await db.commit()
            await db.refresh(memory)
        else:
            logger.warning(
                "Memory %s saved to Postgres but not indexed in Qdrant "
                "(embedded=False)", memory.id,
            )
        return memory

    @staticmethod
    def _payload(memory: Memory) -> Dict:
        return {
            "user_id": memory.user_id,
            "memory_id": memory.id,
            "title": memory.title,
            "content": memory.content,
            "source": memory.source,
            "topic": memory.topic or "",
            "tags": memory.tags or [],
            "importance": memory.importance,
            "user_priority": memory.user_priority,
            "access_count": memory.access_count or 0,
            "created_date": (memory.created_date or utcnow()).isoformat(),
        }

    async def get_memories(
        self,
        db: AsyncSession,
        user_id: str,
        filters: Optional[Dict] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> List[Memory]:
        query = select(Memory).where(Memory.user_id == str(user_id))
        query = self._apply_filters(query, filters or {})
        query = query.order_by(Memory.created_date.desc()).offset(skip).limit(limit)
        return list((await db.execute(query)).scalars().all())

    async def count_memories(
        self, db: AsyncSession, user_id: str, filters: Optional[Dict] = None
    ) -> int:
        query = select(func.count(Memory.id)).where(Memory.user_id == str(user_id))
        query = self._apply_filters(query, filters or {})
        return int((await db.execute(query)).scalar() or 0)

    @staticmethod
    def _apply_filters(query, filters: Dict):
        if topic := filters.get("topic"):
            query = query.where(Memory.topic == topic)
        if priority := filters.get("user_priority"):
            query = query.where(Memory.user_priority == priority)
        if source := filters.get("source"):
            query = query.where(Memory.source == source)
        if date_from := filters.get("date_from"):
            query = query.where(Memory.created_date >= date_from)
        if date_to := filters.get("date_to"):
            query = query.where(Memory.created_date <= date_to)
        if search := filters.get("q"):
            like = f"%{search}%"
            query = query.where(Memory.title.ilike(like) | Memory.content.ilike(like))
        return query

    async def get_memory(
        self, db: AsyncSession, memory_id: str, user_id: Optional[str] = None
    ) -> Optional[Memory]:
        memory = await db.get(Memory, str(memory_id))
        if memory and user_id and memory.user_id != str(user_id):
            return None  # never leak across users
        return memory

    async def update_memory(
        self, db: AsyncSession, memory_id: str, updates: Dict, user_id: Optional[str] = None
    ) -> Optional[Memory]:
        memory = await self.get_memory(db, memory_id, user_id)
        if not memory:
            return None

        content_changed = False
        for field, value in updates.items():
            if value is None or not hasattr(memory, field):
                continue
            if field in ("content", "title") and value != getattr(memory, field):
                content_changed = True
            setattr(memory, field, value)

        await db.commit()
        await db.refresh(memory)

        if content_changed:
            # Re-embed: the old vector no longer represents this memory.
            vector = await embedding_service.embed(f"{memory.title}. {memory.content}")
            if vector:
                await embedding_service.store_vector(
                    memory.id, vector, self._payload(memory)
                )
            else:
                # Stale vector is worse than none — drop it and mark for repair.
                memory.embedded = False
                await db.commit()
                await embedding_service.delete_point(memory.id)
                logger.error("Memory %s edited but could not be re-embedded", memory.id)
        else:
            await embedding_service.update_payload(memory.id, self._payload(memory))
        return memory

    async def update_memory_priority(
        self, db: AsyncSession, user_id: str, memory_id: str, priority: str
    ) -> Optional[Memory]:
        memory = await self.get_memory(db, memory_id, user_id)
        if not memory:
            return None
        memory.user_priority = priority
        await db.commit()
        await db.refresh(memory)
        await embedding_service.update_payload(memory.id, {"user_priority": priority})
        return memory

    async def delete_memory(
        self, db: AsyncSession, user_id: str, memory_id: str
    ) -> bool:
        memory = await self.get_memory(db, memory_id, user_id)
        if not memory:
            return False
        point_id = memory.id
        await db.delete(memory)
        await db.commit()
        # Postgres is the record of truth; a failed Qdrant delete would leave a
        # ghost that can still be retrieved, so log it loudly.
        if not await embedding_service.delete_point(point_id):
            logger.error("Deleted memory %s from Postgres but NOT from Qdrant", point_id)
        return True

    async def search_memories(
        self, user_id: str, query_text: str, limit: int = 20
    ) -> List[Dict]:
        vector = await embedding_service.embed(query_text)
        if not vector:
            logger.warning("Semantic search unavailable — no embedding backend")
            return []
        hits = await embedding_service.search(
            vector=vector, limit=limit, user_id=str(user_id)
        )
        results = []
        for hit in hits:
            payload = hit.get("payload") or {}
            results.append(
                {
                    "id": hit["id"],
                    "title": payload.get("title"),
                    "content": payload.get("content"),
                    "score": hit.get("score", 0.0),
                    "topic": payload.get("topic"),
                    "importance": payload.get("importance"),
                    "user_priority": payload.get("user_priority"),
                    "source": payload.get("source"),
                }
            )
        return results

    async def increment_memory_access(
        self, memory_ids: Sequence[str], db: Optional[AsyncSession] = None
    ) -> None:
        """Bump access_count for retrieved memories (feeds ranker feature f4)."""
        ids = [str(i) for i in memory_ids if i]
        if not ids:
            return

        if db is not None:
            try:
                await db.execute(
                    update(Memory)
                    .where(Memory.id.in_(ids))
                    .values(
                        access_count=Memory.access_count + 1,
                        last_accessed=utcnow(),
                    )
                )
                await db.commit()
            except Exception as exc:
                await db.rollback()
                logger.warning("access_count update failed: %s", exc)

        # Qdrant has no atomic increment, so read-modify-write per point. Small
        # counts (1-5 memories per turn) make this acceptable.
        for memory_id in ids:
            point = await embedding_service.get_point(memory_id)
            if not point:
                continue
            current = int((point.get("payload") or {}).get("access_count", 0) or 0)
            await embedding_service.update_payload(
                memory_id, {"access_count": current + 1}
            )

    async def reembed_orphans(self, db: AsyncSession, user_id: str) -> int:
        """Repair pass: re-index memories whose Qdrant write previously failed."""
        rows = list(
            (
                await db.execute(
                    select(Memory).where(
                        Memory.user_id == str(user_id), Memory.embedded.is_(False)
                    )
                )
            )
            .scalars()
            .all()
        )
        fixed = 0
        for memory in rows:
            vector = await embedding_service.embed(f"{memory.title}. {memory.content}")
            if vector and await embedding_service.store_vector(
                memory.id, vector, self._payload(memory)
            ):
                memory.embedded = True
                fixed += 1
        if fixed:
            await db.commit()
        return fixed

    # ------------------------------------------------------------ sessions
    async def create_session(
        self,
        db: AsyncSession,
        user_id: str,
        mode: str = "founder",
        title: Optional[str] = None,
    ) -> ChatSession:
        await self.ensure_user(db, user_id)
        session = ChatSession(
            id=new_uuid(),
            user_id=str(user_id),
            mode=mode,
            title=title or "New conversation",
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return session

    async def get_session(
        self, db: AsyncSession, session_id: str
    ) -> Optional[ChatSession]:
        return await db.get(ChatSession, str(session_id))

    async def get_sessions(
        self,
        db: AsyncSession,
        user_id: str,
        include_archived: bool = False,
        limit: int = 50,
    ) -> List[ChatSession]:
        query = select(ChatSession).where(ChatSession.user_id == str(user_id))
        if not include_archived:
            query = query.where(ChatSession.archived.is_(False))
        query = query.order_by(ChatSession.updated_date.desc()).limit(limit)
        return list((await db.execute(query)).scalars().all())

    async def update_session(
        self, db: AsyncSession, session_id: str, updates: Dict
    ) -> Optional[ChatSession]:
        session = await self.get_session(db, session_id)
        if not session:
            return None
        for field, value in updates.items():
            if value is not None and hasattr(session, field):
                setattr(session, field, value)
        await db.commit()
        await db.refresh(session)
        return session

    async def archive_session(self, db: AsyncSession, session_id: str) -> bool:
        session = await self.get_session(db, session_id)
        if not session:
            return False
        session.archived = True
        session.is_active = False
        await db.commit()
        return True

    async def get_or_create_session(
        self, db: AsyncSession, user_id: str, session_id: Optional[str], mode: str
    ) -> ChatSession:
        if session_id:
            session = await self.get_session(db, session_id)
            if session:
                return session
        return await self.create_session(db, user_id, mode)

    # ------------------------------------------------------------ messages
    async def get_history(
        self, db: AsyncSession, session_id: str, limit: int = 10
    ) -> List[ChatMessage]:
        """Last N messages, returned oldest-first for prompt building."""
        if not session_id:
            return []
        rows = list(
            (
                await db.execute(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == str(session_id))
                    .order_by(ChatMessage.created_date.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return list(reversed(rows))

    async def get_session_messages(
        self, db: AsyncSession, session_id: str, limit: int = 200
    ) -> List[ChatMessage]:
        rows = (
            await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == str(session_id))
                .order_by(ChatMessage.created_date.asc())
                .limit(limit)
            )
        ).scalars()
        return list(rows.all())

    async def save_message(
        self,
        db: AsyncSession,
        session_id: str,
        user_message: str,
        eka_response: str,
        mode: str,
        sentiment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        complexity: Optional[str] = None,
        user_id: Optional[str] = None,
        retrieved_memory_ids: Optional[List[str]] = None,
        latency_ms: Optional[int] = None,
    ) -> ChatMessage:
        message = ChatMessage(
            id=new_uuid(),
            session_id=str(session_id),
            user_id=str(user_id) if user_id else None,
            user_message=user_message,
            eka_response=eka_response,
            mode=mode,
            sentiment=sentiment,
            tags=tags or [],
            complexity=complexity,
            retrieved_memory_ids=retrieved_memory_ids or [],
            latency_ms=latency_ms,
        )
        db.add(message)

        session = await self.get_session(db, session_id)
        if session:
            session.message_count = (session.message_count or 0) + 1
            session.last_message_preview = user_message[:200]
            session.updated_date = utcnow()
            # Name the conversation after its first message.
            if session.message_count == 1 and session.title == "New conversation":
                session.title = user_message[:60].strip() or "New conversation"

        await db.commit()
        await db.refresh(message)
        return message

    async def get_messages_for_date(
        self, db: AsyncSession, user_id: str, day_start: datetime, day_end: datetime
    ) -> List[ChatMessage]:
        """All of a user's messages in a window — used by insights."""
        rows = (
            await db.execute(
                select(ChatMessage)
                .where(
                    ChatMessage.user_id == str(user_id),
                    ChatMessage.created_date >= day_start,
                    ChatMessage.created_date < day_end,
                )
                .order_by(ChatMessage.created_date.asc())
            )
        ).scalars()
        return list(rows.all())

    # --------------------------------------------------------------- goals
    async def get_active_goals(
        self, db: AsyncSession, user_id: str
    ) -> List[GoalTracking]:
        rows = (
            await db.execute(
                select(GoalTracking)
                .where(
                    GoalTracking.user_id == str(user_id),
                    GoalTracking.status == "active",
                )
                .order_by(GoalTracking.created_date.asc())
            )
        ).scalars()
        return list(rows.all())

    async def get_goals(
        self, db: AsyncSession, user_id: str, status: Optional[str] = None
    ) -> List[GoalTracking]:
        query = select(GoalTracking).where(GoalTracking.user_id == str(user_id))
        if status:
            query = query.where(GoalTracking.status == status)
        rows = (await db.execute(query.order_by(GoalTracking.created_date.desc()))).scalars()
        return list(rows.all())

    async def create_goal(self, db: AsyncSession, user_id: str, data: Dict) -> GoalTracking:
        await self.ensure_user(db, user_id)
        goal = GoalTracking(id=new_uuid(), user_id=str(user_id), **data)
        db.add(goal)
        await db.commit()
        await db.refresh(goal)
        return goal

    async def update_goal(
        self, db: AsyncSession, goal_id: str, updates: Dict
    ) -> Optional[GoalTracking]:
        goal = await db.get(GoalTracking, str(goal_id))
        if not goal:
            return None
        for field, value in updates.items():
            if value is not None and hasattr(goal, field):
                setattr(goal, field, value)
        # Hitting the target completes the goal without a separate call.
        if goal.target_value and goal.current_value >= goal.target_value:
            goal.status = "completed"
        await db.commit()
        await db.refresh(goal)
        return goal

    async def delete_goal(self, db: AsyncSession, goal_id: str) -> bool:
        goal = await db.get(GoalTracking, str(goal_id))
        if not goal:
            return False
        await db.delete(goal)
        await db.commit()
        return True

    # --------------------------------------------------------- reflections
    async def create_reflection(
        self, db: AsyncSession, user_id: str, data: Dict
    ) -> DailyReflection:
        await self.ensure_user(db, user_id)
        reflection = DailyReflection(id=new_uuid(), user_id=str(user_id), **data)
        db.add(reflection)
        await db.commit()
        await db.refresh(reflection)
        return reflection

    async def get_reflections(
        self,
        db: AsyncSession,
        user_id: str,
        limit: int = 30,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[DailyReflection]:
        query = select(DailyReflection).where(DailyReflection.user_id == str(user_id))
        if date_from:
            query = query.where(DailyReflection.date >= date_from)
        if date_to:
            query = query.where(DailyReflection.date <= date_to)
        rows = (
            await db.execute(query.order_by(DailyReflection.date.desc()).limit(limit))
        ).scalars()
        return list(rows.all())

    async def get_reflection(
        self, db: AsyncSession, reflection_id: str
    ) -> Optional[DailyReflection]:
        return await db.get(DailyReflection, str(reflection_id))

    async def update_reflection(
        self, db: AsyncSession, reflection_id: str, updates: Dict
    ) -> Optional[DailyReflection]:
        reflection = await db.get(DailyReflection, str(reflection_id))
        if not reflection:
            return None
        for field, value in updates.items():
            if value is not None and hasattr(reflection, field):
                setattr(reflection, field, value)
        await db.commit()
        await db.refresh(reflection)
        return reflection

    async def delete_reflection(self, db: AsyncSession, reflection_id: str) -> bool:
        reflection = await db.get(DailyReflection, str(reflection_id))
        if not reflection:
            return False
        await db.delete(reflection)
        await db.commit()
        return True

    # --------------------------------------------------------- preferences
    async def get_preferences(
        self, db: AsyncSession, user_id: str
    ) -> UserPreferences:
        row = (
            await db.execute(
                select(UserPreferences).where(UserPreferences.user_id == str(user_id))
            )
        ).scalar_one_or_none()
        if row:
            return row

        await self.ensure_user(db, user_id)
        row = UserPreferences(id=new_uuid(), user_id=str(user_id))
        db.add(row)
        try:
            await db.commit()
            await db.refresh(row)
        except Exception:
            await db.rollback()
            row = (
                await db.execute(
                    select(UserPreferences).where(
                        UserPreferences.user_id == str(user_id)
                    )
                )
            ).scalar_one()
        return row

    async def update_preferences(
        self, db: AsyncSession, user_id: str, updates: Dict
    ) -> UserPreferences:
        prefs = await self.get_preferences(db, user_id)
        for field, value in updates.items():
            if value is not None and hasattr(prefs, field) and field != "user_id":
                setattr(prefs, field, value)
        await db.commit()
        await db.refresh(prefs)
        return prefs


memory_service = MemoryService()
