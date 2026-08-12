"""SQLAlchemy 2.0 models — the eight tables behind Eka.

UUID primary keys are generated in Python (uuid4) rather than by the database,
so the same models work on Postgres and on SQLite in tests, and so a Qdrant
point can be created with the same id before the row is committed.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator

from database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


class JSONField(TypeDecorator):
    """JSONB on Postgres, plain JSON everywhere else."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


# Every id is a uuid string; 36 chars covers the canonical form.
UUID_LEN = 36


class TimestampMixin:
    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    updated_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


# ============================================================== 1. users
class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))

    sessions = relationship(
        "ChatSession", back_populates="user", cascade="all, delete-orphan"
    )
    memories = relationship(
        "Memory", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.id} {self.email or '(anonymous)'}>"


# ====================================================== 2. chat_sessions
class ChatSession(Base, TimestampMixin):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(255), default="New conversation")
    mode: Mapped[str] = mapped_column(String(32), default="founder", index=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    last_message_preview: Mapped[Optional[str]] = mapped_column(String(500))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    user = relationship("User", back_populates="sessions")
    messages = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_sessions_user_updated", "user_id", "updated_date"),
    )

    def __repr__(self) -> str:
        return f"<ChatSession {self.id} mode={self.mode} msgs={self.message_count}>"


# ====================================================== 3. chat_messages
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    # Denormalised from the session so daily-insight queries don't need a join.
    user_id: Mapped[Optional[str]] = mapped_column(String(UUID_LEN), index=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    eka_response: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(32), default="founder", index=True)
    tags: Mapped[Optional[list]] = mapped_column(JSONField, default=list)
    sentiment: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    complexity: Mapped[Optional[str]] = mapped_column(String(32))
    retrieved_memory_ids: Mapped[Optional[list]] = mapped_column(JSONField, default=list)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )

    session = relationship("ChatSession", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_date"),
        Index("ix_messages_user_created", "user_id", "created_date"),
    )

    def __repr__(self) -> str:
        return f"<ChatMessage {self.id} mode={self.mode}>"


# =========================================================== 4. memories
class Memory(Base, TimestampMixin):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # chat | upload | manual | reflection
    source: Mapped[str] = mapped_column(String(32), default="chat", index=True)
    topic: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    tags: Mapped[Optional[list]] = mapped_column(JSONField, default=list)
    importance: Mapped[int] = mapped_column(Integer, default=5)
    # high | normal | low | excluded  ("excluded" is never retrieved)
    # No index=True here: it would auto-generate an index named
    # "ix_memories_user_priority", colliding with the composite index of the
    # same name in __table_args__ below and aborting create_all() partway.
    # The composite (user_id, user_priority) covers every real query anyway —
    # priority is only ever filtered together with user_id.
    user_priority: Mapped[str] = mapped_column(String(16), default="normal")
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    # False when the Qdrant upsert failed — lets a repair job find orphans.
    embedded: Mapped[bool] = mapped_column(Boolean, default=False)

    user = relationship("User", back_populates="memories")

    __table_args__ = (
        Index("ix_memories_user_created", "user_id", "created_date"),
        Index("ix_memories_user_priority", "user_id", "user_priority"),
    )

    def __repr__(self) -> str:
        return f"<Memory {self.id} '{self.title[:30]}' imp={self.importance}>"


# ======================================================= 5. goal_tracking
class GoalTracking(Base, TimestampMixin):
    __tablename__ = "goal_tracking"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    category: Mapped[str] = mapped_column(String(64), default="general")
    goal_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    target_value: Mapped[float] = mapped_column(Float, default=1.0)
    current_value: Mapped[float] = mapped_column(Float, default=0.0)
    unit: Mapped[str] = mapped_column(String(32), default="completion")
    target_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    streak_days: Mapped[int] = mapped_column(Integer, default=0)
    # active | completed | paused
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)

    __table_args__ = (
        Index("ix_goals_user_status", "user_id", "status"),
    )

    @property
    def progress_pct(self) -> float:
        if not self.target_value:
            return 0.0
        return round(min(100.0, (self.current_value / self.target_value) * 100), 1)

    def __repr__(self) -> str:
        return f"<Goal {self.id} '{self.goal_name}' {self.progress_pct}%>"


# =================================================== 6. daily_reflections
class DailyReflection(Base, TimestampMixin):
    __tablename__ = "daily_reflections"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    mood: Mapped[Optional[str]] = mapped_column(String(32))
    challenges_faced: Mapped[Optional[str]] = mapped_column(Text)
    learnings: Mapped[Optional[str]] = mapped_column(Text)
    gratitude: Mapped[Optional[str]] = mapped_column(Text)
    eka_commentary: Mapped[Optional[str]] = mapped_column(Text)
    mode_used: Mapped[Optional[str]] = mapped_column(String(32))

    __table_args__ = (
        Index("ix_reflections_user_date", "user_id", "date"),
    )

    def __repr__(self) -> str:
        return f"<DailyReflection {self.id} {self.date:%Y-%m-%d} mood={self.mood}>"


# ====================================================== 7. daily_insights
class DailyInsight(Base, TimestampMixin):
    __tablename__ = "daily_insights"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # Stored as a date-only string (YYYY-MM-DD) so "one insight per user per
    # day" is enforceable by the database rather than by application code.
    date: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    alignment_score: Mapped[float] = mapped_column(Float, default=0.0)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    key_topics: Mapped[Optional[list]] = mapped_column(JSONField, default=list)
    mood_trend: Mapped[Optional[str]] = mapped_column(String(32))
    achievements: Mapped[Optional[list]] = mapped_column(JSONField, default=list)
    challenges: Mapped[Optional[list]] = mapped_column(JSONField, default=list)
    message_count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_insight_user_date"),
    )

    def __repr__(self) -> str:
        return f"<DailyInsight {self.date} score={self.alignment_score}>"


# ==================================================== 8. user_preferences
class UserPreferences(Base, TimestampMixin):
    __tablename__ = "user_preferences"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(UUID_LEN),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    default_mode: Mapped[str] = mapped_column(String(32), default="founder")
    voice_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    playback_speed: Mapped[float] = mapped_column(Float, default=1.0)
    theme_accent: Mapped[str] = mapped_column(String(32), default="amber")
    always_listening: Mapped[bool] = mapped_column(Boolean, default=False)

    def __repr__(self) -> str:
        return f"<UserPreferences user={self.user_id} mode={self.default_mode}>"


ALL_MODELS = (
    User,
    ChatSession,
    ChatMessage,
    Memory,
    GoalTracking,
    DailyReflection,
    DailyInsight,
    UserPreferences,
)
