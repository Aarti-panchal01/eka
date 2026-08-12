"""Initial schema: the eight Eka tables.

Mirrors backend/models/db_models.py exactly, table for table, column for
column. Reference for anyone diffing this against the models:

* TimestampMixin (created_date indexed, updated_date) is applied to every
  table except chat_messages. chat_messages declares its own created_date
  (indexed, not-null) and has no updated_date column at all — messages are
  immutable once written.
* All primary/foreign keys are `String(36)` uuid strings, not native
  Postgres UUID — ids are generated in Python (uuid4) so the same models
  work against SQLite in tests.
* JSON-ish columns (`tags`, `retrieved_memory_ids`, `key_topics`,
  `achievements`, `challenges`) use the app's JSONField TypeDecorator, which
  is JSONB on Postgres — so here we create them directly as
  postgresql.JSONB since this migration targets Postgres.
* Every composite index declared in a model's __table_args__ is created
  here, as part of the table it belongs to (not deferred to 002):
    - ix_sessions_user_updated      chat_sessions(user_id, updated_date)
    - ix_messages_session_created   chat_messages(session_id, created_date)
    - ix_messages_user_created      chat_messages(user_id, created_date)
    - ix_memories_user_created      memories(user_id, created_date)
    - ix_memories_user_priority     memories(user_id, user_priority)
    - ix_goals_user_status          goal_tracking(user_id, status)
    - ix_reflections_user_date      daily_reflections(user_id, date)
* daily_insights has UniqueConstraint(user_id, date, name="uq_insight_user_date")
  and its `date` column is String(10) (YYYY-MM-DD), not DateTime — "one
  insight per user per day" is enforced by the DB, not application code.
* user_preferences.user_id is unique (one preferences row per user).
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------------------------------------------------------- 1. users
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_date", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_users_email", "users", ["email"], unique=True
    )
    op.create_index("ix_users_created_date", "users", ["created_date"])

    # --------------------------------------------------- 2. chat_sessions
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("message_count", sa.Integer(), nullable=False),
        sa.Column("last_message_preview", sa.String(length=500), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_date", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])
    op.create_index("ix_chat_sessions_mode", "chat_sessions", ["mode"])
    op.create_index("ix_chat_sessions_archived", "chat_sessions", ["archived"])
    op.create_index(
        "ix_chat_sessions_created_date", "chat_sessions", ["created_date"]
    )
    # __table_args__ composite index
    op.create_index(
        "ix_sessions_user_updated",
        "chat_sessions",
        ["user_id", "updated_date"],
    )

    # --------------------------------------------------- 3. chat_messages
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column(
            "session_id",
            sa.String(length=36),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Denormalised from the session so daily-insight queries don't need a join.
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("user_message", sa.Text(), nullable=False),
        sa.Column("eka_response", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column(
            "tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("sentiment", sa.String(length=32), nullable=True),
        sa.Column("complexity", sa.String(length=32), nullable=True),
        sa.Column(
            "retrieved_memory_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        # chat_messages declares created_date itself and has NO updated_date.
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])
    op.create_index("ix_chat_messages_user_id", "chat_messages", ["user_id"])
    op.create_index("ix_chat_messages_mode", "chat_messages", ["mode"])
    op.create_index("ix_chat_messages_sentiment", "chat_messages", ["sentiment"])
    op.create_index(
        "ix_chat_messages_created_date", "chat_messages", ["created_date"]
    )
    # __table_args__ composite indexes
    op.create_index(
        "ix_messages_session_created",
        "chat_messages",
        ["session_id", "created_date"],
    )
    op.create_index(
        "ix_messages_user_created",
        "chat_messages",
        ["user_id", "created_date"],
    )

    # -------------------------------------------------------- 4. memories
    op.create_table(
        "memories",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        # chat | upload | manual | reflection
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("topic", sa.String(length=128), nullable=True),
        sa.Column(
            "tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("importance", sa.Integer(), nullable=False),
        # high | normal | low | excluded  ("excluded" is never retrieved)
        sa.Column("user_priority", sa.String(length=16), nullable=False),
        sa.Column("access_count", sa.Integer(), nullable=False),
        sa.Column("last_accessed", sa.DateTime(timezone=True), nullable=True),
        # False when the Qdrant upsert failed — lets a repair job find orphans.
        sa.Column("embedded", sa.Boolean(), nullable=False),
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_date", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_memories_user_id", "memories", ["user_id"])
    op.create_index("ix_memories_source", "memories", ["source"])
    op.create_index("ix_memories_topic", "memories", ["topic"])
    # NOTE: no single-column index on user_priority. db_models.py deliberately
    # omits index=True there because the auto-generated name would collide with
    # the composite "ix_memories_user_priority" below and abort create_all().
    # The composite covers every real query — priority is always filtered with
    # user_id. Keep these two in sync or alembic autogenerate will see drift.
    op.create_index("ix_memories_created_date", "memories", ["created_date"])
    # __table_args__ composite indexes
    op.create_index(
        "ix_memories_user_created", "memories", ["user_id", "created_date"]
    )
    op.create_index(
        "ix_memories_user_priority", "memories", ["user_id", "user_priority"]
    )

    # --------------------------------------------------- 5. goal_tracking
    op.create_table(
        "goal_tracking",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("goal_name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("target_value", sa.Float(), nullable=False),
        sa.Column("current_value", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=False),
        sa.Column("target_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("streak_days", sa.Integer(), nullable=False),
        # active | completed | paused
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_date", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_goal_tracking_user_id", "goal_tracking", ["user_id"])
    op.create_index("ix_goal_tracking_status", "goal_tracking", ["status"])
    op.create_index(
        "ix_goal_tracking_created_date", "goal_tracking", ["created_date"]
    )
    # __table_args__ composite index
    op.create_index(
        "ix_goals_user_status", "goal_tracking", ["user_id", "status"]
    )

    # ----------------------------------------------- 6. daily_reflections
    op.create_table(
        "daily_reflections",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mood", sa.String(length=32), nullable=True),
        sa.Column("challenges_faced", sa.Text(), nullable=True),
        sa.Column("learnings", sa.Text(), nullable=True),
        sa.Column("gratitude", sa.Text(), nullable=True),
        sa.Column("eka_commentary", sa.Text(), nullable=True),
        sa.Column("mode_used", sa.String(length=32), nullable=True),
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_date", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_daily_reflections_user_id", "daily_reflections", ["user_id"]
    )
    op.create_index("ix_daily_reflections_date", "daily_reflections", ["date"])
    op.create_index(
        "ix_daily_reflections_created_date",
        "daily_reflections",
        ["created_date"],
    )
    # __table_args__ composite index
    op.create_index(
        "ix_reflections_user_date", "daily_reflections", ["user_id", "date"]
    )

    # -------------------------------------------------- 7. daily_insights
    op.create_table(
        "daily_insights",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Stored as a date-only string (YYYY-MM-DD), NOT DateTime, so "one
        # insight per user per day" is enforceable by a unique constraint.
        sa.Column("date", sa.String(length=10), nullable=False),
        sa.Column("alignment_score", sa.Float(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "key_topics", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("mood_trend", sa.String(length=32), nullable=True),
        sa.Column(
            "achievements", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "challenges", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("message_count", sa.Integer(), nullable=False),
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_date", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "date", name="uq_insight_user_date"),
    )
    op.create_index("ix_daily_insights_user_id", "daily_insights", ["user_id"])
    op.create_index("ix_daily_insights_date", "daily_insights", ["date"])
    op.create_index(
        "ix_daily_insights_created_date", "daily_insights", ["created_date"]
    )

    # ------------------------------------------------ 8. user_preferences
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("default_mode", sa.String(length=32), nullable=False),
        sa.Column("voice_enabled", sa.Boolean(), nullable=False),
        sa.Column("playback_speed", sa.Float(), nullable=False),
        sa.Column("theme_accent", sa.String(length=32), nullable=False),
        sa.Column("always_listening", sa.Boolean(), nullable=False),
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_date", sa.DateTime(timezone=True), nullable=False),
    )
    # user_id is unique (one preferences row per user) AND indexed, which
    # SQLAlchemy renders as a single unique index rather than a separate
    # UNIQUE constraint + index.
    op.create_index(
        "ix_user_preferences_user_id", "user_preferences", ["user_id"], unique=True
    )
    op.create_index(
        "ix_user_preferences_created_date",
        "user_preferences",
        ["created_date"],
    )


def downgrade() -> None:
    # Reverse dependency order. Dropping a table drops its own indexes,
    # constraints, and unique constraints automatically in Postgres, so no
    # explicit drop_index/drop_constraint calls are needed here.
    op.drop_table("user_preferences")
    op.drop_table("daily_insights")
    op.drop_table("daily_reflections")
    op.drop_table("goal_tracking")
    op.drop_table("memories")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("users")
