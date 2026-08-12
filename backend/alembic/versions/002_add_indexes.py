"""Additional performance indexes not already declared on the models.

Reasoning for the split between this migration and 001_initial_schema:

Every composite index requested for "phase 9 performance indexes" that is
ALSO declared in a model's __table_args__ in db_models.py was already
created as part of that table's own CREATE TABLE step in 001_initial_schema,
because a table's indexes belong with its own definition, not bolted on
later. That covers:

    - memories(user_id, created_date)          -> ix_memories_user_created
    - memories(user_id, user_priority)         -> ix_memories_user_priority
    - chat_messages(session_id, created_date)  -> ix_messages_session_created
    - chat_messages(user_id, created_date)     -> ix_messages_user_created
    - chat_sessions(user_id, updated_date)     -> ix_sessions_user_updated
    - goal_tracking(user_id, status)           -> ix_goals_user_status
    - daily_reflections(user_id, date)         -> ix_reflections_user_date

Creating any of those again here would fail with "relation already exists",
so this migration is intentionally left with only the composite indexes that
db_models.py does NOT declare anywhere, but that the query patterns in the
app actually need:

    - chat_messages(user_id, sentiment)  — daily-insight / mood-trend queries
      filter a user's messages by sentiment; the model only indexes
      `sentiment` alone (ix_chat_messages_sentiment), which doesn't help a
      per-user filter efficiently.
    - memories(user_id, topic)           — "memories for this user about this
      topic" lookups; the model only indexes `topic` alone
      (ix_memories_topic), same problem as above.

daily_insights(user_id, date) is intentionally skipped: that pair is already
covered by the UniqueConstraint("user_id", "date", name="uq_insight_user_date")
created in 001, and Postgres uses a unique constraint's backing index to
satisfy equality/range lookups on (user_id, date) just fine — a second,
non-unique index on the same columns would be pure duplication.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "002_indexes"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_messages_user_sentiment",
        "chat_messages",
        ["user_id", "sentiment"],
    )
    op.create_index(
        "ix_memories_user_topic",
        "memories",
        ["user_id", "topic"],
    )


def downgrade() -> None:
    op.drop_index("ix_memories_user_topic", table_name="memories")
    op.drop_index("ix_messages_user_sentiment", table_name="chat_messages")
