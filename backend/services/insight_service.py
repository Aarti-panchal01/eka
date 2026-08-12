"""Daily and weekly insights — what Eka noticed about your day.

An insight is derived, never authored: summary from the summarizer model, mood
from the sentiments already stored on each message, alignment from how much the
day's conversation touched the user's stated goals.

Insights are cached in the daily_insights table (unique per user per day) and
recomputed only when the day is still in progress.
"""

import logging
import re
from collections import Counter
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.db_models import ChatMessage, DailyInsight, new_uuid
from services.llm_service import llm_service
from services.memory_service import memory_service
from services.rag_service import STOPWORDS

logger = logging.getLogger(__name__)

MIN_MESSAGES = 3
POSITIVE_SENTIMENTS = {"positive", "motivated"}
NEGATIVE_SENTIMENTS = {"negative", "anxious"}


class InsightService:
    # ------------------------------------------------------------- daily
    async def generate_daily_insight(
        self,
        user_id: str,
        day: date_cls,
        db: AsyncSession,
        force: bool = False,
    ) -> Optional[DailyInsight]:
        """Build (or return the cached) insight for one day.

        Returns None when there isn't enough conversation to say anything
        honest — fewer than 3 messages is noise, not a pattern.
        """
        date_key = day.isoformat()

        existing = (
            await db.execute(
                select(DailyInsight).where(
                    DailyInsight.user_id == str(user_id),
                    DailyInsight.date == date_key,
                )
            )
        ).scalar_one_or_none()

        today = datetime.now(timezone.utc).date()
        if existing and not force and day < today:
            # A past day can't gain new messages — the cache is final.
            return existing

        start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        messages = await memory_service.get_messages_for_date(db, user_id, start, end)

        if len(messages) < MIN_MESSAGES:
            logger.info(
                "Only %d messages on %s for %s — not enough for an insight",
                len(messages), date_key, user_id,
            )
            return existing

        user_texts = [m.user_message for m in messages if m.user_message]
        combined = " ".join(user_texts)[:1000]

        summary = await self._summarize(combined)
        mood_trend = self._mood_trend(messages)
        goals = await memory_service.get_active_goals(db, user_id)
        alignment = self._alignment_score(goals, user_texts, len(messages))
        key_topics = self._key_topics(user_texts)
        achievements = self._extract_by_sentiment(messages, POSITIVE_SENTIMENTS)
        challenges = self._extract_by_sentiment(messages, NEGATIVE_SENTIMENTS)

        if existing:
            insight = existing
        else:
            insight = DailyInsight(
                id=new_uuid(), user_id=str(user_id), date=date_key
            )
            db.add(insight)

        insight.alignment_score = alignment
        insight.summary = summary
        insight.key_topics = key_topics
        insight.mood_trend = mood_trend
        insight.achievements = achievements
        insight.challenges = challenges
        insight.message_count = len(messages)

        try:
            await db.commit()
            await db.refresh(insight)
        except Exception as exc:
            # Another request raced us to the same (user, date) unique key.
            await db.rollback()
            logger.warning("Insight commit failed (%s) — re-reading", exc)
            insight = (
                await db.execute(
                    select(DailyInsight).where(
                        DailyInsight.user_id == str(user_id),
                        DailyInsight.date == date_key,
                    )
                )
            ).scalar_one_or_none()
        return insight

    # ------------------------------------------------------------ weekly
    async def generate_weekly_summary(
        self, user_id: str, week_start: date_cls, db: AsyncSession
    ) -> Dict:
        """Aggregate seven daily insights. Generates any that are missing."""
        daily: List[DailyInsight] = []
        for offset in range(7):
            day = week_start + timedelta(days=offset)
            insight = await self.generate_daily_insight(user_id, day, db)
            if insight:
                daily.append(insight)

        if not daily:
            return {
                "user_id": str(user_id),
                "week_start": week_start.isoformat(),
                "week_end": (week_start + timedelta(days=6)).isoformat(),
                "days_with_data": 0,
                "avg_alignment_score": 0.0,
                "mood_trend": None,
                "key_topics": [],
                "total_messages": 0,
                "daily": [],
            }

        scores = [d.alignment_score or 0.0 for d in daily]
        trends = [d.mood_trend for d in daily if d.mood_trend]
        topic_counter: Counter = Counter()
        for insight in daily:
            topic_counter.update(insight.key_topics or [])

        return {
            "user_id": str(user_id),
            "week_start": week_start.isoformat(),
            "week_end": (week_start + timedelta(days=6)).isoformat(),
            "days_with_data": len(daily),
            "avg_alignment_score": round(sum(scores) / len(scores), 1),
            "mood_trend": Counter(trends).most_common(1)[0][0] if trends else None,
            "key_topics": [topic for topic, _count in topic_counter.most_common(8)],
            "total_messages": sum(d.message_count or 0 for d in daily),
            "daily": daily,
        }

    # ---------------------------------------------------------- summarize
    async def _summarize(self, text: str) -> str:
        """Summarizer model -> LLM -> truncation. Always returns something."""
        if not text.strip():
            return ""

        if settings.HF_TOKEN:
            model_id = (
                settings.summarizer_model_id
                if settings.HF_USERNAME
                else settings.SUMMARIZER_FALLBACK_MODEL
            )
            result = await self._hf_summarize(model_id, text)
            if result:
                return result
            if settings.HF_USERNAME:
                result = await self._hf_summarize(
                    settings.SUMMARIZER_FALLBACK_MODEL, text
                )
                if result:
                    return result

        # The chat LLM is already up; it summarizes fine.
        try:
            summary = await llm_service.generate(
                prompt=(
                    "Summarize what this person talked about today in 2-3 "
                    "sentences. Write in third person, present tense. No "
                    "preamble, no advice.\n\n" + text
                ),
                mode="reflection",
                system="You write concise, factual summaries. Nothing else.",
                temperature=0.3,
                max_tokens=160,
            )
            if summary and "trouble reaching my language model" not in summary:
                return summary.strip()
        except Exception as exc:
            logger.warning("LLM summarization failed: %s", exc)

        # Last resort: the first two sentences of what they actually said.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return " ".join(sentences[:2])[:300]

    async def _hf_summarize(self, model_id: str, text: str) -> Optional[str]:
        url = f"{settings.HF_INFERENCE_BASE}/{model_id}"
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {settings.HF_TOKEN}"},
                    json={
                        "inputs": f"summarize: {text}",
                        "parameters": {"max_length": 100, "min_length": 30},
                        "options": {"wait_for_model": True},
                    },
                )
            if response.status_code != 200:
                logger.debug("Summarizer %s -> HTTP %s", model_id, response.status_code)
                return None
            body = response.json()
            if isinstance(body, list) and body:
                return (body[0].get("summary_text") or body[0].get("generated_text") or "").strip() or None
            if isinstance(body, dict):
                return (body.get("summary_text") or body.get("generated_text") or "").strip() or None
        except Exception as exc:
            logger.debug("Summarizer %s failed: %s", model_id, exc)
        return None

    # ------------------------------------------------------------ derived
    @staticmethod
    def _mood_trend(messages: List[ChatMessage]) -> str:
        sentiments = [m.sentiment for m in messages if m.sentiment]
        if not sentiments:
            return "mixed"
        total = len(sentiments)
        negative = sum(1 for s in sentiments if s in NEGATIVE_SENTIMENTS)
        positive = sum(1 for s in sentiments if s in POSITIVE_SENTIMENTS)
        if negative / total > 0.5:
            return "challenging"
        if positive / total > 0.5:
            return "strong"
        return "mixed"

    @staticmethod
    def _alignment_score(goals, user_texts: List[str], message_count: int) -> float:
        """How much of today's conversation touched the user's stated goals.

        With no goals set there's nothing to align to, so engagement alone
        carries the score — otherwise a user who never sets goals looks like
        they're failing at something they never claimed.
        """
        corpus = " ".join(user_texts).lower()
        engagement_bonus = 10.0 if message_count > 5 else 0.0

        if not goals:
            return round(min(100.0, 40.0 + engagement_bonus + min(30.0, message_count * 3)), 1)

        matched = 0
        for goal in goals:
            keywords = [
                word
                for word in re.findall(r"[a-z]{4,}", (goal.goal_name or "").lower())
                if word not in STOPWORDS
            ]
            if goal.category:
                keywords.append(goal.category.lower())
            if any(keyword in corpus for keyword in keywords):
                matched += 1

        score = (matched / max(1, len(goals))) * 100.0 + engagement_bonus
        return round(min(100.0, score), 1)

    @staticmethod
    def _key_topics(user_texts: List[str], limit: int = 5) -> List[str]:
        counter: Counter = Counter()
        for text in user_texts:
            for word in re.findall(r"[a-zA-Z]{4,}", text.lower()):
                if word in STOPWORDS or word.isdigit():
                    continue
                counter[word] += 1
        return [word for word, _count in counter.most_common(limit)]

    @staticmethod
    def _extract_by_sentiment(
        messages: List[ChatMessage], sentiments: set, limit: int = 5
    ) -> List[str]:
        out = []
        for message in messages:
            if message.sentiment in sentiments and message.user_message:
                snippet = message.user_message.strip().replace("\n", " ")[:80]
                out.append(snippet + ("..." if len(message.user_message) > 80 else ""))
            if len(out) >= limit:
                break
        return out


insight_service = InsightService()
