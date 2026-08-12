"""Daily and weekly insight routes."""

import logging
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db
from models.schemas import DailyInsightResponse, WeeklyInsightResponse
from services.insight_service import insight_service

logger = logging.getLogger(__name__)
router = APIRouter()


def _parse_date(value: str) -> date_cls:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"'{value}' is not YYYY-MM-DD")


@router.get("/daily/{date}", response_model=DailyInsightResponse)
async def daily_insight(
    date: str,
    user_id: str = Query(...),
    force: bool = Query(False, description="Recompute even if cached"),
    db: AsyncSession = Depends(get_db),
):
    """Return the day's insight, generating it if it doesn't exist yet."""
    day = _parse_date(date)
    insight = await insight_service.generate_daily_insight(user_id, day, db, force)

    if insight is None:
        # Not an error — there just wasn't enough conversation to say anything.
        return DailyInsightResponse(
            user_id=user_id,
            date=day.isoformat(),
            alignment_score=0.0,
            summary=None,
            mood_trend=None,
            message_count=0,
        )
    return insight


@router.get("/daily", response_model=DailyInsightResponse)
async def today_insight(
    user_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    day = datetime.now(timezone.utc).date()
    return await daily_insight(day.isoformat(), user_id, False, db)


@router.get("/weekly", response_model=WeeklyInsightResponse)
async def weekly_insight(
    user_id: str = Query(...),
    week_start: str = Query(None, description="YYYY-MM-DD; defaults to this Monday"),
    db: AsyncSession = Depends(get_db),
):
    if week_start:
        start = _parse_date(week_start)
    else:
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=today.weekday())  # Monday

    summary = await insight_service.generate_weekly_summary(user_id, start, db)
    return WeeklyInsightResponse(**summary)
