"""Daily reflection journal routes."""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db
from models.schemas import (
    DeleteResponse,
    ReflectionCreate,
    ReflectionResponse,
    ReflectionUpdate,
)
from services.llm_service import llm_service
from services.memory_service import memory_service
from services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("", response_model=List[ReflectionResponse])
@router.get("/", response_model=List[ReflectionResponse], include_in_schema=False)
async def list_reflections(
    user_id: str = Query(...),
    limit: int = Query(30, ge=1, le=200),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
):
    return await memory_service.get_reflections(db, user_id, limit, date_from, date_to)


@router.get("/by-date/{date}", response_model=Optional[ReflectionResponse])
async def get_by_date(
    date: str,
    user_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Fetch the reflection for one calendar day (YYYY-MM-DD)."""
    try:
        day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = await memory_service.get_reflections(
        db, user_id, limit=1, date_from=start, date_to=start + timedelta(days=1)
    )
    return rows[0] if rows else None


@router.post("", response_model=ReflectionResponse, status_code=201)
@router.post("/", response_model=ReflectionResponse, status_code=201, include_in_schema=False)
async def create_reflection(
    payload: ReflectionCreate, db: AsyncSession = Depends(get_db)
):
    data = payload.model_dump(exclude={"user_id", "request_commentary"})
    data["date"] = data.get("date") or datetime.now(timezone.utc)

    # Optional: let Eka respond to the journal entry in the chosen persona.
    if payload.request_commentary:
        data["eka_commentary"] = await _write_commentary(payload)

    return await memory_service.create_reflection(db, payload.user_id, data)


async def _write_commentary(payload: ReflectionCreate) -> Optional[str]:
    parts = []
    if payload.mood:
        parts.append(f"Mood: {payload.mood}")
    if payload.challenges_faced:
        parts.append(f"Challenges: {payload.challenges_faced}")
    if payload.learnings:
        parts.append(f"Learnings: {payload.learnings}")
    if payload.gratitude:
        parts.append(f"Grateful for: {payload.gratitude}")
    if not parts:
        return None

    mode = payload.mode_used or "reflection"
    try:
        return await llm_service.generate(
            prompt=(
                "This is today's journal entry. Respond to it in your voice — "
                "briefly, and without summarising it back.\n\n" + "\n".join(parts)
            ),
            mode=mode,
            system=rag_service.load_persona_prompt(mode),
            max_tokens=300,
        )
    except Exception as exc:
        logger.warning("Reflection commentary failed: %s", exc)
        return None


@router.get("/{reflection_id}", response_model=ReflectionResponse)
async def get_reflection(reflection_id: str, db: AsyncSession = Depends(get_db)):
    reflection = await memory_service.get_reflection(db, reflection_id)
    if not reflection:
        raise HTTPException(status_code=404, detail="Reflection not found")
    return reflection


@router.put("/{reflection_id}", response_model=ReflectionResponse)
async def update_reflection(
    reflection_id: str, payload: ReflectionUpdate, db: AsyncSession = Depends(get_db)
):
    reflection = await memory_service.update_reflection(
        db, reflection_id, payload.model_dump(exclude_unset=True)
    )
    if not reflection:
        raise HTTPException(status_code=404, detail="Reflection not found")
    return reflection


@router.delete("/{reflection_id}", response_model=DeleteResponse)
async def delete_reflection(reflection_id: str, db: AsyncSession = Depends(get_db)):
    if not await memory_service.delete_reflection(db, reflection_id):
        raise HTTPException(status_code=404, detail="Reflection not found")
    return DeleteResponse(deleted=True, id=reflection_id)
