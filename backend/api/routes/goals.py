"""Goal tracking routes."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db
from models.schemas import (
    DeleteResponse,
    GoalCreate,
    GoalProgressUpdate,
    GoalResponse,
    GoalUpdate,
)
from services.memory_service import memory_service

router = APIRouter()


@router.get("", response_model=List[GoalResponse])
@router.get("/", response_model=List[GoalResponse], include_in_schema=False)
async def list_goals(
    user_id: str = Query(...),
    status: Optional[str] = Query(None, pattern="^(active|completed|paused)$"),
    db: AsyncSession = Depends(get_db),
):
    return await memory_service.get_goals(db, user_id, status)


@router.post("", response_model=GoalResponse, status_code=201)
@router.post("/", response_model=GoalResponse, status_code=201, include_in_schema=False)
async def create_goal(payload: GoalCreate, db: AsyncSession = Depends(get_db)):
    data = payload.model_dump(exclude={"user_id"})
    return await memory_service.create_goal(db, payload.user_id, data)


@router.get("/{goal_id}", response_model=GoalResponse)
async def get_goal(goal_id: str, db: AsyncSession = Depends(get_db)):
    from models.db_models import GoalTracking

    goal = await db.get(GoalTracking, goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal


@router.put("/{goal_id}", response_model=GoalResponse)
async def update_goal(
    goal_id: str, payload: GoalUpdate, db: AsyncSession = Depends(get_db)
):
    goal = await memory_service.update_goal(
        db, goal_id, payload.model_dump(exclude_unset=True)
    )
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal


@router.put("/{goal_id}/progress", response_model=GoalResponse)
async def update_progress(
    goal_id: str, payload: GoalProgressUpdate, db: AsyncSession = Depends(get_db)
):
    """Move current_value only. Auto-completes when it reaches target_value."""
    goal = await memory_service.update_goal(
        db, goal_id, {"current_value": payload.current_value}
    )
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal


@router.delete("/{goal_id}", response_model=DeleteResponse)
async def delete_goal(goal_id: str, db: AsyncSession = Depends(get_db)):
    if not await memory_service.delete_goal(db, goal_id):
        raise HTTPException(status_code=404, detail="Goal not found")
    return DeleteResponse(deleted=True, id=goal_id)
