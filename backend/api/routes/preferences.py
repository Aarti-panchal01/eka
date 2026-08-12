"""User preference routes."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db
from models.schemas import PreferencesResponse, PreferencesUpdate
from services.memory_service import memory_service

router = APIRouter()


@router.get("", response_model=PreferencesResponse)
@router.get("/", response_model=PreferencesResponse, include_in_schema=False)
async def get_preferences(
    user_id: str = Query(...), db: AsyncSession = Depends(get_db)
):
    """Returns defaults on first call, creating the row as a side effect."""
    return await memory_service.get_preferences(db, user_id)


@router.put("", response_model=PreferencesResponse)
@router.put("/", response_model=PreferencesResponse, include_in_schema=False)
async def update_preferences(
    payload: PreferencesUpdate, db: AsyncSession = Depends(get_db)
):
    updates = payload.model_dump(exclude={"user_id"}, exclude_unset=True)
    return await memory_service.update_preferences(db, payload.user_id, updates)
