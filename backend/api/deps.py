"""Shared FastAPI dependencies."""

from typing import AsyncGenerator, Optional

from fastapi import HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db as _get_db


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Re-exported so routes import from one place."""
    async for session in _get_db():
        yield session


async def get_current_user(
    user_id: Optional[str] = Query(
        None, description="Caller-supplied user id (placeholder until auth exists)"
    ),
) -> str:
    """Identity placeholder.

    Eka has no auth yet: the frontend generates a uuid on first load and sends
    it with every call. Every query in memory_service is already scoped by
    user_id, so adding real auth later means replacing this function body and
    nothing else.

    # BLOCKER: before this is exposed publicly, swap this for a verified
    # identity (Supabase Auth JWT). Right now anyone who knows a user_id can
    # read that user's memories.
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    return user_id


def pagination(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    return {"skip": skip, "limit": limit}
