"""Chat + session routes."""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db
from models.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    MessageResponse,
    SessionCreate,
    SessionResponse,
    SessionUpdate,
)
from services.memory_service import memory_service
from services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/send", response_model=ChatResponse)
async def send_message(
    request: ChatRequest, db: AsyncSession = Depends(get_db)
) -> ChatResponse:
    """The main endpoint: runs the full RAG pipeline and returns Eka's reply."""
    try:
        return await rag_service.process_message(request, db)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("process_message failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    user_id: str = Query(...),
    include_archived: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    return await memory_service.get_sessions(db, user_id, include_archived, limit)


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(payload: SessionCreate, db: AsyncSession = Depends(get_db)):
    return await memory_service.create_session(
        db, payload.user_id, payload.mode, payload.title
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await memory_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str, payload: SessionUpdate, db: AsyncSession = Depends(get_db)
):
    session = await memory_service.update_session(
        db, session_id, payload.model_dump(exclude_unset=True)
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
async def list_messages(
    session_id: str,
    limit: int = Query(200, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    session = await memory_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return await memory_service.get_session_messages(db, session_id, limit)


@router.delete("/sessions/{session_id}", response_model=DeleteResponse)
async def archive_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """Soft delete — sets archived=True so history is never actually lost."""
    if not await memory_service.archive_session(db, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return DeleteResponse(deleted=True, id=session_id)
