"""Memory CRUD, semantic search, and document upload."""

import io
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db
from models.schemas import (
    DeleteResponse,
    MemoryCreate,
    MemoryListResponse,
    MemoryPriorityUpdate,
    MemoryResponse,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryUpdate,
)
from services.memory_service import memory_service

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5MB
ALLOWED_SUFFIXES = (".txt", ".md", ".markdown", ".pdf")


@router.get("", response_model=MemoryListResponse)
@router.get("/", response_model=MemoryListResponse, include_in_schema=False)
async def list_memories(
    user_id: str = Query(...),
    topic: Optional[str] = None,
    priority: Optional[str] = Query(None, alias="priority"),
    source: Optional[str] = None,
    q: Optional[str] = Query(None, description="Substring match on title/content"),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    filters = {
        "topic": topic,
        "user_priority": priority,
        "source": source,
        "q": q,
        "date_from": date_from,
        "date_to": date_to,
    }
    items = await memory_service.get_memories(db, user_id, filters, skip, limit)
    total = await memory_service.count_memories(db, user_id, filters)
    return MemoryListResponse(items=items, total=total, skip=skip, limit=limit)


@router.post("", response_model=MemoryResponse, status_code=201)
@router.post("/", response_model=MemoryResponse, status_code=201, include_in_schema=False)
async def create_memory(payload: MemoryCreate, db: AsyncSession = Depends(get_db)):
    return await memory_service.create_memory(
        db=db,
        user_id=payload.user_id,
        title=payload.title,
        content=payload.content,
        source=payload.source,
        topic=payload.topic,
        tags=payload.tags,
        importance=payload.importance,
        user_priority=payload.user_priority,
    )


@router.post("/search", response_model=List[MemorySearchResult])
async def search_memories(payload: MemorySearchRequest):
    return await memory_service.search_memories(
        payload.user_id, payload.text, payload.limit
    )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    user_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    memory = await memory_service.get_memory(db, memory_id, user_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    payload: MemoryUpdate,
    user_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    memory = await memory_service.update_memory(
        db, memory_id, payload.model_dump(exclude_unset=True), user_id
    )
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@router.put("/{memory_id}/priority", response_model=MemoryResponse)
async def update_priority(
    memory_id: str,
    payload: MemoryPriorityUpdate,
    user_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Quick priority toggle. 'excluded' removes the memory from retrieval."""
    memory = await memory_service.update_memory_priority(
        db, user_id, memory_id, payload.priority
    )
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@router.delete("/{memory_id}", response_model=DeleteResponse)
async def delete_memory(
    memory_id: str,
    user_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    if not await memory_service.delete_memory(db, user_id, memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return DeleteResponse(deleted=True, id=memory_id)


@router.post("/upload", response_model=MemoryResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    title: Optional[str] = Form(None),
    topic: Optional[str] = Form(None),
    importance: int = Form(6),
    db: AsyncSession = Depends(get_db),
):
    """Turn a .txt/.md/.pdf into a memory."""
    filename = file.filename or "upload"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_SUFFIXES)}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File is empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw) / 1e6:.1f}MB, max 5MB)",
        )

    text = _extract_pdf(raw) if suffix == ".pdf" else _extract_text(raw)
    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted (a scanned PDF needs OCR, which Eka doesn't do)",
        )

    return await memory_service.create_memory(
        db=db,
        user_id=user_id,
        title=(title or filename)[:255],
        content=text,
        source="upload",
        topic=topic,
        tags=[],
        importance=max(1, min(10, importance)),
        user_priority="normal",
    )


def _extract_text(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            # PyPDF2 is the deprecated predecessor with an identical PdfReader
            # API. Accepted only so an environment that has not reinstalled
            # from requirements.txt keeps working; it carries an unfixable
            # advisory, so requirements.txt asks for pypdf.
            from PyPDF2 import PdfReader
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="PDF support needs pypdf — pip install pypdf==6.14.2",
            )
    try:
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {exc}")
