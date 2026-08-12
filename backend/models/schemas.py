"""Pydantic v2 request/response models for every Eka endpoint."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Mode = Literal["founder", "chanakya", "gita", "reflection"]
Priority = Literal["high", "normal", "low", "excluded"]
Complexity = Literal["simple", "normal", "complex", "deep"]
GoalStatus = Literal["active", "completed", "paused"]
MemorySource = Literal["chat", "upload", "manual", "reflection"]

ORM = ConfigDict(from_attributes=True)


# ================================================================== chat
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    user_id: str
    session_id: Optional[str] = None
    mode: Mode = "founder"
    # Client hint only; the complexity classifier still runs and wins.
    stream: bool = False

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be blank")
        return v


class RetrievedMemory(BaseModel):
    id: str
    title: Optional[str] = None
    content: Optional[str] = None
    score: float = 0.0
    rank_score: Optional[float] = None
    topic: Optional[str] = None
    importance: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: Optional[str] = None
    mode: Mode
    complexity: Complexity = "normal"
    sentiment: str = "neutral"
    tags: List[str] = Field(default_factory=list)
    retrieved_memories: List[RetrievedMemory] = Field(default_factory=list)
    memory_created: bool = False
    latency_ms: int = 0
    llm_backend: str = "unknown"
    degraded: List[str] = Field(
        default_factory=list,
        description="Services that fell back during this request.",
    )


class SessionCreate(BaseModel):
    user_id: str
    mode: Mode = "founder"
    title: Optional[str] = None


class SessionUpdate(BaseModel):
    title: Optional[str] = None
    mode: Optional[Mode] = None
    archived: Optional[bool] = None


class SessionResponse(BaseModel):
    model_config = ORM

    id: str
    user_id: str
    title: str
    mode: str
    message_count: int
    last_message_preview: Optional[str] = None
    is_active: bool
    archived: bool
    created_date: datetime
    updated_date: datetime


class MessageResponse(BaseModel):
    model_config = ORM

    id: str
    session_id: str
    user_message: str
    eka_response: str
    mode: str
    tags: Optional[List[str]] = None
    sentiment: Optional[str] = None
    complexity: Optional[str] = None
    created_date: datetime


# ================================================================ memory
class MemoryCreate(BaseModel):
    user_id: str
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    source: MemorySource = "manual"
    topic: Optional[str] = Field(None, max_length=128)
    tags: List[str] = Field(default_factory=list)
    importance: int = Field(5, ge=1, le=10)
    user_priority: Priority = "normal"


class MemoryUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    content: Optional[str] = None
    topic: Optional[str] = None
    tags: Optional[List[str]] = None
    importance: Optional[int] = Field(None, ge=1, le=10)
    user_priority: Optional[Priority] = None


class MemoryPriorityUpdate(BaseModel):
    priority: Priority


class MemoryResponse(BaseModel):
    model_config = ORM

    id: str
    user_id: str
    title: str
    content: str
    source: str
    topic: Optional[str] = None
    tags: Optional[List[str]] = None
    importance: int
    user_priority: str
    access_count: int
    embedded: bool = False
    created_date: datetime
    updated_date: datetime


class MemorySearchRequest(BaseModel):
    text: str = Field(..., min_length=1)
    user_id: str
    limit: int = Field(20, ge=1, le=100)


class MemorySearchResult(BaseModel):
    id: str
    title: Optional[str] = None
    content: Optional[str] = None
    score: float
    topic: Optional[str] = None
    importance: Optional[int] = None
    user_priority: Optional[str] = None
    source: Optional[str] = None


class MemoryListResponse(BaseModel):
    items: List[MemoryResponse]
    total: int
    skip: int
    limit: int


# ================================================================= voice
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    mode: Mode = "founder"


class STTResponse(BaseModel):
    text: str
    backend: Optional[str] = None


# ================================================================= goals
class GoalCreate(BaseModel):
    user_id: str
    goal_name: str = Field(..., min_length=1, max_length=255)
    category: str = "general"
    description: Optional[str] = None
    target_value: float = Field(1.0, gt=0)
    current_value: float = 0.0
    unit: str = "completion"
    target_date: Optional[datetime] = None


class GoalUpdate(BaseModel):
    goal_name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    target_value: Optional[float] = Field(None, gt=0)
    current_value: Optional[float] = None
    unit: Optional[str] = None
    target_date: Optional[datetime] = None
    streak_days: Optional[int] = Field(None, ge=0)
    status: Optional[GoalStatus] = None


class GoalProgressUpdate(BaseModel):
    current_value: float = Field(..., ge=0)


class GoalResponse(BaseModel):
    model_config = ORM

    id: str
    user_id: str
    category: str
    goal_name: str
    description: Optional[str] = None
    target_value: float
    current_value: float
    unit: str
    target_date: Optional[datetime] = None
    streak_days: int
    status: str
    created_date: datetime


# =========================================================== reflections
class ReflectionCreate(BaseModel):
    user_id: str
    date: Optional[datetime] = None
    mood: Optional[str] = None
    challenges_faced: Optional[str] = None
    learnings: Optional[str] = None
    gratitude: Optional[str] = None
    mode_used: Optional[Mode] = None
    # When true, Eka writes eka_commentary on the reflection via the LLM.
    request_commentary: bool = False


class ReflectionUpdate(BaseModel):
    mood: Optional[str] = None
    challenges_faced: Optional[str] = None
    learnings: Optional[str] = None
    gratitude: Optional[str] = None
    eka_commentary: Optional[str] = None


class ReflectionResponse(BaseModel):
    model_config = ORM

    id: str
    user_id: str
    date: datetime
    mood: Optional[str] = None
    challenges_faced: Optional[str] = None
    learnings: Optional[str] = None
    gratitude: Optional[str] = None
    eka_commentary: Optional[str] = None
    mode_used: Optional[str] = None
    created_date: datetime


# ============================================================== insights
class DailyInsightResponse(BaseModel):
    model_config = ORM

    id: Optional[str] = None
    user_id: str
    date: str
    alignment_score: float = 0.0
    summary: Optional[str] = None
    key_topics: List[str] = Field(default_factory=list)
    mood_trend: Optional[str] = None
    achievements: List[str] = Field(default_factory=list)
    challenges: List[str] = Field(default_factory=list)
    message_count: int = 0


class WeeklyInsightResponse(BaseModel):
    user_id: str
    week_start: str
    week_end: str
    days_with_data: int = 0
    avg_alignment_score: float = 0.0
    mood_trend: Optional[str] = None
    key_topics: List[str] = Field(default_factory=list)
    total_messages: int = 0
    daily: List[DailyInsightResponse] = Field(default_factory=list)


# =========================================================== preferences
class PreferencesUpdate(BaseModel):
    user_id: str
    default_mode: Optional[Mode] = None
    voice_enabled: Optional[bool] = None
    playback_speed: Optional[float] = Field(None, ge=0.5, le=2.0)
    theme_accent: Optional[str] = None
    always_listening: Optional[bool] = None


class PreferencesResponse(BaseModel):
    model_config = ORM

    id: Optional[str] = None
    user_id: str
    default_mode: str = "founder"
    voice_enabled: bool = False
    playback_speed: float = 1.0
    theme_accent: str = "amber"
    always_listening: bool = False


# ================================================================ system
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str
    environment: str
    llm_mode: str
    ollama: bool = False
    groq: bool = False
    hf_space: bool = False
    qdrant: bool = False
    database: bool = False
    complexity: bool = False
    ranker: bool = False
    sentiment: bool = False
    sarvam_configured: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class DeleteResponse(BaseModel):
    deleted: bool
    id: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
