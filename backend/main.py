"""Eka backend — FastAPI entrypoint.

    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Startup is deliberately non-fatal: if Postgres or Qdrant is unreachable the app
still boots and serves /health telling you what's broken. A backend that refuses
to start is much harder to debug on Render than one that starts and reports.
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Render/Docker run uvicorn from inside backend/, so `import config` works. This
# also makes `python backend/main.py` work from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.routes import (  # noqa: E402
    chat,
    goals,
    insights,
    memory,
    preferences,
    reflections,
    voice,
)
from config import settings  # noqa: E402
from database import check_connection, create_tables, dispose_engine  # noqa: E402
from models.schemas import HealthResponse  # noqa: E402
from services.asr_service import asr_service  # noqa: E402
from services.complexity_service import complexity_service  # noqa: E402
from services.embedding_service import embedding_service  # noqa: E402
from services.llm_service import llm_service  # noqa: E402
from services.ranker_service import ranker_service  # noqa: E402
from services.safety_service import safety_service  # noqa: E402
from services.sentiment_service import sentiment_service  # noqa: E402
from services.tts_service import tts_service  # noqa: E402

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eka")
# httpx logs every outbound request at INFO; too noisy for a chat backend.
logging.getLogger("httpx").setLevel(logging.WARNING)


KEEP_ALIVE_INTERVAL = int(os.environ.get("KEEP_ALIVE_INTERVAL", 600))  # 10 minutes


async def keep_alive() -> None:
    """Self-ping so Render's free tier doesn't spin the instance down.

    IMPORTANT CAVEAT: pinging 127.0.0.1 does NOT prevent Render from sleeping.
    Render decides based on *inbound external* traffic through its router; a
    loopback request never reaches it. So this task prefers the public URL,
    which Render injects as RENDER_EXTERNAL_URL — that request goes out to the
    internet and back in through the router, which does count.

    Treat this as a backup for the UptimeRobot monitor in
    infra/uptimerobot_note.txt, not a replacement: if the instance has already
    been suspended, no task inside it is running to wake it up. Only an
    external pinger can do that. See that file for the instance-hours tradeoff.
    """
    target = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if target:
        url = f"{target}/health"
        logger.info("keep_alive: pinging %s every %ds", url, KEEP_ALIVE_INTERVAL)
    else:
        port = os.environ.get("PORT", "8000")
        url = f"http://127.0.0.1:{port}/health"
        logger.info(
            "keep_alive: RENDER_EXTERNAL_URL unset, falling back to %s "
            "(loopback — will NOT prevent a Render spin-down)", url
        )

    while True:
        await asyncio.sleep(KEEP_ALIVE_INTERVAL)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
            logger.debug("keep_alive ping -> %s", response.status_code)
        except Exception as exc:
            # Never let this task die; it must outlive transient failures.
            logger.debug("keep_alive ping failed (ignored): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 62)
    logger.info(" Eka %s starting (%s)", settings.APP_VERSION, settings.ENVIRONMENT)
    logger.info(" LLM_MODE=%s  model=%s", settings.LLM_MODE, settings.GROQ_MODEL)
    logger.info("=" * 62)

    try:
        await create_tables()
    except Exception as exc:
        logger.error("Table creation failed: %s", exc)
        logger.error("  Check DATABASE_URL. /health will report database: false.")

    try:
        await embedding_service.ensure_collection()
    except Exception as exc:
        logger.error("Qdrant setup failed: %s", exc)

    if not settings.HF_USERNAME:
        logger.warning(
            "HF_USERNAME is empty — the fine-tuned models cannot be located. "
            "Embeddings and classifiers will use base models or heuristics."
        )
    if not settings.GROQ_API_KEY and settings.LLM_MODE == "groq":
        logger.error("LLM_MODE=groq but GROQ_API_KEY is empty — chat will not work.")

    # Disabled by default locally — a self-ping every 10 min is only useful on
    # a host that sleeps. Set KEEP_ALIVE=1 (render.yaml does) to enable.
    keep_alive_task = None
    if os.environ.get("KEEP_ALIVE", "0") == "1" or settings.ENVIRONMENT == "render":
        keep_alive_task = asyncio.create_task(keep_alive())

    logger.info("Eka ready -> http://localhost:8000/docs")
    yield

    logger.info("Eka shutting down")
    if keep_alive_task:
        keep_alive_task.cancel()
    await embedding_service.close()
    await dispose_engine()


app = FastAPI(
    title="Eka API",
    description=(
        "A lifelong AI companion: four fine-tuned personas over a semantic "
        "memory, with retrieval depth that scales to the question."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Eka-Voice"],
)

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(memory.router, prefix="/memory", tags=["memory"])
app.include_router(voice.router, prefix="/voice", tags=["voice"])
app.include_router(goals.router, prefix="/goals", tags=["goals"])
app.include_router(reflections.router, prefix="/reflections", tags=["reflections"])
app.include_router(insights.router, prefix="/insights", tags=["insights"])
app.include_router(preferences.router, prefix="/preferences", tags=["preferences"])


@app.get("/", tags=["system"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "modes": ["founder", "chanakya", "gita", "reflection"],
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Full service status. UptimeRobot pings this every 5 minutes."""
    llm = await llm_service.health_check()
    vectors = await embedding_service.health()
    database_ok = await check_connection()

    complexity_ok = await complexity_service.available()
    ranker_ok = await ranker_service.available()
    sentiment_ok = await sentiment_service.available()

    llm_ok = bool(llm.get("ollama") or llm.get("groq") or llm.get("hf_space"))
    # A backend that can't generate is broken. Everything else has a working
    # fallback, so it degrades rather than errors.
    if not llm_ok or not database_ok:
        status = "error"
    elif not vectors["qdrant"] or vectors["degraded"]:
        status = "degraded"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        llm_mode=settings.LLM_MODE,
        ollama=bool(llm.get("ollama")),
        groq=bool(llm.get("groq")),
        hf_space=bool(llm.get("hf_space")),
        qdrant=vectors["qdrant"],
        database=database_ok,
        complexity=complexity_ok,
        ranker=ranker_ok,
        sentiment=sentiment_ok,
        sarvam_configured=bool(settings.SARVAM_API_KEY),
        details={
            "llm": llm,
            "vectors": vectors,
            "complexity": await complexity_service.health(),
            "ranker": await ranker_service.health(),
            "sentiment": await sentiment_service.health(),
            "tts": await tts_service.health(),
            "asr": await asr_service.health(),
            "safety": await safety_service.health(),
            "hf_username_set": bool(settings.HF_USERNAME),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc) if settings.DEBUG else "Internal server error",
            "path": request.url.path,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=settings.DEBUG,
    )
