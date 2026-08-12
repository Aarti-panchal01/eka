"""Async SQLAlchemy 2.0 engine/session wiring for Eka (asyncpg + Supabase)."""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for every Eka table."""


def _engine_kwargs() -> dict:
    kwargs: dict = {
        "echo": settings.DB_ECHO,
        "pool_pre_ping": True,
        "future": True,
    }
    # SQLite (used by tests) does not accept pool sizing args.
    if not settings.DATABASE_URL.startswith("sqlite"):
        kwargs.update(
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_recycle=settings.DB_POOL_RECYCLE,
        )
        # Supabase's pgbouncer (port 6543) cannot handle prepared statements,
        # which asyncpg uses by default.
        if "pooler" in settings.DATABASE_URL or ":6543" in settings.DATABASE_URL:
            kwargs["connect_args"] = {"statement_cache_size": 0}
    return kwargs


engine: AsyncEngine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs())

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: one session per request, rolled back on error."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """Create every table declared on Base. Safe to call on every startup.

    Alembic remains the source of truth for production schema changes; this is
    the zero-config path so a fresh Render/Docker deploy just works.
    """
    # Imported for the side effect of registering models on Base.metadata.
    from models import db_models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured (%d tables)", len(Base.metadata.tables))


async def drop_tables() -> None:
    """Test/reset helper. Never call this from application code."""
    from models import db_models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def check_connection() -> bool:
    """Used by GET /health."""
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # pragma: no cover - health path
        logger.warning("Database health check failed: %s", exc)
        return False


async def dispose_engine() -> None:
    await engine.dispose()
