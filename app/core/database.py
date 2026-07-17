"""
CodeMentor AI - Async Database Engine & Session Management
===========================================================
Sets up SQLAlchemy with an async engine backed by asyncpg (PostgreSQL).

Design Rationale:
- Async engine: never blocks the event loop — critical for FastAPI concurrency.
- AsyncSession factory: each request gets its own session (Unit of Work pattern).
- get_db() dependency: FastAPI Depends() injects a session and guarantees cleanup.
- We use NullPool for the async engine to avoid connection issues with asyncpg
  in certain serverless/Gunicorn configurations.

Terminology:
- Engine:  Low-level DB connection pool.
- Session: Unit of Work — a single transaction boundary per request.
- Base:    DeclarativeBase for all ORM models to inherit from.
"""

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ==============================================================
# SQLAlchemy Async Engine
# ==============================================================

def _create_engine() -> AsyncEngine:
    """
    Create and return the async SQLAlchemy engine.

    Pool settings tuned for typical web API workloads:
    - pool_size: number of persistent connections
    - max_overflow: extra connections beyond pool_size (for traffic spikes)
    - pool_pre_ping: validates connections before checkout (avoids stale conn errors)
    """
    engine_kwargs: dict[str, Any] = {
        "echo": settings.debug,             # SQL query logging — off in production
        "pool_pre_ping": True,              # Prevents "server closed connection" errors
        "pool_size": 10,                    # Permanent connection pool
        "max_overflow": 20,                 # Extra connections under burst load
        "pool_recycle": 3600,               # Recycle connections every 1 hour
    }

    logger.info(
        "Creating async database engine: host=%s db=%s",
        settings.postgres_host,
        settings.postgres_db,
    )

    return create_async_engine(settings.database_url, **engine_kwargs)


# Module-level engine — created once, shared across the application
engine: AsyncEngine = _create_engine()


# ==============================================================
# Session Factory
# ==============================================================

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Objects remain accessible after commit (avoids lazy-load errors)
    autocommit=False,
    autoflush=False,
)


# ==============================================================
# Declarative Base for ORM Models
# ==============================================================

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    All models must inherit from this class:
        class User(Base):
            __tablename__ = "users"
            ...
    """
    pass


# ==============================================================
# FastAPI Dependency: Database Session
# ==============================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an async database session.

    Guarantees that the session is always closed after the request,
    even if an exception is raised mid-request.

    Usage in FastAPI:
        from fastapi import Depends
        from app.core.database import get_db

        @router.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            ...

    The `async with` block handles:
    - Session creation
    - Automatic rollback on exception
    - Session closure in all cases
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ==============================================================
# Database Health Check
# ==============================================================

async def check_db_connection() -> bool:
    """
    Verifies the database is reachable by executing a simple query.
    Used in the health check endpoint and startup validation.

    Returns:
        True if connected, False otherwise.
    """
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection verified successfully.")
        return True
    except Exception as exc:
        logger.error("Database connection failed: %s", exc)
        return False


async def close_db_connection() -> None:
    """
    Gracefully dispose the connection pool.
    Called during application shutdown to release all connections cleanly.
    """
    logger.info("Closing database connection pool...")
    await engine.dispose()
    logger.info("Database connection pool closed.")
