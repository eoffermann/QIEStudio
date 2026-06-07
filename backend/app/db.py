"""Database engine and session management (PostgreSQL via SQLModel/SQLAlchemy).

Sync engine by design: request handlers do light CRUD (run in FastAPI's threadpool) and the
single-accelerator inference work runs on a dedicated worker thread, so an async DB layer
would add complexity without benefit at v1's single-node scale.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings

log = logging.getLogger(__name__)

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine, creating it on first use."""
    global _engine
    if _engine is None:
        url = get_settings().database_url
        log.info("Creating database engine for %s", _redact(url))
        _engine = create_engine(url, pool_pre_ping=True, echo=False)
    return _engine


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a transactional session."""
    with Session(get_engine()) as session:
        yield session


def create_all() -> None:
    """Create all tables directly (used by tests; production uses Alembic migrations)."""
    # Import models so they register on SQLModel.metadata before create_all.
    import app.models  # noqa: F401

    SQLModel.metadata.create_all(get_engine())


def _redact(url: str) -> str:
    """Hide credentials when logging a DB URL."""
    if "@" in url and "//" in url:
        scheme, rest = url.split("//", 1)
        if "@" in rest:
            return f"{scheme}//***@{rest.split('@', 1)[1]}"
    return url
