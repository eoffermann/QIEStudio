"""Programmatic Alembic migration runner (invoked on startup; DESIGN §8)."""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.config import get_settings

log = logging.getLogger(__name__)

# backend/ directory (this file is backend/app/migrate.py).
_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _alembic_config() -> Config:
    cfg = Config(str(_BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    return cfg


def run_migrations() -> None:
    """Upgrade the database to head. No-ops if already current (idempotent / resumable)."""
    log.info("Applying Alembic migrations to head")
    command.upgrade(_alembic_config(), "head")


def stamp_head() -> None:
    """Mark the DB as being at head without running migrations (for create_all bootstraps)."""
    command.stamp(_alembic_config(), "head")
