"""FastAPI application factory + lifespan.

Startup logs *before* each slow phase with elapsed-time context (project convention /
DESIGN §8): DB connect, Alembic migrate, queue start, optional warm-up.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import get_settings
from app.interfaces.queue import get_job_queue
from app.logging_utils import configure_logging, phase
from app.routers import ROUTER_MODULES

log = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("Starting %s v%s", settings.app_name, __version__)

    # Ensure local storage roots exist before anything writes to them.
    if settings.is_local_storage:
        for sub in ("assets", "outputs", "loras", "thumbs", "tmp"):
            (settings.data_root / sub).mkdir(parents=True, exist_ok=True)

    if settings.run_migrations_on_startup:
        try:
            with phase(log, "Running database migrations (Alembic)"):
                from app.migrate import run_migrations

                run_migrations()
        except Exception:  # noqa: BLE001
            log.exception("Migrations failed at startup (continuing; check DB connectivity)")

    with phase(log, "Starting in-process job queue worker"):
        get_job_queue()

    if settings.warmup_on_startup:
        log.info("Warm-up enabled — first pipeline load will happen now (may be slow)")
        # Pipeline warm-up is delegated to the pipeline service if present.
        try:
            from app.services.pipeline_service import warmup  # type: ignore

            with phase(log, "Warming up default pipeline"):
                warmup()
        except Exception:  # noqa: BLE001
            log.warning("Warm-up skipped (pipeline service unavailable or failed)")

    log.info("%s ready", settings.app_name)
    yield

    log.info("Shutting down")
    get_job_queue().shutdown(wait=True)


def _include_routers(app: FastAPI) -> None:
    """Include every router module that exists (skip the not-yet-built ones)."""
    for mod_name in ROUTER_MODULES:
        if importlib.util.find_spec(mod_name) is None:
            log.info("Router %s not present yet — skipping", mod_name)
            continue
        module = importlib.import_module(mod_name)
        router = getattr(module, "router", None)
        if router is None:
            log.warning("Router module %s has no `router` attribute", mod_name)
            continue
        app.include_router(router)
        log.info("Included router %s", mod_name)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        summary="Qwen-Image generation + editing studio",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _include_routers(app)

    # Serve the built SPA as static assets if present (DESIGN §4.2). Mounted last so it
    # doesn't shadow /api routes.
    dist = settings.frontend_dist
    if dist.exists() and dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="spa")
        log.info("Serving SPA static assets from %s", dist)
    else:
        log.info("No SPA build at %s — API-only mode", dist)

    return app


app = create_app()
