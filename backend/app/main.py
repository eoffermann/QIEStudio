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
from typing import Any

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

    # Load any stored HuggingFace token into the environment so every weight download
    # (pipelines, rewriter, Nunchaku int4 sources, LoRA imports) authenticates instead of
    # running anonymously. Safe to skip if the table/DB isn't ready.
    try:
        from sqlmodel import Session

        from app.db import get_engine
        from app.interfaces.auth import Principal
        from app.services import integrations

        with Session(get_engine()) as session:
            applied = integrations.apply_hf_token_to_env(
                session=session, principal=Principal()
            )
        log.info("HuggingFace token %s at startup", "applied" if applied else "not configured")
    except Exception:  # noqa: BLE001 — never let token loading abort startup
        log.exception("Could not load HuggingFace token at startup (continuing)")

    with phase(log, "Starting in-process job queue worker"):
        get_job_queue()

    # Wire the progress hub to this event loop so the (sync) worker thread can push
    # WebSocket updates, then re-enqueue any interrupted jobs (resumability, RUN §7).
    try:
        import asyncio

        from app.services.progress_hub import get_progress_hub

        get_progress_hub().set_loop(asyncio.get_running_loop())
        from app.services.job_service import recover_jobs

        with phase(log, "Recovering queued/running jobs"):
            recover_jobs()
    except Exception:  # noqa: BLE001 — never let recovery abort startup
        log.exception("Job recovery skipped (continuing)")

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

    # Expose the spec's top-level WebSocket path (DESIGN §7: WS /ws/jobs/{id}) in addition
    # to the prefixed router route, if the jobs router is present.
    if importlib.util.find_spec("app.routers.jobs") is not None:
        from app.routers.jobs import job_ws

        app.add_api_websocket_route("/ws/jobs/{job_id}", job_ws)
        log.info("Mounted WebSocket /ws/jobs/{job_id}")

    # Serve the built SPA (DESIGN §4.2) with **client-side-routing fallback**: unknown paths
    # (e.g. /compose on reload/deep-link) return index.html so React Router can handle them,
    # instead of a 404. /api and /ws still get real JSON 404s. Mounted last so it never
    # shadows the API routes registered above.
    dist = settings.frontend_dist
    if dist.exists() and dist.is_dir():
        app.mount("/", _SPAStaticFiles(directory=str(dist), html=True), name="spa")
        log.info("Serving SPA (with client-routing fallback) from %s", dist)
    else:
        log.info("No SPA build at %s — API-only mode", dist)

    return app


class _SPAStaticFiles(StaticFiles):
    """StaticFiles that falls back to ``index.html`` for unknown non-API paths (SPA routing)."""

    async def get_response(self, path: str, scope: Any) -> Any:
        from starlette.exceptions import HTTPException as StarletteHTTPException

        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            # Don't serve HTML for API/WS misses — let those be real 404s.
            if exc.status_code == 404 and not path.startswith(("api", "ws")):
                return await super().get_response("index.html", scope)
            raise


app = create_app()
