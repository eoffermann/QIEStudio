"""API routers.

Each subsystem owns one module here exposing a module-level ``router: APIRouter`` with its
own prefix + tags. ``app.main`` discovers and includes them defensively, so the app boots
even while some routers are still being built. Router→prefix contract:

    health        -> (no prefix)        /healthz, /readyz
    device        -> /api/device
    models        -> /api/models, /api/models/advise
    resolution    -> /api/presets/resolution
    assets        -> /api/assets
    prompts       -> /api/prompts
    loras         -> /api/loras
    integrations  -> /api/integrations
    jobs          -> /api/jobs  (+ WS /ws/jobs/{id})
    rewriter      -> /api/rewriter
"""

# Modules that app.main will try to include, in order. Add to this list when a new router
# subsystem lands; main.py skips any that don't exist yet (find_spec check).
ROUTER_MODULES: list[str] = [
    "app.routers.health",
    "app.routers.device",
    "app.routers.models",
    "app.routers.resolution",
    "app.routers.assets",
    "app.routers.prompts",
    "app.routers.loras",
    "app.routers.integrations",
    "app.routers.jobs",
    "app.routers.rewriter",
]
