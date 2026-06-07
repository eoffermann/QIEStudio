"""Resolution preset endpoints (DESIGN §5.4, §7).

``GET /api/presets/resolution`` returns the preset enumerations; the ``/resolve`` POST turns
a ``{base, orientation, aspect, source_dims?}`` selection into a concrete ``{w, h}``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.advisor import PresetsResponse, ResolveRequest, ResolveResponse
from app.services import resolution

router = APIRouter(prefix="/api/presets/resolution", tags=["presets"])


@router.get("", response_model=PresetsResponse)
def get_presets() -> PresetsResponse:
    """Return the resolution preset enumerations (DESIGN §5.4)."""
    return PresetsResponse(**resolution.enumerate_presets())


@router.post("/resolve", response_model=ResolveResponse)
def resolve_preset(req: ResolveRequest) -> ResolveResponse:
    """Resolve a preset selection into a concrete ``W×H`` (DESIGN §5.4)."""
    try:
        w, h = resolution.resolve(
            base=req.base,
            orientation=req.orientation,
            aspect=req.aspect,
            source_dims=req.source_dims,
            max_long_edge=req.max_long_edge,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ResolveResponse(w=w, h=h)
