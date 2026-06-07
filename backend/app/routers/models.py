"""Models + VRAM-aware advisor endpoints (DESIGN §2, §7, §9).

``GET /api/models`` lists the configured model ids per mode plus the precision options the
current device supports; ``POST /api/models/advise`` ranks precision options for a planned
run and recommends the best fit.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.schemas.advisor import (
    AdviceRead,
    AdviseRequest,
    DeviceInfoRead,
    ModelOptionRead,
    ModelsResponse,
)
from app.services import resolution
from app.services.device_manager import DeviceInfo, get_device_info
from app.services.model_advisor import advise

router = APIRouter(prefix="/api/models", tags=["models"])


def _available_precisions(device: DeviceInfo) -> list[str]:
    """The precision options offered on this device, in preference order (DESIGN §9.1)."""
    if device.backend == "mps":
        return ["bf16"]
    if device.backend in ("cuda", "rocm"):
        opts = ["bf16", "fp8"]
        if device.supports_int4_nunchaku:
            opts.append("int4")
        return opts
    return ["bf16"]


@router.get("", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """List configured model ids + per-mode available precisions (DESIGN §2, §7)."""
    settings = get_settings()
    device = get_device_info()
    precisions = _available_precisions(device)
    models = [
        ModelOptionRead(
            mode="edit",
            model_id=settings.default_edit_model,
            available_precisions=precisions,
        ),
        ModelOptionRead(
            mode="generate",
            model_id=settings.default_generate_model,
            available_precisions=precisions,
        ),
    ]
    return ModelsResponse(
        device=DeviceInfoRead.from_info(device),
        models=models,
        default_precision=settings.default_precision,
    )


@router.post("/advise", response_model=AdviceRead)
def advise_models(req: AdviseRequest) -> AdviceRead:
    """Rank precision options for a planned run (DESIGN §9.2).

    Computes the longer edge from an explicit ``longer_edge`` or by resolving the supplied
    ``resolution`` preset, then delegates to the model advisor.
    """
    longer_edge = _resolve_longer_edge(req)
    advice = advise(
        mode=req.mode,
        longer_edge=longer_edge,
        batch=req.batch,
        num_loras=len(req.loras),
        device=get_device_info(refresh=True),
    )
    return AdviceRead.from_advice(advice)


def _resolve_longer_edge(req: AdviseRequest) -> int:
    """Derive the run's longer edge from the request (resolution spec wins over explicit)."""
    if req.resolution is not None:
        w, h = resolution.resolve(
            base=req.resolution.base,
            orientation=req.resolution.orientation,
            aspect=req.resolution.aspect,
            source_dims=req.resolution.source_dims,
            max_long_edge=req.resolution.max_long_edge,
        )
        return max(w, h)
    if req.longer_edge is not None:
        return req.longer_edge
    # Neither supplied — fall back to the default base size.
    return resolution.DEFAULT_BASE_SIZE
