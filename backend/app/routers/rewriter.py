"""Qwen-VL prompt-enhancer API router (DESIGN §5.8, §7).

Exposes the rewriter subsystem:

- ``GET  /api/rewriter/models``  — selectable VL models + quant/backend options (device-filtered).
- ``POST /api/rewriter/advise``  — co-reside-vs-swap advice for a chosen VL model.
- ``POST /api/rewriter/enhance`` — multimodal (Edit) / text (Generate) prompt rewrite.

All heavy work is delegated to :mod:`app.services.rewriter`; image inputs are resolved through
:mod:`app.services.asset_store` (binaries via the :class:`StorageProvider`, never absolute
paths). Heavy/model errors are wrapped so a missing model or runtime yields a clear ``503`` and
a bad request a ``400`` — never an unhandled crash (DESIGN §5.8).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.deps import CurrentPrincipal, DbSession, Storage
from app.schemas.rewriter import (
    AdviseRewriterRequest,
    EnhanceRequest,
    EnhanceResponse,
    RewriterAdviceRead,
    RewriterModelRead,
    RewriterModelsResponse,
)
from app.services import asset_store
from app.services.device_manager import get_device_info
from app.services.rewriter import advise_rewriter, get_rewriter_service, list_rewriter_models

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rewriter", tags=["rewriter"])


@router.get("/models", response_model=RewriterModelsResponse)
def get_models() -> RewriterModelsResponse:
    """List selectable VL rewriter models + quant/backend options for this device (§5.8)."""
    device = get_device_info()
    models = [RewriterModelRead(**m) for m in list_rewriter_models(device=device)]
    return RewriterModelsResponse(
        models=models,
        default_model=get_settings().default_rewriter_model,
    )


@router.post("/advise", response_model=RewriterAdviceRead)
def advise(req: AdviseRewriterRequest) -> RewriterAdviceRead:
    """Estimate whether ``vl_model`` co-resides with the loaded image model or swaps (§5.8)."""
    advice = advise_rewriter(
        vl_model=req.vl_model,
        image_model_resident_mb=req.image_model_resident_mb,
        device=get_device_info(refresh=True),
    )
    return RewriterAdviceRead.from_advice(advice)


@router.post("/enhance", response_model=EnhanceResponse)
def enhance(
    req: EnhanceRequest,
    session: DbSession,
    principal: CurrentPrincipal,  # noqa: ARG001 — resolved for ownership/seam parity (§4.1a)
    storage: Storage,
) -> EnhanceResponse:
    """Rewrite the prompt via the VL model (multimodal in Edit mode) (DESIGN §5.8, §7).

    Resolves ``image_ids`` to PIL images (Edit mode), runs the enhancer, and returns both the
    original and enhanced prompts so the caller can show an editable diff and persist both for
    reproducibility. Missing models/runtimes surface as ``503``; bad input as ``400``.
    """
    if req.mode not in ("generate", "edit"):
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode!r}")

    images = []
    if req.mode == "edit":
        for asset_id in req.image_ids:
            asset = asset_store.get_asset(asset_id=asset_id, session=session)
            if asset is None:
                raise HTTPException(status_code=404, detail=f"Unknown image id: {asset_id!r}")
            try:
                images.append(asset_store.open_pil(asset=asset, storage=storage))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400, detail=f"Could not open image {asset_id!r}: {exc}"
                ) from exc

    service = get_rewriter_service()
    try:
        enhanced = service.enhance(
            mode=req.mode,
            prompt=req.prompt,
            images=images or None,
            vl_model=req.vl_model,
            edit_model_id=req.edit_model_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ImportError, ModuleNotFoundError, NotImplementedError, OSError) as exc:
        # torch/transformers absent, runtime missing, or model weights unavailable: a clear
        # 503 (service not ready) rather than a crash (DESIGN §5.8).
        log.warning("Rewriter unavailable for %s: %s", req.vl_model, exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Prompt enhancer is unavailable: the VL model or its runtime could not be "
                f"loaded ({exc})."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001 — never leak an unhandled 500 from a model run
        log.exception("Rewriter run failed for %s", req.vl_model)
        raise HTTPException(status_code=503, detail=f"Prompt enhancer failed: {exc}") from exc

    return EnhanceResponse(enhanced_prompt=enhanced, original_prompt=req.prompt)
