"""LoRA API (DESIGN §5.3, §7) — registry + multi-source import.

    GET    /api/loras
    POST   /api/loras/upload             # install via .safetensors file
    POST   /api/loras/import             # {source: hf|civitai|url, ref}
    GET    /api/loras/search?source=&q=  # search HF Hub / CivitAI (degrades to [])
    PATCH  /api/loras/{id}
    DELETE /api/loras/{id}

Imports run synchronously with progress logging (see :mod:`app.services.lora_manager`).
Search isolates network access and returns ``[]`` gracefully when keys are missing or a
provider call fails.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.deps import CurrentPrincipal, DbSession, Storage
from app.schemas.loras import LoraImport, LoraRead, LoraSearchResult, LoraUpdate
from app.services import civitai, integrations, lora_manager

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/loras", tags=["loras"])


@router.get("", response_model=list[LoraRead])
def list_loras(session: DbSession, principal: CurrentPrincipal) -> list[LoraRead]:
    """List registered LoRAs (newest first)."""
    return [LoraRead.from_lora(lora) for lora in lora_manager.list_loras(
        session=session, principal=principal
    )]


@router.post("/upload", response_model=LoraRead, status_code=status.HTTP_201_CREATED)
async def upload_lora(
    session: DbSession,
    storage: Storage,
    principal: CurrentPrincipal,
    file: UploadFile = File(...),
    name: str | None = Form(None),
) -> LoraRead:
    """Install a LoRA from an uploaded ``.safetensors`` file."""
    data = await file.read()
    try:
        lora = lora_manager.import_from_upload(
            data=data,
            filename=file.filename or "lora.safetensors",
            name=name,
            session=session,
            storage=storage,
            principal=principal,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return LoraRead.from_lora(lora)


@router.post("/import", response_model=LoraRead, status_code=status.HTTP_201_CREATED)
def import_lora(
    body: LoraImport,
    session: DbSession,
    storage: Storage,
    principal: CurrentPrincipal,
) -> LoraRead:
    """Install a LoRA from Hugging Face, CivitAI, or a direct URL."""
    src = body.source.lower().strip()
    try:
        if src == "hf":
            lora = lora_manager.import_from_hf(
                repo_id=body.ref,
                filename=body.filename,
                name=body.name,
                session=session,
                storage=storage,
                principal=principal,
            )
        elif src == "civitai":
            lora = lora_manager.import_from_civitai(
                url_or_id=body.ref,
                name=body.name,
                session=session,
                storage=storage,
                principal=principal,
            )
        elif src == "url":
            lora = lora_manager.import_from_url(
                url=body.ref,
                name=body.name,
                session=session,
                storage=storage,
                principal=principal,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown import source {body.source!r}; expected hf|civitai|url",
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return LoraRead.from_lora(lora)


@router.get("/search", response_model=list[LoraSearchResult])
def search_loras(
    session: DbSession,
    principal: CurrentPrincipal,
    source: str,
    q: str = "",
) -> list[LoraSearchResult]:
    """Search HF Hub or CivitAI for importable LoRAs. Degrades to ``[]`` on any failure."""
    src = source.lower().strip()
    if not q.strip():
        return []
    try:
        if src == "hf":
            hits = lora_manager.search_hf(query=q)
            return [
                LoraSearchResult(source="hf", name=h["name"], ref=h["repo_id"]) for h in hits
            ]
        if src == "civitai":
            api_key = integrations.get_key_plaintext(
                provider="civitai", session=session, principal=principal
            )
            hits = civitai.search_models(query=q, api_key=api_key)
            results: list[LoraSearchResult] = []
            for h in hits:
                model_id = h.get("model_id")
                version_id = h.get("version_id")
                ref = f"{model_id}@{version_id}" if version_id else str(model_id)
                results.append(
                    LoraSearchResult(
                        source="civitai",
                        name=h.get("name") or str(model_id),
                        ref=ref,
                        base_model=h.get("base_model", ""),
                        trigger_words=h.get("trigger_words") or [],
                    )
                )
            return results
    except Exception as exc:  # noqa: BLE001 — search must degrade gracefully (DESIGN §7)
        log.warning("LoRA search (%s) failed: %s", src, exc)
        return []
    return []


@router.patch("/{lora_id}", response_model=LoraRead)
def update_lora(
    lora_id: str,
    body: LoraUpdate,
    session: DbSession,
    principal: CurrentPrincipal,
) -> LoraRead:
    """Edit LoRA metadata (rename, trigger words, weight, base/mode tags, enabled)."""
    lora = lora_manager.update_lora(
        lora_id=lora_id,
        session=session,
        principal=principal,
        name=body.name,
        description=body.description,
        trigger_words=body.trigger_words,
        recommended_weight=body.recommended_weight,
        base_compat=body.base_compat,
        modes=body.modes,
        license=body.license,
        enabled=body.enabled,
    )
    if lora is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LoRA not found")
    return LoraRead.from_lora(lora)


@router.delete("/{lora_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_lora(
    lora_id: str,
    session: DbSession,
    storage: Storage,
    principal: CurrentPrincipal,
) -> None:
    """Delete a LoRA and its stored blob."""
    removed = lora_manager.delete_lora(
        lora_id=lora_id, session=session, storage=storage, principal=principal
    )
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LoRA not found")
