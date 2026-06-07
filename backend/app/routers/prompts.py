"""Prompt library API (DESIGN §5.2, §7).

Exposes the prompt store: list/search, create/save, rename/edit/bindings (PATCH),
duplicate, and delete. The ordered image list (pinned assets + open slots) and LoRA
bindings round-trip through here; slot *resolution* happens later in the jobs subsystem
via :func:`app.services.prompt_store.resolve_inputs`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.deps import CurrentPrincipal, DbSession
from app.schemas.prompts import PromptCreate, PromptRead, PromptUpdate
from app.services import prompt_store

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


def _read(prompt_id: str, *, session: DbSession) -> PromptRead:
    """Assemble a :class:`PromptRead` for ``prompt_id`` (children loaded in order)."""
    prompt = prompt_store.get_prompt(prompt_id=prompt_id, session=session)
    if prompt is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prompt not found")
    images = prompt_store.get_prompt_images(prompt_id=prompt_id, session=session)
    loras = prompt_store.get_prompt_loras(prompt_id=prompt_id, session=session)
    return PromptRead.build(prompt, images, loras)


@router.get("", response_model=list[PromptRead])
def list_prompts(
    session: DbSession,
    principal: CurrentPrincipal,
    q: Annotated[str | None, Query()] = None,
    tag: Annotated[str | None, Query()] = None,
    mode: Annotated[str | None, Query()] = None,
    favorites: Annotated[bool, Query()] = False,
    recent: Annotated[bool, Query()] = False,
) -> list[PromptRead]:
    """List prompts, filtered by ``q``/``tag``/``mode`` and favorites/recent (DESIGN §7)."""
    prompts = prompt_store.list_prompts(
        session=session,
        principal=principal,
        q=q,
        tag=tag,
        mode=mode,
        favorites_only=favorites,
        recently_used=recent,
    )
    out: list[PromptRead] = []
    for prompt in prompts:
        images = prompt_store.get_prompt_images(prompt_id=prompt.id, session=session)
        loras = prompt_store.get_prompt_loras(prompt_id=prompt.id, session=session)
        out.append(PromptRead.build(prompt, images, loras))
    return out


@router.post("", response_model=PromptRead, status_code=status.HTTP_201_CREATED)
def create_prompt(
    body: PromptCreate,
    session: DbSession,
    principal: CurrentPrincipal,
) -> PromptRead:
    """Create / save a prompt with optional ordered image + LoRA bindings (DESIGN §7)."""
    prompt = prompt_store.create_prompt(
        session=session,
        principal=principal,
        name=body.name,
        text=body.text,
        tags=body.tags,
        mode=body.mode,
        favorite=body.favorite,
        defaults_json=body.defaults_json,
        images=body.images,
        loras=body.loras,
    )
    return _read(prompt.id, session=session)


@router.patch("/{prompt_id}", response_model=PromptRead)
def update_prompt(
    prompt_id: str,
    body: PromptUpdate,
    session: DbSession,
    principal: CurrentPrincipal,
) -> PromptRead:
    """Rename / edit a prompt and/or replace its ordered bindings (DESIGN §7)."""
    prompt = prompt_store.get_prompt(prompt_id=prompt_id, session=session)
    if prompt is None or prompt.owner_id != principal.owner_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prompt not found")
    prompt_store.update_prompt(
        session=session,
        prompt=prompt,
        name=body.name,
        text=body.text,
        tags=body.tags,
        mode=body.mode,
        favorite=body.favorite,
        defaults_json=body.defaults_json,
        images=body.images,
        loras=body.loras,
    )
    return _read(prompt_id, session=session)


@router.post(
    "/{prompt_id}/duplicate",
    response_model=PromptRead,
    status_code=status.HTTP_201_CREATED,
)
def duplicate_prompt(
    prompt_id: str,
    session: DbSession,
    principal: CurrentPrincipal,
    name: Annotated[str | None, Query()] = None,
) -> PromptRead:
    """Deep-copy a prompt (incl. children) under a new name (DESIGN §5.2)."""
    prompt = prompt_store.get_prompt(prompt_id=prompt_id, session=session)
    if prompt is None or prompt.owner_id != principal.owner_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prompt not found")
    copy = prompt_store.duplicate_prompt(
        session=session, principal=principal, prompt=prompt, new_name=name
    )
    return _read(copy.id, session=session)


@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_prompt(
    prompt_id: str,
    session: DbSession,
    principal: CurrentPrincipal,
) -> None:
    """Delete a prompt, cascading its child rows (DESIGN §5.2)."""
    prompt = prompt_store.get_prompt(prompt_id=prompt_id, session=session)
    if prompt is None or prompt.owner_id != principal.owner_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prompt not found")
    prompt_store.delete_prompt(session=session, prompt=prompt)
