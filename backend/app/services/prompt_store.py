"""Prompt store service (DESIGN §5.2, §5.6).

Owns the prompt library: CRUD, tags/search/favorites/recently-used ordering, the two
ordered child collections (image slots & bindings, LoRA bindings), and — the key
workflow — **slot resolution**.

A saved prompt holds an *ordered* image list where each entry is either a **pinned**
library asset (always included) or an **open slot** the user fills at run time. At run
time :func:`resolve_inputs` walks that list in ``position`` order and composes the final
ordered model input — e.g. ``[livingroom(pinned), <uploaded furniture...>(slot)]`` — which
the jobs subsystem consumes for Edit mode and batch slot-sweep (DESIGN §5.6: fill one slot
with N images → N outputs).

This module talks only to the DB session and the request :class:`Principal`; it performs no
I/O against the storage provider or any accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlmodel import col, select

from app.models.prompt import Prompt, PromptImage, PromptLora

if TYPE_CHECKING:
    from sqlmodel import Session

    from app.interfaces.auth import Principal
    from app.schemas.prompts import PromptImageIn, PromptLoraIn

ROLE_PINNED = "pinned"
ROLE_SLOT = "slot"


@dataclass
class ResolvedInput:
    """One concrete input in the composed, ordered model-input list (DESIGN §5.2).

    Produced by :func:`resolve_inputs`. ``slot_name`` is the originating slot for entries
    that came from a fill, and ``None`` for pinned assets.
    """

    asset_id: str
    role: str  # "pinned" | "slot"
    slot_name: str | None


# --------------------------------------------------------------------------------------
# Read contracts (called by the jobs subsystem — exact names must not drift).
# --------------------------------------------------------------------------------------


def get_prompt(*, prompt_id: str, session: Session) -> Prompt | None:
    """Return the prompt with ``prompt_id`` (or ``None`` if it does not exist)."""
    return session.get(Prompt, prompt_id)


def get_prompt_images(*, prompt_id: str, session: Session) -> list[PromptImage]:
    """Return a prompt's image entries ordered by ``position`` (DESIGN §5.2)."""
    stmt = (
        select(PromptImage)
        .where(PromptImage.prompt_id == prompt_id)
        .order_by(col(PromptImage.position), col(PromptImage.id))
    )
    return list(session.exec(stmt).all())


def get_prompt_loras(*, prompt_id: str, session: Session) -> list[PromptLora]:
    """Return a prompt's LoRA bindings (DESIGN §5.2/§5.3)."""
    stmt = (
        select(PromptLora)
        .where(PromptLora.prompt_id == prompt_id)
        .order_by(col(PromptLora.id))
    )
    return list(session.exec(stmt).all())


# --------------------------------------------------------------------------------------
# Slot resolution — THE key workflow (DESIGN §5.2, the furniture example).
# --------------------------------------------------------------------------------------


def resolve_inputs(
    *,
    prompt: Prompt,
    prompt_images: list[PromptImage],
    slot_fills: dict[str, list[str]],
    session: Session,  # noqa: ARG001 — kept for signature stability / future validation
) -> list[ResolvedInput]:
    """Compose the ordered model-input list from a prompt's images + run-time slot fills.

    Walk ``prompt_images`` in ``position`` order. For ``role="pinned"`` emit its
    ``asset_id``; for ``role="slot"`` emit the asset ids the user supplied in
    ``slot_fills[slot_name]`` (in the order given). The resulting list follows the prompt's
    declared positions, e.g. ``[livingroom(pinned), <furniture...>(slot)]`` — that order is
    semantically meaningful to the Edit model and must be preserved.

    Args:
        prompt: The prompt being run (used for error messages).
        prompt_images: The prompt's image entries, ordered by ``position``.
        slot_fills: Map of ``slot_name`` → ordered asset ids the user dropped into that slot.
        session: DB session (reserved for future asset-existence validation; unused today).

    Returns:
        The composed ordered list of :class:`ResolvedInput`.

    Raises:
        ValueError: If a pinned entry has no ``asset_id``; a slot entry has no ``slot_name``;
            a slot's fill count violates its declared ``slot_min``/``slot_max``; or fills are
            supplied for slot names the prompt does not declare.
    """
    declared_slots = {pi.slot_name for pi in prompt_images if pi.role == ROLE_SLOT}
    unknown = set(slot_fills) - declared_slots
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(
            f"Prompt {prompt.name!r} has no slot(s) named: {names}"
        )

    resolved: list[ResolvedInput] = []
    for entry in prompt_images:
        if entry.role == ROLE_PINNED:
            if not entry.asset_id:
                raise ValueError(
                    f"Pinned image at position {entry.position} has no asset_id"
                )
            resolved.append(
                ResolvedInput(asset_id=entry.asset_id, role=ROLE_PINNED, slot_name=None)
            )
            continue

        if entry.role == ROLE_SLOT:
            name = entry.slot_name
            if not name:
                raise ValueError(
                    f"Slot at position {entry.position} has no slot_name"
                )
            fills = list(slot_fills.get(name, []))
            _validate_slot_count(entry, len(fills))
            resolved.extend(
                ResolvedInput(asset_id=aid, role=ROLE_SLOT, slot_name=name)
                for aid in fills
            )
            continue

        raise ValueError(
            f"Unknown image role {entry.role!r} at position {entry.position}"
        )

    return resolved


def _validate_slot_count(entry: PromptImage, count: int) -> None:
    """Validate a slot's fill count against its declared min/max (DESIGN §5.2)."""
    lo = entry.slot_min
    hi = entry.slot_max
    if lo is not None and count < lo:
        raise ValueError(
            f"Slot {entry.slot_name!r} requires at least {lo} image(s), got {count}"
        )
    if hi is not None and count > hi:
        raise ValueError(
            f"Slot {entry.slot_name!r} accepts at most {hi} image(s), got {count}"
        )


# --------------------------------------------------------------------------------------
# CRUD + bindings (called by the router; principal scopes every owned row).
# --------------------------------------------------------------------------------------


def create_prompt(
    *,
    session: Session,
    principal: Principal,
    name: str,
    text: str = "",
    tags: list[str] | None = None,
    mode: str = "any",
    favorite: bool = False,
    defaults_json: dict[str, Any] | None = None,
    images: list[PromptImageIn] | None = None,
    loras: list[PromptLoraIn] | None = None,
) -> Prompt:
    """Create (save-with-name) a prompt, optionally with its ordered child collections."""
    prompt = Prompt(
        owner_id=principal.owner_id,
        workspace_id=principal.workspace_id,
        name=name,
        text=text,
        tags=list(tags or []),
        mode=mode,
        favorite=favorite,
        defaults_json=dict(defaults_json or {}),
    )
    session.add(prompt)
    session.flush()  # assign prompt.id before attaching children

    if images is not None:
        _replace_images(session=session, prompt_id=prompt.id, images=images)
    if loras is not None:
        _replace_loras(session=session, prompt_id=prompt.id, loras=loras)

    session.commit()
    session.refresh(prompt)
    return prompt


def update_prompt(
    *,
    session: Session,
    prompt: Prompt,
    name: str | None = None,
    text: str | None = None,
    tags: list[str] | None = None,
    mode: str | None = None,
    favorite: bool | None = None,
    defaults_json: dict[str, Any] | None = None,
    images: list[PromptImageIn] | None = None,
    loras: list[PromptLoraIn] | None = None,
) -> Prompt:
    """Rename / edit a prompt and optionally replace its bindings (DESIGN §7).

    Scalar fields are updated only when not ``None``. Supplying ``images`` or ``loras``
    atomically replaces that ordered collection (see :func:`set_bindings`).
    """
    if name is not None:
        prompt.name = name
    if text is not None:
        prompt.text = text
    if tags is not None:
        prompt.tags = list(tags)
    if mode is not None:
        prompt.mode = mode
    if favorite is not None:
        prompt.favorite = favorite
    if defaults_json is not None:
        prompt.defaults_json = dict(defaults_json)
    prompt.updated_at = datetime.now(tz=prompt.created_at.tzinfo)

    session.add(prompt)
    if images is not None:
        _replace_images(session=session, prompt_id=prompt.id, images=images)
    if loras is not None:
        _replace_loras(session=session, prompt_id=prompt.id, loras=loras)

    session.commit()
    session.refresh(prompt)
    return prompt


def set_bindings(
    *,
    session: Session,
    prompt: Prompt,
    images: list[PromptImageIn] | None = None,
    loras: list[PromptLoraIn] | None = None,
) -> Prompt:
    """Atomically replace a prompt's ordered image list and/or LoRA list (DESIGN §5.2).

    Passing ``None`` leaves that collection untouched; passing ``[]`` clears it.
    """
    if images is not None:
        _replace_images(session=session, prompt_id=prompt.id, images=images)
    if loras is not None:
        _replace_loras(session=session, prompt_id=prompt.id, loras=loras)
    prompt.updated_at = datetime.now(tz=prompt.created_at.tzinfo)
    session.add(prompt)
    session.commit()
    session.refresh(prompt)
    return prompt


def duplicate_prompt(
    *,
    session: Session,
    principal: Principal,
    prompt: Prompt,
    new_name: str | None = None,
) -> Prompt:
    """Deep-copy a prompt (including its ordered children) under a new name.

    Defaults the copy's name to ``"<name> (copy)"`` (DESIGN §5.2 CRUD).
    """
    copy = Prompt(
        owner_id=principal.owner_id,
        workspace_id=principal.workspace_id,
        name=new_name or f"{prompt.name} (copy)",
        text=prompt.text,
        tags=list(prompt.tags),
        mode=prompt.mode,
        favorite=prompt.favorite,
        defaults_json=dict(prompt.defaults_json),
    )
    session.add(copy)
    session.flush()

    for src in get_prompt_images(prompt_id=prompt.id, session=session):
        session.add(
            PromptImage(
                prompt_id=copy.id,
                position=src.position,
                role=src.role,
                asset_id=src.asset_id,
                slot_name=src.slot_name,
                slot_min=src.slot_min,
                slot_max=src.slot_max,
                hint=src.hint,
            )
        )
    for src_lora in get_prompt_loras(prompt_id=prompt.id, session=session):
        session.add(
            PromptLora(
                prompt_id=copy.id,
                lora_id=src_lora.lora_id,
                weight=src_lora.weight,
            )
        )

    session.commit()
    session.refresh(copy)
    return copy


def delete_prompt(*, session: Session, prompt: Prompt) -> None:
    """Delete a prompt and cascade-delete its child rows (DESIGN §5.2)."""
    for child in get_prompt_images(prompt_id=prompt.id, session=session):
        session.delete(child)
    for child_lora in get_prompt_loras(prompt_id=prompt.id, session=session):
        session.delete(child_lora)
    # Flush the child deletes before removing the parent so the FK constraint holds
    # (no ORM relationship cascade is declared on the models).
    session.flush()
    session.delete(prompt)
    session.commit()


def list_prompts(
    *,
    session: Session,
    principal: Principal,
    q: str | None = None,
    tag: str | None = None,
    mode: str | None = None,
    favorites_only: bool = False,
    recently_used: bool = False,
) -> list[Prompt]:
    """List prompts for the principal, filtered + ordered (DESIGN §5.2, §7).

    Filters: ``q`` (case-insensitive substring over name/text), ``tag`` (membership in the
    ``tags`` array), ``mode`` (exact), and ``favorites_only``. Ordering: by ``last_used_at``
    (most recent first, nulls last) when ``recently_used`` else by ``updated_at`` desc.
    """
    stmt = select(Prompt).where(col(Prompt.owner_id) == principal.owner_id)

    if q:
        like = f"%{q}%"
        stmt = stmt.where(col(Prompt.name).ilike(like) | col(Prompt.text).ilike(like))
    if tag:
        # PostgreSQL array-containment (the base ARRAY type has no .contains()).
        stmt = stmt.where(col(Prompt.tags).op("@>")([tag]))
    if mode:
        stmt = stmt.where(col(Prompt.mode) == mode)
    if favorites_only:
        stmt = stmt.where(col(Prompt.favorite).is_(True))

    if recently_used:
        stmt = stmt.order_by(col(Prompt.last_used_at).desc().nulls_last())
    else:
        stmt = stmt.order_by(col(Prompt.updated_at).desc())

    return list(session.exec(stmt).all())


def touch_last_used(*, session: Session, prompt: Prompt) -> Prompt:
    """Stamp ``last_used_at`` (recently-used ordering); used when a prompt is run."""
    prompt.last_used_at = datetime.now(tz=prompt.created_at.tzinfo)
    session.add(prompt)
    session.commit()
    session.refresh(prompt)
    return prompt


# --------------------------------------------------------------------------------------
# Internal helpers — atomic child-collection replacement.
# --------------------------------------------------------------------------------------


def _replace_images(
    *, session: Session, prompt_id: str, images: list[PromptImageIn]
) -> None:
    """Replace a prompt's ordered image list, assigning sequential ``position`` values."""
    for existing in get_prompt_images(prompt_id=prompt_id, session=session):
        session.delete(existing)
    session.flush()
    for position, spec in enumerate(images):
        session.add(
            PromptImage(
                prompt_id=prompt_id,
                position=position,
                role=spec.role,
                asset_id=spec.asset_id,
                slot_name=spec.slot_name,
                slot_min=spec.slot_min,
                slot_max=spec.slot_max,
                hint=spec.hint,
            )
        )
    session.flush()


def _replace_loras(
    *, session: Session, prompt_id: str, loras: list[PromptLoraIn]
) -> None:
    """Replace a prompt's LoRA bindings."""
    for existing in get_prompt_loras(prompt_id=prompt_id, session=session):
        session.delete(existing)
    session.flush()
    for spec in loras:
        session.add(
            PromptLora(prompt_id=prompt_id, lora_id=spec.lora_id, weight=spec.weight)
        )
    session.flush()
