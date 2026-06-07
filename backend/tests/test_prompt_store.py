"""Tests for the prompt store service (DESIGN §5.2).

Covers CRUD, deep-copy duplicate, bindings round-trip, and — the key workflow —
:func:`~app.services.prompt_store.resolve_inputs` (pinned + slot ordering and
min/max validation, success and failure). DB-backed tests use the ``session``
fixture and skip when no PostgreSQL is reachable (see ``conftest``).
"""

from __future__ import annotations

import pytest
from app.interfaces.auth import Principal
from app.models.asset import Asset
from app.models.lora import Lora
from app.schemas.prompts import PromptImageIn, PromptLoraIn
from app.services import prompt_store
from app.services.prompt_store import ResolvedInput

pytestmark = pytest.mark.db

PRINCIPAL = Principal()


def _seed_asset(session: object, asset_id: str) -> None:
    """Insert a minimal Asset so PromptImage.asset_id FK references resolve."""
    if session.get(Asset, asset_id) is None:
        session.add(Asset(id=asset_id, storage_key=f"assets/{asset_id}.png"))
        session.commit()


def _seed_lora(session: object, lora_id: str) -> None:
    """Insert a minimal Lora so PromptLora.lora_id FK references resolve."""
    if session.get(Lora, lora_id) is None:
        session.add(Lora(id=lora_id, name=lora_id, storage_key=f"loras/{lora_id}.st"))
        session.commit()


# --------------------------------------------------------------------------------------
# resolve_inputs — pure-ish logic, but exercised with persisted rows for realism.
# --------------------------------------------------------------------------------------


def _furniture_prompt(session: object) -> object:
    """Create the canonical pinned-livingroom + furniture-slot prompt (DESIGN §5.2)."""
    _seed_asset(session, "livingroom-001")
    return prompt_store.create_prompt(
        session=session,
        principal=PRINCIPAL,
        name="Furniture in living room",
        text="Place the furniture in this living room.",
        mode="edit",
        images=[
            PromptImageIn(role="pinned", asset_id="livingroom-001"),
            PromptImageIn(
                role="slot", slot_name="furniture", slot_min=1, slot_max=3,
                hint="Upload the furniture piece(s)",
            ),
        ],
    )


def test_resolve_inputs_pinned_then_slot_order(session: object) -> None:
    prompt = _furniture_prompt(session)
    images = prompt_store.get_prompt_images(prompt_id=prompt.id, session=session)

    resolved = prompt_store.resolve_inputs(
        prompt=prompt,
        prompt_images=images,
        slot_fills={"furniture": ["chair-1", "sofa-2"]},
        session=session,
    )

    assert resolved == [
        ResolvedInput(asset_id="livingroom-001", role="pinned", slot_name=None),
        ResolvedInput(asset_id="chair-1", role="slot", slot_name="furniture"),
        ResolvedInput(asset_id="sofa-2", role="slot", slot_name="furniture"),
    ]


def test_resolve_inputs_min_violation(session: object) -> None:
    prompt = _furniture_prompt(session)
    images = prompt_store.get_prompt_images(prompt_id=prompt.id, session=session)

    with pytest.raises(ValueError, match="at least 1"):
        prompt_store.resolve_inputs(
            prompt=prompt, prompt_images=images, slot_fills={"furniture": []},
            session=session,
        )


def test_resolve_inputs_max_violation(session: object) -> None:
    prompt = _furniture_prompt(session)
    images = prompt_store.get_prompt_images(prompt_id=prompt.id, session=session)

    with pytest.raises(ValueError, match="at most 3"):
        prompt_store.resolve_inputs(
            prompt=prompt, prompt_images=images,
            slot_fills={"furniture": ["a", "b", "c", "d"]}, session=session,
        )


def test_resolve_inputs_unknown_slot(session: object) -> None:
    prompt = _furniture_prompt(session)
    images = prompt_store.get_prompt_images(prompt_id=prompt.id, session=session)

    with pytest.raises(ValueError, match="no slot"):
        prompt_store.resolve_inputs(
            prompt=prompt, prompt_images=images,
            slot_fills={"furniture": ["a"], "nope": ["x"]}, session=session,
        )


def test_resolve_inputs_multiple_slots_interleaved(session: object) -> None:
    _seed_asset(session, "mid")
    prompt = prompt_store.create_prompt(
        session=session,
        principal=PRINCIPAL,
        name="Two slots",
        images=[
            PromptImageIn(role="slot", slot_name="a", slot_min=0, slot_max=2),
            PromptImageIn(role="pinned", asset_id="mid"),
            PromptImageIn(role="slot", slot_name="b", slot_min=1, slot_max=1),
        ],
    )
    images = prompt_store.get_prompt_images(prompt_id=prompt.id, session=session)

    resolved = prompt_store.resolve_inputs(
        prompt=prompt, prompt_images=images,
        slot_fills={"a": ["a1"], "b": ["b1"]}, session=session,
    )
    assert [r.asset_id for r in resolved] == ["a1", "mid", "b1"]
    assert [r.role for r in resolved] == ["slot", "pinned", "slot"]


def test_resolve_inputs_pinned_missing_asset(session: object) -> None:
    prompt = prompt_store.create_prompt(
        session=session, principal=PRINCIPAL, name="Bad pinned",
        images=[PromptImageIn(role="pinned", asset_id=None)],
    )
    images = prompt_store.get_prompt_images(prompt_id=prompt.id, session=session)
    with pytest.raises(ValueError, match="no asset_id"):
        prompt_store.resolve_inputs(
            prompt=prompt, prompt_images=images, slot_fills={}, session=session,
        )


# --------------------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------------------


def test_create_and_get(session: object) -> None:
    prompt = prompt_store.create_prompt(
        session=session, principal=PRINCIPAL, name="Hello",
        text="world", tags=["x"], mode="generate", defaults_json={"steps": 40},
    )
    fetched = prompt_store.get_prompt(prompt_id=prompt.id, session=session)
    assert fetched is not None
    assert fetched.name == "Hello"
    assert fetched.tags == ["x"]
    assert fetched.defaults_json == {"steps": 40}


def test_get_missing_returns_none(session: object) -> None:
    assert prompt_store.get_prompt(prompt_id="does-not-exist", session=session) is None


def test_update_scalar_fields(session: object) -> None:
    prompt = prompt_store.create_prompt(
        session=session, principal=PRINCIPAL, name="Orig", text="t",
    )
    prompt_store.update_prompt(
        session=session, prompt=prompt, name="Renamed", favorite=True, tags=["new"],
    )
    fetched = prompt_store.get_prompt(prompt_id=prompt.id, session=session)
    assert fetched.name == "Renamed"
    assert fetched.favorite is True
    assert fetched.tags == ["new"]
    assert fetched.text == "t"  # untouched (None means leave unchanged)


def test_delete_cascades_children(session: object) -> None:
    prompt = _furniture_prompt(session)
    _seed_lora(session, "L1")
    prompt_store.set_bindings(
        session=session, prompt=prompt,
        loras=[PromptLoraIn(lora_id="L1", weight=0.8)],
    )
    pid = prompt.id
    prompt_store.delete_prompt(session=session, prompt=prompt)

    assert prompt_store.get_prompt(prompt_id=pid, session=session) is None
    assert prompt_store.get_prompt_images(prompt_id=pid, session=session) == []
    assert prompt_store.get_prompt_loras(prompt_id=pid, session=session) == []


# --------------------------------------------------------------------------------------
# Bindings round-trip
# --------------------------------------------------------------------------------------


def test_bindings_round_trip_and_ordering(session: object) -> None:
    prompt = prompt_store.create_prompt(
        session=session, principal=PRINCIPAL, name="Bind",
    )
    _seed_asset(session, "bg")
    _seed_lora(session, "L1")
    _seed_lora(session, "L2")
    prompt_store.set_bindings(
        session=session,
        prompt=prompt,
        images=[
            PromptImageIn(role="pinned", asset_id="bg"),
            PromptImageIn(role="slot", slot_name="s", slot_min=1, slot_max=2, hint="h"),
        ],
        loras=[
            PromptLoraIn(lora_id="L1", weight=0.5),
            PromptLoraIn(lora_id="L2", weight=1.1),
        ],
    )

    images = prompt_store.get_prompt_images(prompt_id=prompt.id, session=session)
    assert [i.position for i in images] == [0, 1]
    assert images[0].role == "pinned"
    assert images[0].asset_id == "bg"
    assert images[1].role == "slot"
    assert images[1].slot_name == "s"
    assert images[1].slot_min == 1
    assert images[1].hint == "h"

    loras = prompt_store.get_prompt_loras(prompt_id=prompt.id, session=session)
    assert {(loro.lora_id, loro.weight) for loro in loras} == {("L1", 0.5), ("L2", 1.1)}


def test_set_bindings_replaces_atomically(session: object) -> None:
    prompt = _furniture_prompt(session)
    _seed_asset(session, "only")
    # Replace with a single pinned image; old pinned + slot must be gone.
    prompt_store.set_bindings(
        session=session, prompt=prompt,
        images=[PromptImageIn(role="pinned", asset_id="only")],
    )
    images = prompt_store.get_prompt_images(prompt_id=prompt.id, session=session)
    assert len(images) == 1
    assert images[0].asset_id == "only"
    assert images[0].position == 0


def test_set_bindings_clear_with_empty_list(session: object) -> None:
    prompt = _furniture_prompt(session)
    prompt_store.set_bindings(session=session, prompt=prompt, images=[])
    assert prompt_store.get_prompt_images(prompt_id=prompt.id, session=session) == []


# --------------------------------------------------------------------------------------
# Duplicate (deep copy)
# --------------------------------------------------------------------------------------


def test_duplicate_deep_copies_children(session: object) -> None:
    prompt = _furniture_prompt(session)
    _seed_lora(session, "L1")
    prompt_store.set_bindings(
        session=session, prompt=prompt,
        loras=[PromptLoraIn(lora_id="L1", weight=0.8)],
    )

    copy = prompt_store.duplicate_prompt(
        session=session, principal=PRINCIPAL, prompt=prompt,
    )

    assert copy.id != prompt.id
    assert copy.name == "Furniture in living room (copy)"

    orig_images = prompt_store.get_prompt_images(prompt_id=prompt.id, session=session)
    copy_images = prompt_store.get_prompt_images(prompt_id=copy.id, session=session)
    assert len(copy_images) == len(orig_images) == 2
    # Distinct child rows (new ids) but identical content + ordering.
    assert {i.id for i in copy_images}.isdisjoint({i.id for i in orig_images})
    assert [(i.role, i.position, i.asset_id, i.slot_name) for i in copy_images] == [
        (i.role, i.position, i.asset_id, i.slot_name) for i in orig_images
    ]

    copy_loras = prompt_store.get_prompt_loras(prompt_id=copy.id, session=session)
    assert [(loro.lora_id, loro.weight) for loro in copy_loras] == [("L1", 0.8)]


def test_duplicate_is_independent(session: object) -> None:
    prompt = _furniture_prompt(session)
    copy = prompt_store.duplicate_prompt(
        session=session, principal=PRINCIPAL, prompt=prompt, new_name="Indep",
    )
    # Mutating the copy must not affect the original.
    prompt_store.set_bindings(session=session, prompt=copy, images=[])
    assert prompt_store.get_prompt_images(prompt_id=copy.id, session=session) == []
    assert len(prompt_store.get_prompt_images(prompt_id=prompt.id, session=session)) == 2


# --------------------------------------------------------------------------------------
# list / search / favorites / recently-used
# --------------------------------------------------------------------------------------


def test_list_filters_and_favorites(session: object) -> None:
    prompt_store.create_prompt(
        session=session, principal=PRINCIPAL, name="Alpha sunset",
        tags=["scenery"], mode="generate", favorite=True,
    )
    prompt_store.create_prompt(
        session=session, principal=PRINCIPAL, name="Beta room",
        tags=["staging"], mode="edit",
    )

    by_q = prompt_store.list_prompts(session=session, principal=PRINCIPAL, q="sunset")
    assert [p.name for p in by_q] == ["Alpha sunset"]

    by_tag = prompt_store.list_prompts(session=session, principal=PRINCIPAL, tag="staging")
    assert [p.name for p in by_tag] == ["Beta room"]

    by_mode = prompt_store.list_prompts(session=session, principal=PRINCIPAL, mode="edit")
    assert [p.name for p in by_mode] == ["Beta room"]

    favs = prompt_store.list_prompts(
        session=session, principal=PRINCIPAL, favorites_only=True,
    )
    assert [p.name for p in favs] == ["Alpha sunset"]


def test_list_recently_used_ordering(session: object) -> None:
    p1 = prompt_store.create_prompt(session=session, principal=PRINCIPAL, name="P1")
    p2 = prompt_store.create_prompt(session=session, principal=PRINCIPAL, name="P2")
    prompt_store.touch_last_used(session=session, prompt=p2)

    recent = prompt_store.list_prompts(
        session=session, principal=PRINCIPAL, recently_used=True,
    )
    # p2 has last_used_at set, so it sorts first (nulls last).
    assert recent[0].name == "P2"
    assert {p.name for p in recent} == {"P1", "P2"}
    _ = p1  # referenced for clarity
