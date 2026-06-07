"""Tests for the recipe store service (DESIGN §13 #4).

Covers CRUD, owner-scoped list/filter (q/tag/mode), and ``to_job_submit`` (the one-click
run-config payload). DB-backed tests use the ``session`` fixture and skip when no PostgreSQL
is reachable (see ``conftest``).
"""

from __future__ import annotations

import pytest
from app.interfaces.auth import Principal
from app.services import recipe_store

pytestmark = pytest.mark.db

PRINCIPAL = Principal()

_CONFIG = {
    "mode": "edit",
    "prompt": "place the furniture in this living room",
    "loras": [{"lora_id": "L1", "weight": 0.8}],
    "resolution": {"base": 1024, "orientation": "landscape", "aspect": "3:2"},
    "num_inference_steps": 40,
}


def test_create_and_get(session: object) -> None:
    recipe = recipe_store.create_recipe(
        session=session,
        principal=PRINCIPAL,
        name="Staged room",
        description="furniture staging",
        tags=["staging"],
        mode="edit",
        config_json=_CONFIG,
    )
    fetched = recipe_store.get_recipe(recipe_id=recipe.id, session=session)
    assert fetched is not None
    assert fetched.name == "Staged room"
    assert fetched.tags == ["staging"]
    assert fetched.config_json == _CONFIG
    assert fetched.owner_id == PRINCIPAL.owner_id


def test_get_missing_returns_none(session: object) -> None:
    assert recipe_store.get_recipe(recipe_id="nope", session=session) is None


def test_update_partial_fields(session: object) -> None:
    recipe = recipe_store.create_recipe(
        session=session, principal=PRINCIPAL, name="Orig", mode="generate",
    )
    recipe_store.update_recipe(
        session=session, recipe=recipe, name="Renamed", tags=["a", "b"],
    )
    fetched = recipe_store.get_recipe(recipe_id=recipe.id, session=session)
    assert fetched.name == "Renamed"
    assert fetched.tags == ["a", "b"]
    assert fetched.mode == "generate"  # untouched


def test_update_config_json(session: object) -> None:
    recipe = recipe_store.create_recipe(
        session=session, principal=PRINCIPAL, name="C", config_json={"x": 1},
    )
    recipe_store.update_recipe(session=session, recipe=recipe, config_json={"y": 2})
    fetched = recipe_store.get_recipe(recipe_id=recipe.id, session=session)
    assert fetched.config_json == {"y": 2}


def test_delete(session: object) -> None:
    recipe = recipe_store.create_recipe(session=session, principal=PRINCIPAL, name="Bye")
    rid = recipe.id
    recipe_store.delete_recipe(session=session, recipe=recipe)
    assert recipe_store.get_recipe(recipe_id=rid, session=session) is None


def test_list_filters_q_tag_mode(session: object) -> None:
    recipe_store.create_recipe(
        session=session, principal=PRINCIPAL, name="Alpha sunset",
        description="warm tones", tags=["scenery"], mode="generate",
    )
    recipe_store.create_recipe(
        session=session, principal=PRINCIPAL, name="Beta room",
        description="staged", tags=["staging"], mode="edit",
    )

    by_q = recipe_store.list_recipes(session=session, principal=PRINCIPAL, q="sunset")
    assert [r.name for r in by_q] == ["Alpha sunset"]

    by_desc = recipe_store.list_recipes(session=session, principal=PRINCIPAL, q="staged")
    assert [r.name for r in by_desc] == ["Beta room"]

    by_tag = recipe_store.list_recipes(session=session, principal=PRINCIPAL, tag="staging")
    assert [r.name for r in by_tag] == ["Beta room"]

    by_mode = recipe_store.list_recipes(session=session, principal=PRINCIPAL, mode="generate")
    assert [r.name for r in by_mode] == ["Alpha sunset"]


def test_list_is_owner_scoped(session: object) -> None:
    recipe_store.create_recipe(session=session, principal=PRINCIPAL, name="Mine")
    other = Principal(owner_id="someone-else", workspace_id="someone-else")
    recipe_store.create_recipe(session=session, principal=other, name="Theirs")

    mine = recipe_store.list_recipes(session=session, principal=PRINCIPAL)
    assert [r.name for r in mine] == ["Mine"]


def test_to_job_submit_returns_config(session: object) -> None:
    recipe = recipe_store.create_recipe(
        session=session, principal=PRINCIPAL, name="R", mode="edit", config_json=_CONFIG,
    )
    payload = recipe_store.to_job_submit(recipe=recipe)
    assert payload["mode"] == "edit"
    assert payload["prompt"] == _CONFIG["prompt"]
    # A copy — mutating the payload must not touch the stored config.
    payload["prompt"] = "changed"
    assert recipe.config_json["prompt"] == _CONFIG["prompt"]


def test_to_job_submit_defaults_mode_from_column(session: object) -> None:
    recipe = recipe_store.create_recipe(
        session=session, principal=PRINCIPAL, name="NoModeInConfig",
        mode="generate", config_json={"prompt": "hi"},
    )
    payload = recipe_store.to_job_submit(recipe=recipe)
    assert payload["mode"] == "generate"
