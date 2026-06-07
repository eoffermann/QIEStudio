"""Recipe store service (DESIGN §13 #4).

Owns the recipe library: CRUD over the :class:`~app.models.recipe.Recipe` table plus
:func:`to_job_submit`, which hands back the stored composition ready to POST to
``/api/jobs``. A recipe is a saved, named **composition** — a full
:class:`~app.schemas.jobs.JobSubmit`-shaped config in ``config_json`` — so a whole setup
(prompt + LoRAs + resolution + settings + slot layout) re-runs in one click (DESIGN §13 #4).

This module talks only to the DB session and the request :class:`Principal`; it performs no
I/O against the storage provider or any accelerator. Every owned row is scoped by
``owner_id`` so multi-user becomes filtering, not a migration (DESIGN §4.1a).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlmodel import col, select

from app.models.recipe import Recipe

if TYPE_CHECKING:
    from sqlmodel import Session

    from app.interfaces.auth import Principal


def create_recipe(
    *,
    session: Session,
    principal: Principal,
    name: str,
    description: str = "",
    tags: list[str] | None = None,
    mode: str = "generate",
    config_json: dict[str, Any] | None = None,
) -> Recipe:
    """Create (save-with-name) a recipe from a JobSubmit-shaped config (DESIGN §13 #4)."""
    recipe = Recipe(
        owner_id=principal.owner_id,
        workspace_id=principal.workspace_id,
        name=name,
        description=description,
        tags=list(tags or []),
        mode=mode,
        config_json=dict(config_json or {}),
    )
    session.add(recipe)
    session.commit()
    session.refresh(recipe)
    return recipe


def get_recipe(*, recipe_id: str, session: Session) -> Recipe | None:
    """Return the recipe with ``recipe_id`` (or ``None`` if it does not exist)."""
    return session.get(Recipe, recipe_id)


def list_recipes(
    *,
    session: Session,
    principal: Principal,
    q: str | None = None,
    tag: str | None = None,
    mode: str | None = None,
) -> list[Recipe]:
    """List recipes for the principal, filtered + ordered newest-first (DESIGN §7, §13 #4).

    Filters: ``q`` (case-insensitive substring over name/description), ``tag`` (membership
    in the ``tags`` array), and ``mode`` (exact). Ordering is by ``updated_at`` descending.
    """
    stmt = select(Recipe).where(col(Recipe.owner_id) == principal.owner_id)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(col(Recipe.name).ilike(like) | col(Recipe.description).ilike(like))
    if tag:
        # PostgreSQL array-containment (the base ARRAY type has no .contains()).
        stmt = stmt.where(col(Recipe.tags).op("@>")([tag]))
    if mode:
        stmt = stmt.where(col(Recipe.mode) == mode)
    stmt = stmt.order_by(col(Recipe.updated_at).desc())
    return list(session.exec(stmt).all())


def update_recipe(
    *,
    session: Session,
    recipe: Recipe,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    mode: str | None = None,
    config_json: dict[str, Any] | None = None,
) -> Recipe:
    """Rename / edit a recipe (DESIGN §7). ``None`` fields are left unchanged."""
    if name is not None:
        recipe.name = name
    if description is not None:
        recipe.description = description
    if tags is not None:
        recipe.tags = list(tags)
    if mode is not None:
        recipe.mode = mode
    if config_json is not None:
        recipe.config_json = dict(config_json)
    recipe.updated_at = datetime.now(tz=recipe.created_at.tzinfo)
    session.add(recipe)
    session.commit()
    session.refresh(recipe)
    return recipe


def delete_recipe(*, session: Session, recipe: Recipe) -> None:
    """Delete a recipe (DESIGN §13 #4)."""
    session.delete(recipe)
    session.commit()


def to_job_submit(*, recipe: Recipe) -> dict[str, Any]:
    """Return the recipe's stored composition as a JobSubmit-shaped dict (DESIGN §13 #4).

    The result is a deep-ish copy of ``config_json`` with ``mode`` ensured to match the
    recipe's ``mode`` column, ready to POST to ``/api/jobs`` (the run composer's
    "one-click run config"). Validation of the payload is left to the jobs subsystem.
    """
    config = dict(recipe.config_json or {})
    config.setdefault("mode", recipe.mode)
    return config
