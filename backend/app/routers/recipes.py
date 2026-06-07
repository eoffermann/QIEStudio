"""Recipes API — one-click run configs (DESIGN §13 #4, §7).

Exposes the recipe store: list/search, create/save, rename/edit (PATCH), delete, and
``instantiate`` (which hands back the stored composition as a JobSubmit-shaped payload ready
to POST to ``/api/jobs``). All work is delegated to :mod:`app.services.recipe_store`; every
row is owner-scoped (DESIGN §4.1a).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status

from app.deps import CurrentPrincipal, DbSession
from app.schemas.recipes import RecipeCreate, RecipeRead, RecipeUpdate
from app.services import recipe_store

router = APIRouter(prefix="/api/recipes", tags=["recipes"])


def _get_owned(recipe_id: str, *, session: DbSession, principal: CurrentPrincipal):  # noqa: ANN202
    """Fetch a recipe scoped to the principal, or raise 404 (DESIGN §4.1a)."""
    recipe = recipe_store.get_recipe(recipe_id=recipe_id, session=session)
    if recipe is None or recipe.owner_id != principal.owner_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Recipe not found")
    return recipe


@router.get("", response_model=list[RecipeRead])
def list_recipes(
    session: DbSession,
    principal: CurrentPrincipal,
    q: Annotated[str | None, Query()] = None,
    tag: Annotated[str | None, Query()] = None,
    mode: Annotated[str | None, Query()] = None,
) -> list[RecipeRead]:
    """List recipes, filtered by ``q``/``tag``/``mode`` (DESIGN §7, §13 #4)."""
    recipes = recipe_store.list_recipes(
        session=session, principal=principal, q=q, tag=tag, mode=mode
    )
    return [RecipeRead.from_recipe(r) for r in recipes]


@router.post("", response_model=RecipeRead, status_code=status.HTTP_201_CREATED)
def create_recipe(
    body: RecipeCreate,
    session: DbSession,
    principal: CurrentPrincipal,
) -> RecipeRead:
    """Create / save a recipe from a JobSubmit-shaped config (DESIGN §13 #4)."""
    recipe = recipe_store.create_recipe(
        session=session,
        principal=principal,
        name=body.name,
        description=body.description,
        tags=body.tags,
        mode=body.mode,
        config_json=body.config_json,
    )
    return RecipeRead.from_recipe(recipe)


@router.get("/{recipe_id}", response_model=RecipeRead)
def get_recipe(
    recipe_id: str,
    session: DbSession,
    principal: CurrentPrincipal,
) -> RecipeRead:
    """Fetch a single recipe (DESIGN §13 #4)."""
    return RecipeRead.from_recipe(_get_owned(recipe_id, session=session, principal=principal))


@router.patch("/{recipe_id}", response_model=RecipeRead)
def update_recipe(
    recipe_id: str,
    body: RecipeUpdate,
    session: DbSession,
    principal: CurrentPrincipal,
) -> RecipeRead:
    """Rename / edit a recipe (DESIGN §7)."""
    recipe = _get_owned(recipe_id, session=session, principal=principal)
    recipe_store.update_recipe(
        session=session,
        recipe=recipe,
        name=body.name,
        description=body.description,
        tags=body.tags,
        mode=body.mode,
        config_json=body.config_json,
    )
    return RecipeRead.from_recipe(recipe)


@router.delete("/{recipe_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recipe(
    recipe_id: str,
    session: DbSession,
    principal: CurrentPrincipal,
) -> None:
    """Delete a recipe (DESIGN §13 #4)."""
    recipe = _get_owned(recipe_id, session=session, principal=principal)
    recipe_store.delete_recipe(session=session, recipe=recipe)


@router.post("/{recipe_id}/instantiate")
def instantiate_recipe(
    recipe_id: str,
    session: DbSession,
    principal: CurrentPrincipal,
) -> dict[str, Any]:
    """Return the recipe's stored composition as a JobSubmit payload (DESIGN §13 #4).

    The returned dict is ready to POST to ``/api/jobs`` — the "one-click run config".
    """
    recipe = _get_owned(recipe_id, session=session, principal=principal)
    return recipe_store.to_job_submit(recipe=recipe)
