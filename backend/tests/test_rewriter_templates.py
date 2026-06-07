"""Tests for verbatim template extraction (DESIGN §5.8 / CLAUDE.md constraint).

These assert the AST extraction returns the **exact** vendored template strings (the official
ones must be reused verbatim, not re-typed/invented). We round-trip: parse the vendored file
ourselves with an independent ``ast`` call and assert the module's getters equal that literal,
plus check a known unique substring is present.
"""

from __future__ import annotations

import ast
from importlib import resources

import pytest
from app.services.rewriter_templates import (
    get_edit_system_prompt,
    get_generate_system_prompt,
    select_template_variant,
)

_VENDOR = "app.services.vendor.qwen_prompt_utils"


def _literal(filename: str, func_name: str, var_name: str) -> str | None:
    """Independently extract the string literal assigned to ``var_name`` in ``func_name``."""
    source = resources.files(_VENDOR).joinpath(filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for stmt in ast.walk(node):
                if (
                    isinstance(stmt, ast.Assign)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == var_name:
                            return stmt.value.value
    return None


# --- Generate-mode SYSTEM_PROMPT -------------------------------------------------------


def test_generate_default_matches_vendored_literal() -> None:
    extracted = get_generate_system_prompt(variant="default")
    expected = _literal("prompt_utils.py", "polish_prompt_en", "SYSTEM_PROMPT")
    assert expected is not None
    assert extracted == expected
    assert extracted.strip() != ""
    # Known unique substring from the default Generate template.
    assert "You are a Prompt optimizer" in extracted


def test_generate_2512_matches_vendored_literal() -> None:
    extracted = get_generate_system_prompt(variant="2512")
    expected = _literal("prompt_utils_2512.py", "polish_prompt_en", "SYSTEM_PROMPT")
    assert expected is not None
    assert extracted == expected
    # The 2512 Generate template is the expanded "Image Prompt Rewriting Expert".
    assert "Image Prompt Rewriting Expert" in extracted


# --- Edit-mode EDIT_SYSTEM_PROMPT ------------------------------------------------------


def test_edit_default_matches_vendored_literal() -> None:
    extracted = get_edit_system_prompt(variant="default")
    expected = _literal("prompt_utils.py", "polish_edit_prompt", "EDIT_SYSTEM_PROMPT")
    assert expected is not None
    assert extracted == expected
    assert extracted.strip() != ""
    assert "Edit Prompt Enhancer" in extracted


def test_edit_2512_falls_back_to_default_edit_template() -> None:
    # The 2512 vendored file ships no EDIT_SYSTEM_PROMPT, so the 2512 edit getter must fall
    # back to the default file's verbatim edit template.
    assert _literal("prompt_utils_2512.py", "polish_edit_prompt", "EDIT_SYSTEM_PROMPT") is None
    extracted = get_edit_system_prompt(variant="2512")
    default = get_edit_system_prompt(variant="default")
    assert extracted == default
    assert "Edit Prompt Enhancer" in extracted


# --- Variant selection -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen/Qwen-Image-Edit-2512", "2512"),
        ("Qwen/Qwen-Image-Edit-2512-refresh", "2512"),
        ("some/Custom-2512-Model", "2512"),
        ("Qwen/Qwen-Image-Edit-2511", "default"),
        ("Qwen/Qwen-Image", "default"),
        ("", "default"),
    ],
)
def test_select_template_variant(model_id: str, expected: str) -> None:
    assert select_template_variant(edit_model_id=model_id) == expected


def test_generate_and_edit_templates_differ() -> None:
    # Sanity: the two templates are genuinely different official prompts.
    assert get_generate_system_prompt() != get_edit_system_prompt()
