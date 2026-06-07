"""Verbatim extraction of the official Qwen prompt-rewriting templates (DESIGN §5.8).

CLAUDE.md mandates we **reuse the official prompt-rewriting templates verbatim** rather than
invent our own. The official templates are vendored unmodified under
``app/services/vendor/qwen_prompt_utils/`` (see that package's ``NOTICE.md``) as
triple-quoted string literals assigned to local variables inside functions:

- ``SYSTEM_PROMPT`` inside ``polish_prompt_en``  — the Generate-mode (text-to-image) rewriter.
- ``EDIT_SYSTEM_PROMPT`` inside ``polish_edit_prompt`` — the Edit-mode (multimodal) rewriter.

To keep the vendored files the *single source of truth* (and avoid re-typing prompt text that
must stay byte-for-byte identical to upstream), this module **parses the vendored source with
:mod:`ast`** and pulls the exact string literal assigned to the target name out of the target
function. Nothing is hand-copied here.

Variant selection (DESIGN §5.8): the ``_2512`` vendored file is the 2512-series template; the
plain file is the default. The 2512 file only ships the Generate-mode ``SYSTEM_PROMPT`` (its
expanded rewriting-expert prompt subsumes the portrait/text/general cases), so the edit-mode
template gracefully falls back to the default file's ``EDIT_SYSTEM_PROMPT`` when the requested
variant lacks one.
"""

from __future__ import annotations

import ast
from functools import cache
from importlib import resources

# Vendored package + the two source files within it (verbatim upstream copies).
_VENDOR_PACKAGE = "app.services.vendor.qwen_prompt_utils"
_DEFAULT_FILE = "prompt_utils.py"
_VARIANT_FILES: dict[str, str] = {
    "default": _DEFAULT_FILE,
    "2512": "prompt_utils_2512.py",
}

# Where each template literal lives in the vendored source: (function name, assigned var).
_GENERATE_TARGET = ("polish_prompt_en", "SYSTEM_PROMPT")
_EDIT_TARGET = ("polish_edit_prompt", "EDIT_SYSTEM_PROMPT")


def _read_vendor_source(filename: str) -> str:
    """Read a vendored source file's text from the package (verbatim upstream copy)."""
    return resources.files(_VENDOR_PACKAGE).joinpath(filename).read_text(encoding="utf-8")


def _extract_assigned_string(source: str, *, func_name: str, var_name: str) -> str | None:
    """Return the string literal assigned to ``var_name`` inside ``func_name``, or ``None``.

    Walks the parsed AST to the target function, then finds the *first* assignment whose
    target is ``var_name`` and whose value is a string constant, returning that constant's
    exact value. Returns ``None`` when the function or assignment is absent (so callers can
    fall back to another variant) — this keeps the vendored files the sole source of truth.

    Args:
        source: The full Python source text of a vendored file.
        func_name: The enclosing function to search (e.g. ``"polish_prompt_en"``).
        var_name: The local variable whose string literal to extract (e.g. ``"SYSTEM_PROMPT"``).

    Returns:
        The verbatim string value, or ``None`` if not found.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for stmt in ast.walk(node):
                if not isinstance(stmt, ast.Assign):
                    continue
                if not (
                    isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    continue
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        return stmt.value.value
    return None


@cache
def _template(*, variant: str, func_name: str, var_name: str) -> str:
    """Extract a template string for ``variant``, falling back to the default file.

    Tries the requested variant's vendored file first; if that file does not define the
    target (e.g. the 2512 file has no ``EDIT_SYSTEM_PROMPT``), falls back to the default
    vendored file so a usable verbatim template is always returned.
    """
    filename = _VARIANT_FILES.get(variant, _DEFAULT_FILE)
    value = _extract_assigned_string(
        _read_vendor_source(filename), func_name=func_name, var_name=var_name
    )
    if value is None and filename != _DEFAULT_FILE:
        value = _extract_assigned_string(
            _read_vendor_source(_DEFAULT_FILE), func_name=func_name, var_name=var_name
        )
    if value is None:
        raise LookupError(
            f"Vendored template {var_name!r} not found in {func_name!r} "
            f"(variant={variant!r}); the vendored prompt_utils files may have changed."
        )
    return value


def get_generate_system_prompt(*, variant: str = "default") -> str:
    """Return the verbatim Generate-mode rewriter system prompt (``SYSTEM_PROMPT``).

    This is the text-to-image enrichment template the official tool feeds the model in
    ``polish_prompt_en`` (DESIGN §5.8). Extracted from the vendored file matching ``variant``.

    Args:
        variant: ``"default"`` or ``"2512"`` (see :func:`select_template_variant`).

    Returns:
        The exact upstream ``SYSTEM_PROMPT`` string (unmodified).
    """
    func_name, var_name = _GENERATE_TARGET
    return _template(variant=variant, func_name=func_name, var_name=var_name)


def get_edit_system_prompt(*, variant: str = "default") -> str:
    """Return the verbatim Edit-mode rewriter system prompt (``EDIT_SYSTEM_PROMPT``).

    This is the multimodal edit-instruction enhancement template from ``polish_edit_prompt``
    (DESIGN §5.8). The 2512 vendored file does not ship this symbol, so requesting the 2512
    variant falls back to the default file's verbatim ``EDIT_SYSTEM_PROMPT``.

    Args:
        variant: ``"default"`` or ``"2512"`` (see :func:`select_template_variant`).

    Returns:
        The exact upstream ``EDIT_SYSTEM_PROMPT`` string (unmodified).
    """
    func_name, var_name = _EDIT_TARGET
    return _template(variant=variant, func_name=func_name, var_name=var_name)


def select_template_variant(*, edit_model_id: str) -> str:
    """Pick the template variant matching the chosen edit base model (DESIGN §5.8).

    Returns ``"2512"`` for a 2512-series edit base (so the 2512 templates are used with the
    2512 base, per the spec), else ``"default"``. The match is on the ``2512`` marker anywhere
    in the model id, case-insensitively (covers ``Qwen/Qwen-Image-Edit-2512`` and refreshes).

    Args:
        edit_model_id: The selected edit-model HF repo id (may be empty/None-ish).

    Returns:
        ``"2512"`` or ``"default"``.
    """
    if edit_model_id and "2512" in edit_model_id.lower():
        return "2512"
    return "default"
