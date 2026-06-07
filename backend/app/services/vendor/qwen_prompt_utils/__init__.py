"""Vendored Qwen-Image prompt-rewriting utilities (verbatim; see NOTICE.md).

These modules are copied **unmodified** from the official QwenLM/Qwen-Image repository so the
prompt-enhancement templates (the ``SYSTEM_PROMPT`` / ``EDIT_SYSTEM_PROMPT`` strings) match
the model authors' tuning exactly (DESIGN §5.8, CLAUDE.md constraint). Do not edit them; a
template update is a deliberate, attributed re-vendor.

The QIE rewriter does NOT call the upstream ``api()`` (which targets Alibaba DashScope) — it
extracts the system-prompt template text and runs it through a local Qwen-VL model. Use
:mod:`app.services.rewriter_templates` to pull the verbatim template strings out of these
files (single source of truth).
"""
