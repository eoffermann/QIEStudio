# Vendored: QwenLM/Qwen-Image prompt-rewriting utilities

These files are copied **verbatim** (unmodified) from the official Qwen-Image repository and
used under its upstream license. They provide the prompt-enhancement system-prompt templates
that QIE Studio's Qwen-VL rewriter reuses exactly, per DESIGN §5.8 and the CLAUDE.md
constraint ("reuse the official prompt-rewriting templates verbatim … pinned and attributed
— do not invent prompt-rewriting templates").

| File | Upstream path |
|------|---------------|
| `prompt_utils.py` | `src/examples/tools/prompt_utils.py` |
| `prompt_utils_2512.py` | `src/examples/tools/prompt_utils_2512.py` |

- **Source repository:** https://github.com/QwenLM/Qwen-Image
- **Pinned commit:** `3453042c9f35f284abce58c68302a1183f6b40f2` (2025-12-23)
- **Raw source:** https://raw.githubusercontent.com/QwenLM/Qwen-Image/3453042c9f35f284abce58c68302a1183f6b40f2/src/examples/tools/prompt_utils.py
- **License:** as published by QwenLM/Qwen-Image (Apache-2.0 at time of vendoring). The
  upstream `LICENSE` governs these files.
- **Fetched:** 2026-06-07 during the QIE Studio v1 build.

## How QIE uses them

QIE does **not** call the upstream `api()` (it targets Alibaba DashScope). Instead,
`app/services/rewriter_templates.py` extracts the verbatim template strings
(`SYSTEM_PROMPT` for Generate, `EDIT_SYSTEM_PROMPT` for Edit; the `_2512` variant is selected
when the chosen edit base is a 2512-series model) and runs them through a locally-hosted
Qwen-VL model. Keeping the originals here unmodified makes them the single source of truth;
a template update is a deliberate, attributed re-vendor (bump the pinned commit above).

To update: re-fetch both files at a newer commit, update the pinned SHA + date here, and
verify `rewriter_templates.py` still extracts the expected symbols.
