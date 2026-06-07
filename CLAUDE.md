# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repo is **pre-implementation**. The only artifact is **`DESIGN.md`**, a finalized
(v1.0) specification for *QIE Studio*. There is no code, build system, dependency manifest,
or test suite yet. **`DESIGN.md` is the source of truth** — read it before doing anything,
and keep it in sync when scope or decisions change.

When scaffolding, the spec's §12 build order is the intended sequence (backend skeleton →
device manager + advisor → pipeline service → resolution → asset store → job queue/previews
→ LoRA manager + integrations → prompt store → frontend → round-out features → Docker →
polish).

## What we're building

A self-hosted, Docker-deployable image **generation + editing** app around the Qwen-Image
family, with a bespoke modern web UI (explicitly **not** Gradio). Two run **modes** share
one workflow, libraries, and output flow:

- **Edit** — `Qwen/Qwen-Image-Edit-2511` via `diffusers.QwenImageEditPlusPipeline`; takes
  an **ordered list** of input images + an instruction. Image order is semantically
  meaningful to the model and must be preserved end-to-end.
- **Generate** — `Qwen/Qwen-Image` via `diffusers.QwenImagePipeline`; text-to-image, no
  input image.

The product is organized around four reusable building blocks that combine at run time:
**Images**, **Prompts**, **LoRAs**, **Resolution presets**.

## Locked stack (do not silently substitute)

- **Backend:** Python **3.13**, FastAPI + Uvicorn, Pydantic v2.
- **Persistence:** **PostgreSQL** (SQLModel/SQLAlchemy + Alembic). Not SQLite.
- **Inference:** PyTorch (CUDA / ROCm / MPS builds), `diffusers`, `transformers`,
  `accelerate`, `peft` (external LoRA loading), `safetensors`, `Pillow` (+ `pillow-heif`),
  plus `optimum-quanto`/`torchao` (fp8) and `nunchaku` (SVDQuant int4, **CUDA-only**).
- **Frontend:** React (Vite SPA) + TypeScript, Tailwind CSS v4 + shadcn/ui, Framer Motion,
  TanStack Query (server state) + Zustand (UI state). Served as static assets by the
  backend.

## Architecture concepts that span multiple parts of the spec

- **Future-proofing seams (§4.1a).** v1 is single-user/local, but the architecture must
  stay cloud- and multi-user-ready *without a rewrite*. Concretely, when implementing:
  - Access all binaries through a **`StorageProvider`** interface (`LocalFsStorage` now,
    object store later). The DB stores **storage keys, never absolute paths**.
  - Gate requests through an **`AuthProvider`** (`NoAuthProvider` / single implicit owner
    now). Every user-owned row carries a nullable **`owner_id`/`workspace_id`** defaulted
    to that implicit owner — multi-user becomes filtering, not a migration.
  - Enqueue work through a **`JobQueue`** interface (in-process FIFO now, broker later).
    Keep request handlers stateless.
- **Single-accelerator job queue.** One inference at a time; jobs are persisted (survive
  reloads), cancelable, and stream progress over WebSocket **including live latent
  previews** (throttled latent→RGB decode). **Single-GPU only in v1**; multi-GPU is
  explicitly future work.
- **Two VRAM-aware advisors.** Both read the `device_manager` (backend + SM/compute
  capability + free VRAM/RAM) and the planned run, then rank options:
  1. **Model advisor (§9)** — picks image-model precision among **bf16 / fp8_e4m3fn (scaled)
     / Nunchaku SVDQuant int4**, accounting for resolution, batch, and active LoRAs. Never
     offers CUDA-only options (int4) on ROCm/MPS.
  2. **Rewriter advisor (§5.8)** — picks the Qwen-VL prompt-enhancer model and decides
     **co-reside vs runtime-swap** with the loaded image pipeline.
- **Prompt slots & bindings (§5.2).** A saved prompt holds an *ordered* image list where
  each entry is either a **pinned** library asset (always included) or an **open slot** the
  user fills at run time. This drives the catalog workflow (pinned background + uploaded
  subjects) and **batch/sweep** (fill one slot with N images → N outputs).
- **Reproducibility.** Every job records mode, model id/revision, precision/quant, device,
  resolution, seed, LoRAs+weights, input image hashes, and **both original and enhanced
  prompts**. Persist this and embed it in PNG sidecar/metadata.

## Project-specific constraints (easy to get wrong)

- **Reuse the official prompt-rewriting templates verbatim.** The Qwen-VL prompt enhancer
  must vendor the system prompts from `QwenLM/Qwen-Image:
  src/examples/tools/prompt_utils.py` and `prompt_utils_2512.py` (version-matched to the
  chosen edit base), pinned and attributed — do not invent prompt-rewriting templates.
- **Reference inference params (Edit):** `num_inference_steps=40`, `true_cfg_scale=4.0`,
  `guidance_scale=1.0`, `negative_prompt=" "` (a single space), `torch_dtype=bfloat16`.
- **CivitAI imports** accept `civitai.com` / `civitai.red` / `civitai.green` URLs — these
  are filtered views over one DB. Strip the domain, extract the model/version id, and
  resolve via the canonical `civitai.com/api/v1` API (follow domain-color redirects; the
  API key authorizes mature content).
- **Resolution:** base size = **longer edge**; short edge derived from aspect ratio; both
  snapped to a multiple of **16**. "Match source" only applies in Edit mode.
- **Apple Silicon (MPS) runs the backend natively, not in Docker** (containers can't reach
  the Mac GPU); Postgres still runs in Docker. CUDA/ROCm get container images.
- **Integration API keys** (HF, CivitAI) are stored **encrypted at rest** and never
  returned to the client in plaintext.

## Commands

None yet — nothing is scaffolded. Populate this section (install, run dev backend/frontend,
Alembic migrations, lint, test, run a single test, Docker/Compose) as the tooling is added.
