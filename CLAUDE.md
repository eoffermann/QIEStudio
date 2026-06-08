# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**v1 is implemented** (built unattended on branch `DEVRUN_202606070903`; see
`PROGRESS_DIARY.md` for the full build record + decision logs). The full §12 scope is in
place: FastAPI backend (`backend/app`) with config/DB/models/migrations and the
storage/auth/queue seams; device manager + VRAM advisor; resolution service; asset / prompt /
LoRA / integrations stores; pipeline service (Generate + Edit, bf16 / fp8 / int4 paths,
LoRA stacking, live latent previews); job queue + REST + WebSocket progress + batch/sweep;
Qwen-VL prompt enhancer (§5.8) and the round-out features (background removal, upscale,
recipes, catalog export); a React/Vite SPA (`frontend/`); and the CUDA + dev Docker images.

The full unit suite (~200 tests) is green in Docker against Postgres, and the **real
Generate→Edit loop has run end-to-end on the reference A6000 across all three precisions**
(bf16, fp8, and **Nunchaku SVDQuant int4** — the recommended path on Ampere/workstation/gaming
cards), plus a **LoRA-applied** generation (on bf16 *and* int4 via a precision-aware path), the
**Qwen-VL prompt enhancer** (multimodal), and **live latent previews** — all with output
evidence + reproducibility sidecars committed under `images/`. See DESIGN §9.4 for delivered
precision notes (fp8 is emulated on SM 8.6; int4 is native + fastest; int4 LoRAs use a
manual fp16 hook since PEFT can't wrap SVDQuant layers).

**`DESIGN.md` remains the source of truth** — keep it (and this file) in sync when scope or
decisions change.

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

Repo layout: `backend/` (FastAPI app + `pyproject.toml`, package `app`), `frontend/` (Vite
SPA), `docker-compose.yml` at root. Two images: **`backend/Dockerfile`** (CUDA, full
inference) and **`backend/Dockerfile.dev`** (lightweight `python:3.13-slim`, no torch — fast
unit loop). Postgres always runs in Docker (`db` service).

### Run the full app (CUDA)
```bash
cp .env.example .env          # set QIE_SECRET_KEY
docker compose --profile full up --build      # UI+API at http://localhost:8000 (/docs)
```

### Dev / tests (fast, no CUDA download)
```bash
docker compose up -d db                                   # Postgres only
docker compose --profile test run --rm backend-dev pytest                 # full unit suite
docker compose --profile test run --rm backend-dev pytest tests/test_resolution.py   # one file
docker compose --profile test run --rm backend-dev pytest tests/test_job_service.py::test_cancel_queued_job   # one test
docker compose --profile test run --rm backend-dev ruff check .           # lint
docker compose --profile test run --rm backend-dev mypy                   # type-check
```
DB-backed tests use the `session` fixture (skip if no Postgres). GPU tests are marked
`@pytest.mark.gpu` and skip without CUDA — run them in the CUDA image (below). The dev image
has no torch, so `pipeline_service`/`rewriter` import lazily and their pure logic is tested
without it.

### Migrations (Alembic; also auto-run on app startup)
```bash
# autogenerate against a DB at head, then apply:
docker compose --profile test run --rm backend-dev alembic revision --autogenerate -m "msg"
docker compose --profile test run --rm backend-dev alembic upgrade head
```

### Real GPU smoke test (CUDA image, on the accelerator)
```bash
docker build -f backend/Dockerfile -t qwenimageedit-backend .
# self-contained Generate->Edit loop (downloads real Qwen weights to ./models; writes to ./images):
docker run --rm --gpus all --network qwenimageedit_default \
  -e QIE_DATABASE_URL=postgresql+psycopg://qie:qie@db:5432/qie \
  -e HF_HOME=/models -v $PWD/models:/models -v $PWD/data:/data -v $PWD/images:/out \
  qwenimageedit-backend python -m scripts.smoke_generate_edit --precision bf16 --steps 30 --offload
# or the pytest GPU suite:
docker run --rm --gpus all --network qwenimageedit_default \
  -e QIE_DATABASE_URL=postgresql+psycopg://qie:qie@db:5432/qie \
  -e HF_HOME=/models -v $PWD/models:/models qwenimageedit-backend pytest -m gpu
```

### Frontend (host, Node 24)
```bash
cd frontend && npm install && npm run build      # -> frontend/dist (served by backend)
cd frontend && npm run dev                        # Vite dev server (proxies /api + /ws to :8000)
```

### Apple Silicon (MPS)
Run the backend natively (Python 3.13 venv, MPS) — GPU isn't reachable inside Docker on
macOS — and keep only `db` in Docker (DESIGN §8).
