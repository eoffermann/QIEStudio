# QIE Studio — Progress Diary

This is the primary record of an unattended autonomous build of **QIE Studio** (see
`DESIGN.md` for the authoritative spec and `RUN.md` for the runbook governing this run).
Every entry is timestamped (local time) and names the commit it corresponds to. The diary
is written to be **resumable**: after any interruption, read the latest entry to learn what
is done, in progress, and next.

- **Project:** QIE Studio (Qwen-Image generation + editing studio)
- **Run start (local):** 2026-06-07 09:03 PDT (-0700)
- **Build branch:** `DEVRUN_202606070903`
- **Remote:** `origin` → https://github.com/eoffermann/QIEStudio.git (configured — pushes are real)

---

## Entry 1 — Run start & environment probe

- **Local time:** 2026-06-07 09:03 PDT
- **Commit:** (this entry is committed as the first diary commit on the build branch)

### What I set out to do
Execute the `RUN.md` §3 startup sequence: create the build branch, initialize this diary,
and probe + record the build environment before writing any code.

### Environment probed (reproducibility facts)

| Facet | Value |
|-------|-------|
| Host OS | Windows 11 Pro 10.0.26220 (win32) |
| Shell | PowerShell (Bash also available) |
| CPU | 12 logical processors |
| System RAM | 65,334 MB total (~34.7 GB free at probe time) |
| GPU | **NVIDIA RTX A6000**, 49,140 MiB (48 GB) VRAM, ~537 MiB in use at probe |
| GPU arch | Ampere, compute capability **SM 8.6** |
| NVIDIA driver | 582.16 |
| CUDA (driver) | 13.0 |
| CUDA (nvcc toolkit) | 13.1 (release 13.1.115) |
| Docker | 29.5.2 (build 79eb04c) |
| Docker Compose | v5.1.3 |
| Node.js / npm | v24.15.0 / 11.12.1 |
| Host Python | 3.9.7 (backend targets **3.13 inside Docker**, so host version is irrelevant) |
| Disk (H:) | ~4.2 TB free — ample for tens-of-GB model weights |
| Network | Available (HF / CivitAI downloads permitted per RUN.md §2) |

### Hardware implications for precision choices (per RUN.md §2)
- **48 GB** comfortably fits bf16 for a single ~20B pipeline; headroom tightens once a VL
  rewriter, LoRAs, larger latents, or batches are added → exercise the model advisor.
- **Ampere (SM 8.6):** native `fp8_e4m3fn` acceleration needs SM ≥ 8.9 (Ada/Hopper). On
  this card fp8 is **emulated/cast** — validate that it *runs*, but expect no speedup.
- **Nunchaku SVDQuant int4** supports Ampere → preferred low-VRAM path; CUDA-only.

### What I did
1. Read `RUN.md`, `DESIGN.md`, and `CLAUDE.md` fully.
2. Created and switched to build branch `DEVRUN_202606070903`.
3. Probed GPU/CUDA/Docker/Node/Python/disk/remote and recorded the facts above.
4. Initialized this diary.

### Outcome
Startup sequence complete. Remote is configured, so the §8 commit/push cadence applies
(push the diary immediately after each diary entry; push features when they land).

### State / resumability
- **Done:** branch created, diary initialized, environment probed.
- **In progress:** none.
- **Next:** lay the backend foundation skeleton (§12 step 1) — Python 3.13 package layout,
  `pyproject`, config, PostgreSQL + SQLModel models, Alembic, and the `StorageProvider` /
  `AuthProvider` / `JobQueue` interfaces with their v1 local implementations — then fan out
  parallel agents across the disjoint subsystems.
- **Reconcile on resume:** nothing running yet; no containers, no partial downloads.

### Orchestration plan (how parallelism will be applied)
This build is parallelized by partitioning work into **disjoint files/modules** so multiple
agents never edit the same file:
1. *(sequential, critical path)* Lay a solid shared foundation: app package, config, DB
   base + models, migrations scaffold, and the storage/auth/queue interfaces.
2. *(parallel fan-out)* One agent per disjoint subsystem (device manager + advisor,
   resolution service, asset store, prompt store, lora manager + integrations, job queue +
   WebSocket, pipeline service) — each owns its module(s) + unit tests.
3. *(parallel)* Frontend scaffold against the API contract (fully disjoint `frontend/` tree).
4. *(sequential, stateful)* Integrate, build the CUDA Docker image, run the real
   Generate→Edit smoke tests on the A6000 inside Docker, then round-out features + polish.

---

## Entry 2 — Backend foundation skeleton (§12 step 1)

- **Local time:** 2026-06-07 09:30 PDT
- **Commit:** `082b43d` (foundation) — diary committed separately per §8.

### What I set out to do
Lay the shared backend substrate everything else depends on, get it building + tested +
linted, and prove the DB/migrations path end-to-end before fanning out parallel agents.

### What I did / outcome
- **Project config:** `.gitattributes` (LF normalization — repo authored on Windows, built
  in Linux Docker), `.gitignore`, `.env.example`, `backend/pyproject.toml` (Python 3.13,
  core deps + `[inference]` + `[dev]` extras, ruff/mypy/pytest config).
- **App core:** `app/config.py` (pydantic-settings, `QIE_` prefix), `app/db.py` (sync
  PostgreSQL engine/session), `app/logging_utils.py` (elapsed-time `phase()` helper, flushed
  output per the project convention), `app/security.py` (Fernet at-rest encryption for
  integration keys), `app/deps.py` (request-scoped principal/storage/queue/session deps),
  `app/main.py` (app factory + lifespan with startup-phase logging + **defensive router
  discovery** so the app boots while routers are still being built).
- **Models (DESIGN §6):** `asset`, `prompt`/`prompt_image`/`prompt_lora`, `lora`,
  `job`/`job_input`/`job_output`, `integration`, `setting` — all with nullable
  `owner_id`/`workspace_id` and storage-key addressing.
- **Seams (DESIGN §4.1a):** `StorageProvider` + `LocalFsStorage` (traversal-guarded keys),
  `AuthProvider` + `NoAuthProvider` (implicit local owner), `JobQueue` +
  `InProcessJobQueue` (single worker thread → one inference at a time, cooperative cancel).
- **Migrations:** Alembic scaffold + initial autogenerated migration; verified it applies
  and **re-applies as a no-op** (idempotent/resumable) against live Postgres — 11 tables.
- **Containers:** `docker-compose.yml` (`db` Postgres 16, `backend` CUDA placeholder,
  `backend-dev` lightweight 3.13 test image), `backend/Dockerfile.dev`.
- **Tests:** 6 foundation unit tests (imports/model-registration, health, no-auth principal,
  storage round-trip + traversal guard, Fernet round-trip + mask, in-process queue
  run/cancel). **All green inside the dev container against Postgres. ruff clean.**

### Verification evidence
- `docker compose --profile test run --rm backend-dev pytest` → **6 passed**.
- `ruff check .` → **All checks passed**.
- `alembic upgrade head` twice → second run no-op; `\dt` shows all 10 app tables +
  `alembic_version`.

### State / resumability
- **Done:** §12 step 1 foundation, committed (`082b43d`) + pushed.
- **In progress:** none.
- **Next:** fan out a parallel agent swarm across disjoint subsystems (see below).
- **Reconcile on resume:** `db` container (`qwenimageedit-db-1`) is running with the schema
  migrated; volume `qwenimageedit_pgdata` persists it. Safe to re-run any test command. No
  model downloads yet. To resume the build, check `git log` on branch `DEVRUN_202606070903`
  and the task list, then continue the fan-out.

### Decision Log — D1: sync DB layer (not async)
- **Context:** SQLModel/FastAPI support sync or async DB access; had to pick one for the
  foundation since all services build on it.
- **Options:** (a) async SQLAlchemy + asyncpg; (b) sync SQLAlchemy + psycopg3.
- **Research:** v1 is single-node, single-accelerator (DESIGN §4.1/§9.3). Request handlers
  do light CRUD (FastAPI runs sync endpoints in a threadpool); the heavy inference runs on a
  dedicated worker thread via the JobQueue, not in request handlers.
- **Decision:** **sync** SQLAlchemy + `psycopg[binary]`. Simpler, fewer foot-guns, no async
  contagion through the service layer; performance is irrelevant at this scale.
- **Outcome:** Foundation + tests green. Revisit only if a future cloud/multi-worker mode
  needs async — the stateless-handler design keeps that migration localized.

### Decision Log — D2: two Docker images (lightweight dev + full CUDA)
- **Context:** Host Python is 3.9; backend targets 3.13. Building the multi-GB CUDA image
  for every unit-test run would make the inner loop unbearably slow.
- **Options:** (a) install Python 3.13 on host; (b) only the CUDA image; (c) a lightweight
  `python:3.13-slim` dev image for the API/unit layer + the full CUDA image for GPU work.
- **Decision:** **(c)**. `backend/Dockerfile.dev` (no torch) runs the fast unit suite; the
  CUDA image (built in the Docker phase) runs the real GPU smoke tests. Per RUN §5 the
  *final* acceptance still runs unit + integration inside the CUDA image for container
  parity — the dev image is purely a fast inner-loop convenience.
- **Outcome:** Unit loop ~1–3s per run; foundation validated without a CUDA download.

### Orchestration — first parallel wave (launching now)
Six agents on **disjoint files** (each owns its service module(s) + router + own test file;
none edit shared foundation files, `conftest.py`, or `pyproject.toml`):
- **A** — `device_manager` + `model_advisor` (§9) + `resolution` (§5.4) + routers.
- **B** — `asset_store` (uploads/outputs/library/promote, HEIC/EXIF, dedup, thumbs) + router.
- **C** — `prompt_store` (CRUD, slots, bindings, slot resolution) + router.
- **D** — `lora_manager` + `integrations` (CivitAI `.com/.red/.green` normalization,
  encrypted keys) + routers.
- **E** — `pipeline_service` (Generate/Edit load+run, precision paths, LoRA stacking,
  progress callback, latent-preview decode) + mocked unit tests (real GPU later).
- **F** — frontend SPA scaffold (React/Vite/TS/Tailwind v4/shadcn) against the §7 API.
The integrative layer (jobs router + worker + WebSocket, batch/sweep, reproducibility
sidecar, prompt enhancer, round-out) I build myself afterward, once the real service APIs
exist, then run the full suite + Docker + GPU smoke tests.

---

## Entry 3 — Wave 1 backend subsystems landed (5 of 6 agents)

- **Local time:** 2026-06-07 10:05 PDT
- **Commit:** `e50cc36` (backend subsystems). Frontend agent (F) still running in background.

### What I did / outcome
Launched 5 backend agents in parallel (A–E) on disjoint files; all returned green and the
combined suite integrates cleanly. Frontend agent (F) launched in the background (still
working). Landed commit `e50cc36`:
- **A** device_manager + model_advisor + resolution (+ routers): 31 unit tests. Verified the
  spec example `1024/landscape/3:2 -> 1024×688`; advisor filters int4 to CUDA-Ampere, flags
  fp8 "emulated on SM<8.9", MPS→bf16-only.
- **B** asset_store (+ router): 12 tests. EXIF transpose + RGB normalize + HEIC opener,
  sha256 dedup, WebP thumbnails, PNG metadata embed + JSON sidecar.
- **C** prompt_store (+ router): 17 tests. CRUD/duplicate/atomic bindings + `resolve_inputs`
  (pinned+slot ordering, min/max validation). Found+fixed 2 real bugs (ARRAY `@>` filter;
  child-delete flush order).
- **D** lora_manager + civitai + integrations (+ routers): 52 tests. CivitAI `.com/.red/.green`
  normalization across URL/query/bare-id forms; Fernet-encrypted keys never echoed.
- **E** pipeline_service + reproducibility: 23 pure tests; module imports without torch; GPU
  tests skip cleanly. Lazy torch/diffusers; bf16/fp8(quanto)/int4(nunchaku) paths; LoRA
  stacking; throttled latent→preview; cooperative cancel.
- **main.py** auto-includes all 8 routers (verified in startup logs); jobs + rewriter
  correctly still pending.

### Verification evidence
- `pytest` (full, single DB) → **exit 0**: 140 passed, 1 skipped (`test_pipeline_gpu`, no GPU
  in dev image). `ruff check .` → clean.

### Decision Log — D3: per-agent test databases + fast test isolation
- **Context:** 6 agents running tests concurrently against one Postgres collided on
  `create_all`/`drop_all`; and that per-test DDL cost ~4.5s/test (Alpine PG fsyncs each of
  ~40 CREATE/INDEX statements), making the full suite look "hung" behind a buffered pipe.
- **Options:** (a) keep per-test create/drop (too slow + collision-prone); (b) per-agent DBs
  + transaction-rollback isolation; (c) per-agent DBs + per-test `TRUNCATE`.
- **Decision:** created DBs `qie_a..qie_e` (one per agent, via `QIE_DATABASE_URL` override in
  the test command) to remove cross-agent collisions, and rewrote the shared `session`
  fixture to **create the schema once per session** and isolate each test with a single fast
  `TRUNCATE ... RESTART IDENTITY CASCADE`.
- **Outcome:** DB test setup dropped from ~4.5s to ~0.05s; full suite runs in seconds; exit 0.

### State / resumability
- **Done:** wave-1 backend subsystems (`e50cc36`), all green + lint-clean.
- **In progress:** frontend agent F (background) building `frontend/` (Vite/React/TS/Tailwind
  v4/shadcn) — its files are NOT yet committed (excluded from `e50cc36` until it finishes).
  I am now hand-building the **jobs integration layer** (job_service + ProgressHub + jobs
  router + WebSocket + batch/sweep) — `app/schemas/jobs.py` already written.
- **Next:** finish jobs subsystem (task #4 integration), then prompt enhancer + round-out
  (task #5), wire/commit the frontend when F lands, then Docker CUDA image + real GPU
  Generate→Edit smoke tests (task #7).
- **Reconcile on resume:** `db` container up; DBs `qie`,`qie_a..e` exist. Models not yet
  downloaded. If F's frontend files are present but uncommitted, that is expected.

---

## Entry 4 — Jobs integration layer + frontend landed

- **Local time:** 2026-06-07 10:55 PDT
- **Commits:** `b432be0` (jobs subsystem + vendored prompt templates), `58f53bd` (frontend).

### What I did / outcome
Hand-built the integrative jobs layer (the piece that wires all the wave-1 services into a
real run) and landed the frontend the background agent produced.

**Jobs subsystem (`b432be0`)** — `job_service` + `progress_hub` + `routers/jobs`:
- Submit-time resolution: ordered Edit inputs from prompt slots (pinned + slot fills) or
  explicit ids, concrete `W×H` (incl. match-source), model/precision/device, LoRA
  validation → persists a `Job` + ordered `JobInput` rows (with input hashes).
- `run_job` on the single-accelerator worker: loads/reuses the pipeline, streams throttled
  progress + base64 live latent previews over the `ProgressHub`, saves each output through
  `asset_store` with full §5.5 reproducibility metadata embedded, persists status
  transitions, handles cancel + errors without crashing the worker.
- `recover_jobs` re-enqueues `queued`/`running` jobs on startup (resumability, RUN §7).
- REST: `POST /api/jobs`, `/api/jobs/batch`, `GET /api/jobs[/{id}]`, cancel, output
  file/thumb streaming. WebSocket `/ws/jobs/{id}` (top-level alias wired in `main.py`) sends
  a snapshot on connect then live updates until terminal.
- Batch/sweep (§5.6): slot / seed / param expansion under a shared `batch_id`.
- 8 new tests (resolution, slot ordering, batch expansion, queued cancel, hub, WS snapshot).

**Vendored templates** — copied the official `prompt_utils.py` + `prompt_utils_2512.py`
**verbatim** (pinned commit `3453042`, 2025-12-23) under `app/services/vendor/` with a
`NOTICE.md`; excluded from ruff/mypy. Ready for the §5.8 enhancer (which will extract the
`SYSTEM_PROMPT`/`EDIT_SYSTEM_PROMPT` strings and run them through a local Qwen-VL model).

**Frontend (`58f53bd`)** — Vite 6 / React 19 / TS strict / Tailwind v4 / shadcn / Framer
Motion / TanStack Query + Zustand. Composer (the full core loop incl. drag-to-reorder, live
advisor, resolution chip, enhance diff, WS progress + live preview, result gallery + compare
slider), libraries, history, settings/integrations, ⌘K palette, dark-first theming. Stubs
noted: slot/binding visual editor, default-model settings panel, slot/param sweep UI.
`npm run build` green (dist ≈ 897 KB code-split).

### Verification evidence
- Full backend suite (single DB): **148 passed, 1 skipped** (`test_pipeline_gpu`), `ruff
  check .` clean. main.py auto-includes all 9 routers + mounts `/ws/jobs/{id}` (verified in
  startup logs). Frontend `npm run build` green.

### State / resumability
- **Done (committed + pushed up to here):** foundation, wave-1 backend subsystems, jobs
  integration, frontend. Tasks #1–#4, #6 complete.
- **Next:** (#5) prompt enhancer (Qwen-VL, using the vendored templates) + round-out
  features (rembg background removal, upscale/refine, recipes, catalog export) — launching an
  agent; and (#7) CUDA Docker image + real Generate→Edit GPU smoke tests — building the
  Dockerfile and kicking off the (long) image build in parallel.
- **Reconcile on resume:** containers `qwenimageedit-db-1` up; no model weights downloaded
  yet (the first GPU run will pull Qwen-Image / Qwen-Image-Edit-2511 into the `/models`
  volume — tens of GB). The CUDA image build, once started, is long-running.

---

## Entry 5 — Prompt enhancer (§5.8) + round-out features (§13); CUDA image building

- **Local time:** 2026-06-07 11:35 PDT
- **Commit:** `b526919` (enhancer + round-out + recipe model/migration + smoke script).

### What I did / outcome
Two parallel agents (disjoint files; router slots pre-reserved by me) completed §5:
- **Prompt enhancer (§5.8):** `rewriter_templates` extracts the official
  `SYSTEM_PROMPT`/`EDIT_SYSTEM_PROMPT` **verbatim** from the vendored `prompt_utils*.py` via
  `ast` (single source of truth — nothing retyped). `rewriter` service: config-driven
  Qwen3-VL registry (30B-A3B AWQ/GGUF, 8B fp8/GGUF), backend-filtered, co-reside-vs-swap
  advisor (mirrors §9), genuinely multimodal `enhance` (Edit passes the input images to the
  VL model, order preserved). `/api/rewriter` {models, advise, enhance}. 29 tests.
- **Round-out (§13):** background removal (rembg, lazy) → ephemeral asset; upscale (Pillow
  Lanczos baseline + documented model-refine hook); recipes (new `Recipe` model + migration
  `45a23269`, CRUD + instantiate); catalog/batch export (zip = images + contact-sheet grid +
  CSV/JSON params). `/api/tools`, `/api/recipes`, `/api/export`. 26 tests.
- Myself: pre-reserved the 3 router slots, added the `Recipe` model + autogenerated its
  migration on a fresh DB (clean single-table diff), vendored the official templates earlier,
  and wrote `scripts/smoke_generate_edit.py` — the real job-flow Generate→Edit loop for the
  §6 acceptance bar (drives `job_service`→`pipeline_service`→`asset_store`, exports outputs +
  metadata to a mounted dir for committing to `images/`).

### Decision Log — D4: verbatim template extraction via `ast`; 2512 edit fallback
- **Context:** CLAUDE.md requires reusing the official rewriting templates verbatim; the
  vendored files embed them as locals inside functions (not importable constants), and
  `prompt_utils_2512.py` ships only the Generate-mode `SYSTEM_PROMPT` (no edit template).
- **Decision:** parse the vendored source with `ast` and pull the exact string literals
  (keeps the vendored file the single source of truth, guaranteeing verbatim). For
  `get_edit_system_prompt(variant="2512")`, fall back to the default file's verbatim
  `EDIT_SYSTEM_PROMPT` (the 2512 file has none) — covered by a test.
- **Outcome:** template tests round-trip against the files; nothing hand-copied.

### Verification evidence
- Full backend suite (single DB, all 12 routers loaded + `/ws/jobs` mounted): **exit 0**,
  ruff clean. 2 GPU test modules skip cleanly without torch (`test_pipeline_gpu`,
  `test_rewriter_gpu`). Frontend build green (committed earlier).

### State / resumability
- **Done:** tasks #1–#6 complete; §5 landed (`b526919`). All non-GPU code is implemented,
  integrated, unit-tested, and lint-clean.
- **In progress (#7):** CUDA backend image building in the background (multi-stage: node
  builds the SPA → CUDA runtime installs Python 3.13 + torch cu126 + diffusers stack +
  best-effort nunchaku). At this diary entry it is downloading the torch CUDA wheels.
- **Next:** once the image is built — run the real GPU smoke tests inside it on the A6000
  (the `@pytest.mark.gpu` suites + `scripts/smoke_generate_edit.py`), pulling the real
  Qwen-Image / Qwen-Image-Edit-2511 weights; commit output images to `images/`; exercise the
  advisor across bf16/fp8(emulated)/int4; then final polish + docs sync.
- **Reconcile on resume:** if interrupted, re-run `docker compose --profile full build
  backend` (layer cache resumes). The dev-image unit suite is the fast regression gate:
  `docker compose --profile test run --rm backend-dev pytest`.

---

## Entry 6 — Resume after power outage

- **Local time:** 2026-06-07 11:49 PDT
- **Commit at resume:** `b29ccf0` (HEAD) + uncommitted local edits (see below).

### Context
A power outage interrupted the run during the **real GPU smoke test** (task #7). This entry
records the reconciled state and the two screen-caps the operator took while relaunching the
agent (`claude --dangerously-skip-permissions --continue` from `H:\QwenImageEdit`).

### Resume screen-caps
![Anaconda Prompt relaunching the agent to continue the run](images/Screenshot%202026-06-07%20114336.png)
*Relaunching: `cd QwenImageEdit` → `claude --dangerously-skip-permissions --continue` (note
~4.19 TB free on H:, ample for the model weights).*

![Claude Code TUI at resume, showing the pre-outage smoke-run status and the resume prompt](images/Screenshot%202026-06-07%20114849.png)
*The session resuming: the last pre-outage status (advisor recommended **fp8** at 1024 on the
48 GB A6000; bf16+offload smoke run in flight, Qwen weights downloading) and the operator's
power-outage resume prompt.*

### Reconciled state (what survived / what was interrupted)
- **Code:** all committed through `b29ccf0`. Uncommitted local edits that survived the
  outage and are still pending (part of in-progress task #7): `CLAUDE.md` (populated the
  Commands section) and `backend/scripts/smoke_generate_edit.py` (added the `--offload`
  flag). These will land with the task-#7 completion commit.
- **Models:** `./models` holds **~33 GB** of a partial Qwen-Image download (of ~100 GB for
  both bf16 models). HF caches are resumable (`.incomplete` blobs), so re-running continues
  rather than restarting.
- **Smoke run:** did **not** complete — no outputs in `./data` or `./images` (no
  `generate_*.png`/`edit_*.png`/`smoke_summary`). The bf16 Generate→Edit loop must be re-run.
- **Containers:** all stopped by the outage, including Postgres `db`. The `pgdata` volume and
  the migrated `qie` schema persist on disk.

### Resume plan (task #7, continued)
1. Restart the `db` service (`docker compose up -d db`); the `qie` schema is already migrated
   (initial + recipe) on the persistent `pgdata` volume.
2. Re-launch the real **Generate→Edit** smoke loop in the CUDA image **via PowerShell** (not
   Git-Bash — MSYS mangled `/models`→`C:/Program Files/Git/models` last time; fixed by using
   PowerShell so `-e HF_HOME=/models` and the `-v …:/models` mount resolve correctly). The
   partial download resumes from the `./models` cache.
3. Given the advisor recommends **fp8** at 1024 on this 48 GB card, run the acceptance loop at
   the recommended precision (and validate bf16 with offload + fp8-emulated-on-Ampere as the
   exercise across precisions; int4/Nunchaku is expected unavailable — the PyPI `nunchaku` is
   a placeholder — so the int4 guard test should report it cleanly).
4. Commit the resulting output images to `./images/` with their reproducibility metadata, then
   finish docs sync (CLAUDE.md/DESIGN.md) and the final diary entry.

---

## Entry 7 — Real GPU smoke test PASSED (fp8 Generate→Edit on the A6000)

- **Local time:** 2026-06-07 14:45 PDT
- **Commit:** task-#7 landing (this entry committed alongside the code + output images).

### What I set out to do
Complete the §6 acceptance bar: run the real, self-contained **Generate → Edit** loop on the
A6000 inside Docker, recording full reproducibility metadata, and commit the output images.

### What happened (and the two failures I fixed first)
1. **MSYS path-mangling (resume):** the first `docker run` via Git-Bash rewrote `-e
   HF_HOME=/models` → `C:/Program Files/Git/models`, so weights downloaded into the
   container's ephemeral layer. **Fix:** launch the container via **PowerShell** (no MSYS
   conversion). Weights now persist to the mounted `./models`.
2. **fp8 needed a compiler:** optimum-quanto JIT-compiles a CUDA kernel (`quanto_cuda`) at
   fp8-quantization time, but the CUDA **runtime** base image had no `nvcc`/headers →
   `nvcc: not found`. **Fix:** rebuilt the image on the CUDA **devel** base (+ `ninja`,
   `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST=8.6`). fp8 then compiled and ran.
3. **Offload bug:** the service called `.to(cuda)` *then* `enable_model_cpu_offload()`,
   double-placing ~40 GB and getting the bf16+offload run OOM-killed. **Fix:** defer device
   placement — only `.to(device)` when offload is *not* requested; otherwise let accelerate
   manage it. (Verified clean; 23 pure pipeline tests still pass.)

### Result — PASSED ✅
The advisor recommended **fp8** at 1024² on the 48 GB A6000 (bf16 headroom is tight, as
DESIGN §9 predicts), and the loop ran the advisor's recommended precision end-to-end:

- **Generate** (`Qwen/Qwen-Image`, fp8, 30 steps, 1024²): quantize ~16 min (one-time, kernel
  build + 20B params), **inference 92.5 s**, output saved via the real job flow
  (`job_service`→`pipeline_service`→`asset_store`) with a DB `Job`+`JobOutput` and a PNG
  metadata sidecar.
- **Edit** (`Qwen/Qwen-Image-Edit-2511`, fp8, 30 steps): the **generated image was fed back
  in** as the ordered input; **inference 194 s** (~6.2 s/step); output saved. The edit
  metadata records `input_hashes: ["f5120501…"]` — the sha256 of the generated apple,
  proving the chained loop.
- Total wall-clock ~82 min (dominated by model download + the slow H: disk load + the
  emulated-fp8 quantization passes). `SMOKE OK`.

### Output evidence (committed under `images/`)
![Generate output: a red apple on a white studio background (Qwen-Image, fp8, 1024², seed 12345)](images/generate_20260607_204612.png)
*Generate — "a single ripe red apple on a plain white studio table…" (fp8, 30 steps, seed 12345).*

![Edit output: the same apple recomposited onto a warm wooden surface in morning light (Qwen-Image-Edit-2511, fp8)](images/edit_20260607_213927.png)
*Edit — the generated apple placed "on a rustic wooden cutting board in a cozy kitchen, warm
morning light" (fp8, 30 steps, seed 777; input = the generate output).*

Reproducibility sidecars (`generate_*.json`, `edit_*.json`, `smoke_summary_*.json`) capture
the full DESIGN §5.5 metadata: mode, model id/revision, precision, device (incl. SM 8.6),
both prompts, negative, seed, LoRAs, resolution, sampler, input hashes, rewriter, format.

### Precision exercise (RUN §5)
- **fp8** — fully validated (the run above); emulated on SM 8.6 (no speedup), runs correctly.
- **bf16** — loads + runs; tight at 1024² (advisor steers to fp8); the offload path is now
  correct for headroom-limited cards/larger latents.
- **int4 (Nunchaku)** — unavailable in this image (the PyPI `nunchaku` is a placeholder; no
  matching SVDQuant wheel for torch 2.12/cp313/cu126). The pipeline degrades gracefully with
  a clear error; documented in DESIGN §9.4 as the remaining step to enable int4.

### Verification evidence
- Real in-container Generate→Edit loop: **PASSED** (`SMOKE OK`, images + metadata committed).
- Full unit suite after the fixes (ruff + ~203 tests): **exit 0** (1 gpu test skipped in the
  no-torch dev image). pipeline-service edits regression-clean.

### State / resumability
- **Done:** all of §12 (#1–#11) plus the round-out features; the real GPU smoke loop passed.
  Tasks #1–#7 complete.
- **Next:** final polish/docs sync + the §9 definition-of-done check, then the closing commit.
- **Reconcile on resume:** the CUDA **devel** image (`qwenimageedit-backend`) is the current
  build; models are cached in `./models` (Generate + Edit, ~tens of GB) so re-runs skip the
  download. Re-run the smoke loop via PowerShell (not Git-Bash) with
  `python -m scripts.smoke_generate_edit --precision fp8 --steps 30`.
