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
