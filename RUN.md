# RUN.md — Autonomous Build Runbook

This document instructs an autonomous Claude Code instance that will **build QIE Studio
completely unattended** — there will be no human observation, input, or course-correction
once the run begins. Read this file fully, then read `DESIGN.md` (the authoritative spec)
and `CLAUDE.md` (locked stack + constraints) before doing anything.

You have **full permission** to create and update Docker images, create and destroy
containers, install packages, and download/quantize models as needed to get the job done.

---

## 1. Operating principles

- **You are unattended.** Nobody will answer questions or unblock you mid-run. When you hit
  something the spec doesn't cover, make a reasonable, well-researched decision, **proceed**,
  and disclose it in the Progress Diary (§7). Never stop to wait for input.
- **Work until complete.** Continue through the full v1 scope (§4) until it is implemented
  and passes the acceptance bar (§6), or until genuinely and irrecoverably blocked — in
  which case document the blocker exhaustively in the diary before stopping.
- **`DESIGN.md` is the contract.** Build what it specifies, in the order it specifies
  (§12 build order). If the spec is wrong, ambiguous, or infeasible, update `DESIGN.md`
  (and `CLAUDE.md` if a locked decision changes) in the same commit as the code, and log
  why in the diary.
- **Bias to working software.** Prefer a running, tested vertical slice over broad
  half-built scaffolding. Every committed piece should build and pass its checks.
- **Honest reporting.** If something is stubbed, skipped, or failing, say so plainly — never
  present incomplete work as complete.

---

## 2. Build environment (what you can assume)

- **Network access:** yes — install OS/Python/Node packages and download model weights and
  LoRAs from Hugging Face / CivitAI.
- **GPU:** an **NVIDIA RTX A6000 (48 GB GDDR6, Ampere, compute capability 8.6)** with CUDA
  installed and available. You can and should run the real Qwen-Image / Qwen-Image-Edit
  pipelines and the Qwen-VL rewriter here.
- **Models & quantization:** download the required models. If a suitable prebuilt quant
  isn't available online, **quantize it yourself**. Hardware notes that affect precision
  choices:
  - 48 GB comfortably fits bf16 for a single ~20B pipeline, but headroom tightens once a VL
    rewriter, LoRAs, larger latents, or batches are added — exercise the model advisor.
  - The A6000 is **Ampere (SM 8.6)**: native fp8_e4m3fn acceleration needs SM ≥ 8.9
    (Ada/Hopper), so fp8 here is effectively emulated/cast — validate it runs, but don't
    expect a speedup. Nunchaku SVDQuant int4 supports Ampere; prefer it for the low-VRAM
    path.
- **Containers:** build and run the app via **Docker** (the CUDA backend image + the
  Postgres service per `DESIGN.md` §8). Do real builds and real runs — don't just author
  Dockerfiles.

Probe and record actual capabilities (driver/CUDA version, free VRAM, Docker/Compose
versions) in the first diary entry so the run is reproducible.

---

## 3. Startup sequence

When told to begin:

1. **Create and switch to a build branch** in the current repo, named
   `DEVRUN_YYYYMMDDHHMM`, where `YYYYMMDDHHMM` is the **local** datestamp at the moment the
   run starts (e.g. `DEVRUN_202606071430`). (If the directory is not yet a git repo,
   initialize it first.)
2. **Initialize `PROGRESS_DIARY.md`** (§7) with a top heading naming the project, the start
   date/time (local), and the build-environment facts probed in §2.
3. **Probe the environment** (network, GPU/CUDA, Docker) and record results in the first
   diary entry before writing code.

> **Remote/push:** the cadence in §8 assumes a configured remote. If none is configured,
> treat "push" as a no-op, note it in the diary, and keep committing locally.

---

## 4. Development workflow & scope

- **Target: full v1**, following the **`DESIGN.md` §12 build order**:
  backend skeleton (FastAPI + PostgreSQL + Alembic + storage/auth/queue interfaces) →
  device manager + model advisor → pipeline service (Generate + Edit) → resolution service →
  asset store → job queue + WebSocket progress + live latent previews → LoRA manager +
  integrations → prompt store → frontend → round-out features (prompt enhancer, background
  removal, upscale/refine, recipes, catalog export) → Docker/Compose → polish.
- **Work in coherent vertical slices.** Get each subsystem importable, wired, and
  test-covered before moving on.
- **Honor the locked constraints in `CLAUDE.md`** (Python 3.13, PostgreSQL,
  storage-keys-not-paths, `owner_id` scoping, single-GPU queue, verbatim official
  `prompt_utils*` templates, CivitAI domain normalization, etc.).

---

## 5. Testing strategy

Tests grow in sophistication alongside the code, and earlier tests are kept green.

- **Early development:** simple **unit tests** verifying individual classes, methods, and
  modules (e.g. resolution math, advisor logic, prompt-slot resolution, storage/auth/queue
  interfaces, API request/response shapes).
- **As functionality lands:** progress to **integration tests that exercise the real
  pipelines on the A6000**. The canonical pattern: **use the Generate pipeline to create
  images, then feed those generated images into the Edit pipeline** — a self-contained
  end-to-end loop that needs no external fixtures. Document each such test and run it.
  Include image outputs in an `images` directory in the project, committing them to git,
  and inserting them in the progress diary along with relevant information about the test.
  As support is implemented, include at least one path **with a LoRA applied** and one
  through the **Qwen-VL prompt enhancer**, and exercise the model advisor across **bf16 / int4**
  (and fp8, validating it runs on Ampere even if emulated).
- **Regression discipline:** at every major development junction, **re-run all earlier
  tests** to catch regressions before moving on.
- **Container parity:** the test suites — including the real smoke tests — must pass
  **inside Docker** (CUDA backend image + Postgres), not only on the host.

Record test commands, timings, and output evidence (paths/thumbnails) in the diary.

---

## 6. Acceptance bar ("done" for a slice / for v1)

A piece of work is done only when it is **demonstrably running**, not merely written:

- Linter clean; type-checks pass (backend type hints + TypeScript); backend imports
  cleanly; frontend builds.
- Relevant unit + integration tests pass, and earlier tests still pass (§5).
- The real **Generate → Edit** loop runs end-to-end on the A6000, inside Docker.
- A completed job records the full reproducibility metadata from `DESIGN.md` §5.5 (mode,
  model id/revision, precision, device, seed, LoRAs, input hashes, original + enhanced
  prompts).

v1 is complete when the entire §12 scope meets this bar — see §9.

---

## 7. Progress Diary (`PROGRESS_DIARY.md`)

The diary is the primary record of an otherwise unobserved run, and it is the **resume
point** if the process is interrupted.

- **Every entry includes** the **local date and time** and the **current commit id**.
- **Log each meaningful step:** what you set out to do, what you did, and the outcome
  (including failures and how you resolved them).
- **Resumability / idempotency.** Write the diary so that *you, another agent, or a human*
  could pick up exactly where you left off after a crash, power loss, or interruption.
  Each entry should make the current state unambiguous: what is done, what is in progress,
  what is next, and any state that must be reconciled (e.g. partially downloaded models,
  half-applied migrations, running containers). **Re-running the process after an
  interruption must be safe** — design steps to be idempotent (check-before-create,
  resumable downloads, migrations that no-op if already applied) and note in the diary how
  to safely resume.
- **Decision Log entries (required for any original decision)** — whenever you decide
  something not explicitly stated or clearly implied by `DESIGN.md`, add a clearly marked
  decision entry containing:
  - **Context** — what forced a decision.
  - **Options considered** — the alternatives you weighed.
  - **Research** — resources/docs/experiments consulted (with links where applicable).
  - **Decision** — what you chose, and the reasoning.
  - **Outcome** — what happened after applying it (update later if it changes).

---

## 8. Commit & push cadence

- **Commit locally often** during development — frequent small commits make it easy to roll
  back when a test fails.
- **Commit and push** when a feature or coherent piece of code is ready to land, with a
  meaningful message describing what was done and why. Follow each landed piece with a
  Progress Diary entry.
- **Whenever the diary is updated**, commit it on its own with a message like
  `[DIARY] <brief description of the entry>` and **push it immediately**.
- Keep history readable: small, self-contained, building commits over large mixed ones.

---

## 9. Definition of done

The run is complete when **all** hold:

1. The full v1 scope from `DESIGN.md` §12 is implemented.
2. The acceptance bar (§6) passes across the system — including the real, in-container
   Generate → Edit smoke tests on the A6000, with earlier tests still green (§5).
3. `DESIGN.md` and `CLAUDE.md` accurately reflect the delivered system, including any
   decisions or changes made during the run.
4. `PROGRESS_DIARY.md` is complete and resumable — all steps, decision logs, and test
   evidence recorded — and the final commit + diary entry are pushed per §8.
