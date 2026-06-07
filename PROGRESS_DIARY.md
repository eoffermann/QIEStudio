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
