# Qwen Image Edit Studio — Design Specification

> **Status:** v1.0 — finalized spec (three rounds of decisions). Ready to scaffold.
> **Working title:** *QIE Studio* (name TBD — see open questions).
> **Goal:** A self-hosted, Docker-deployable image **generation & editing** application
> built around the **Qwen-Image** family (Qwen-Image-Edit-2511 for editing, Qwen-Image for
> text-to-image generation), with a modern 2026-era web UI, LoRA management, reusable
> prompt and image libraries, flexible output-resolution control, and VRAM-aware model
> selection.

---

## 1. Overview

QIE Studio is a single-node application that wraps the Qwen-Image-Edit-2511 diffusion
model in an ergonomic, attractive editing workflow. It is aimed at power users and
small teams who want repeatable, library-driven image editing — e.g. compositing
products into staged scenes, multi-image fusion, character-consistent edits, and
instruction-based image manipulation.

The product offers two run **modes** that share the same libraries, UI, and output flow:

- **Edit mode** — Qwen-Image-Edit-2511, with one or more input images + an instruction.
- **Generate mode** — Qwen-Image text-to-image, no input image required; outputs can be
  downloaded or promoted into the library (and then used as inputs to Edit mode).

The product is organized around four reusable building blocks that combine at run time:

1. **Images** — ephemeral per-run uploads, generated outputs, *and* a persistent, managed
   asset library. Both uploads and outputs can be **promoted to the library**.
2. **Prompts** — free-text plus a savable, named prompt library, optionally bound to
   specific LoRAs and library images (including reusable image "slots").
3. **LoRAs** — an installable, selectable registry of LoRA adapters with weights,
   discoverable from local upload, Hugging Face, CivitAI, or direct URL.
4. **Resolution presets** — guided output sizing (base size × orientation × aspect ratio,
   or "match source").

A run is simply: *pick a mode + (images for Edit) + a prompt + (optional) LoRAs + a
resolution* → generate.

### Decisions locked (rounds 1–2)

| Decision | Choice |
|----------|--------|
| Modes | **Generate** (Qwen-Image text-to-image) **and Edit** (Qwen-Image-Edit-2511). |
| Model versions | Sensible **defaults**, but **user-selectable** at run time; newer *and* older versions supported (some LoRAs need a specific base). |
| GPUs | **Single-GPU in v1**; multi-GPU scheduling is future work. |
| Live previews | **Live latent previews during generation** are in scope for v1. |
| Target hardware | High-VRAM reference (40GB+), but **multi-backend**: NVIDIA/CUDA, AMD/ROCm, Apple Silicon (MPS). |
| Precision options | **bf16**, **fp8_e4m3fn (scaled)**, and **Nunchaku SVDQuant int4** — chosen via a VRAM-aware advisor (§9). |
| UI delivery | **Browser web app** served by the backend; React/Tailwind/shadcn stack confirmed. |
| Resolution semantics | **Base size = longer edge**; short edge derived from aspect ratio, snapped to ×16. |
| Deployment | **Local workstation only for v1**, but architecture kept **cloud- and multi-user-ready** (pluggable storage/auth/queue, workspace scoping in the schema). |
| Users / auth | **Single-user, no auth in v1**; auth is a pluggable provider so accounts can be added later without a schema rewrite. |
| Metadata store | **PostgreSQL** from the start. |
| LoRA sources | **Local upload + Hugging Face + CivitAI + direct URL**, with an **Integrations** settings page for API keys. No size/license caps. |
| CivitAI domains | Accept `civitai.com` / `.red` / `.green` URLs — they're filtered views over one DB sharing the `civitai.com/api/v1` API; imports normalize to the same model/version id. |
| Editing scope | **Whole-image instruction editing** for v1 (no masking/regional edits yet). |
| Image formats | In: **PNG, JPEG, WebP, GIF** (+ optional **HEIC** for iPhone/iPad). Out: PNG/WebP/JPEG. |
| v1 round-out features | **Committed:** background removal, upscale/refine, recipes, catalog export, and a **Qwen-VL prompt enhancer** (§5.8). |

---

## 2. The models

The app drives two members of the Qwen-Image family through diffusers. They share the
same VAE/text-encoder architecture, so precision/quantization strategy and the LoRA stack
apply to both.

| Mode | Model id | Pipeline | Input |
|------|----------|----------|-------|
| **Edit** | `Qwen/Qwen-Image-Edit-2511` | `QwenImageEditPlusPipeline` | ordered list of images + prompt |
| **Generate** | `Qwen/Qwen-Image` | `QwenImagePipeline` | prompt only (text-to-image) |

**Model versions are first-class.** Each mode ships a sensible **default** model id but
exposes a **model picker** so the user can choose a different version at run time — both
newer releases *and* older ones (a given LoRA may require, or look best on, a specific
base). Defaults and the list of known versions are config-driven; arbitrary HF repo ids
can also be entered. The chosen model id + revision is recorded in job metadata for
reproducibility, and the model advisor (§9) re-evaluates VRAM per chosen version.

Defaults (overridable):
- **Edit:** `Qwen/Qwen-Image-Edit-2511`
- **Generate:** `Qwen/Qwen-Image` (newer refreshes such as a 2512 can be selected when
  available)

### 2.1 Edit model: Qwen-Image-Edit-2511

- **Pipeline:** `diffusers.QwenImageEditPlusPipeline`
- **Model id:** `Qwen/Qwen-Image-Edit-2511` (the `Plus` pipeline is the 25xx-series API)
- **Capabilities relevant to us:** multi-image input/fusion, improved character &
  multi-person consistency, reduced image drift, integrated popular LoRAs, stronger
  geometric reasoning and industrial-design generation.
- **Multi-image input:** images are passed as an ordered list to the `image=` argument.
  Order is semantically meaningful to the model, so the UI must preserve and expose
  ordering.
- **Reference inference parameters (from the model card):**
  - `num_inference_steps = 40`
  - `true_cfg_scale = 4.0`
  - `guidance_scale = 1.0`
  - `negative_prompt = " "` (a single space is the documented default)
  - `num_images_per_prompt = 1`
  - `torch_dtype = bfloat16`

### Minimal reference call (for grounding the pipeline service)

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16,
).to("cuda")

output = pipeline(
    image=[image1, image2],            # ordered list of PIL images
    prompt=prompt,
    negative_prompt=" ",
    num_inference_steps=40,
    true_cfg_scale=4.0,
    guidance_scale=1.0,
    num_images_per_prompt=1,
    generator=torch.manual_seed(seed),
)
output.images[0].save("output.png")
```

### 2.2 Generate model: Qwen-Image (text-to-image)

- **Pipeline:** `diffusers.QwenImagePipeline`
- **Model id:** `Qwen/Qwen-Image` (configurable; the generation base iterates over time).
- **Input:** prompt only — no input image. Outputs feed straight into the shared output
  flow (download, promote-to-library, send-to-Edit).

```python
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")
output = pipeline(
    prompt=prompt,
    negative_prompt=" ",
    width=w, height=h,
    num_inference_steps=steps,
    true_cfg_scale=4.0,
    num_images_per_prompt=batch,
    generator=torch.manual_seed(seed),
)
```

> **VRAM note:** Qwen-Image is a ~20B-parameter MMDiT. Full bf16 is heavy, and at 40GB+
> the DiT + text encoder + VAE leave limited headroom once LoRAs, larger latents, and
> batches are added. The model loader therefore offers selectable precision/quantization
> with VRAM-aware guidance — see §9.

---

## 3. Goals & non-goals

### Goals
- Clean, fast, modern web UI — explicitly **not** Gradio; a bespoke React/Tailwind UI.
- Two modes sharing one workflow: **Generate** (text-to-image) and **Edit** (multi-image).
- First-class **multi-image** editing with drag-to-reorder.
- **LoRA** install + multi-select with per-adapter weights, discoverable from upload, HF,
  CivitAI, and URL.
- **Prompt library** with naming, rename, duplicate, delete, tagging, and search.
- **Image library** (persistent assets) distinct from ephemeral per-run uploads, fed by
  both uploads and generated outputs via **promote-to-library**.
- **Prompt ↔ image binding**, including reusable "slots" (pinned background + open slots
  the user fills at run time).
- Guided **resolution presets**.
- **VRAM-aware model selection** across bf16 / fp8 / int4 with actionable advice.
- **Multi-backend** acceleration: NVIDIA/CUDA, AMD/ROCm, Apple Silicon (MPS).
- Reproducible **generation history** with full metadata.
- Containerized, GPU-accelerated, Python 3.13 backend.
- **Cloud- and multi-user-ready** architecture (pluggable storage/auth/queue, workspace
  scoping) even though v1 ships single-user/local.

### Non-goals (v1)
- Multi-tenant SaaS with billing (architecture stays open to it, but not built).
- Training/fine-tuning LoRAs (we *install and use* them, not train them).
- A full layered raster editor (no Photoshop-style layers/brushes).
- Masking / regional editing (deferred; whole-image instruction editing only in v1).
- Mobile-native apps (responsive web is in scope; native is not).

---

## 4. Architecture

A two-process design behind one container (or a small Compose stack):

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser (React SPA — modern 2026 UI)                              │
└───────────────┬───────────────────────────────┬──────────────────┘
                │ REST (JSON)                     │ WebSocket (job progress)
┌───────────────▼───────────────────────────────▼──────────────────┐
│  FastAPI backend (Python 3.13, Uvicorn)                           │
│   ├─ API routers: assets / prompts / loras / presets / jobs /     │
│   │              integrations / models                            │
│   ├─ Auth provider (no-op in v1 → token/OIDC later) [pluggable]   │
│   ├─ Job queue (in-process FIFO in v1 → Redis/RQ later) [iface]   │
│   ├─ Services:                                                     │
│   │    pipeline_service   (loads & runs Generate + Edit pipelines)│
│   │    device_manager     (CUDA / ROCm / MPS detection + caps)    │
│   │    model_advisor      (VRAM-aware precision recommendation)   │
│   │    lora_manager       (registry, install, HF/CivitAI search)  │
│   │    asset_store        (uploads + outputs + library, thumbs)   │
│   │    prompt_store       (CRUD, slots, bindings)                 │
│   │    resolution         (preset → concrete WxH)                 │
│   │    integrations       (encrypted API keys: HF, CivitAI)       │
│   ├─ Storage provider (local FS in v1 → S3/object store) [iface]  │
│   └─ Persistence: PostgreSQL (metadata) + storage provider (bins) │
└───────────────┬───────────────────────────────────────────────────┘
                │
   ┌────────────▼──────┐ ┌──────────┐ ┌────────────┐ ┌──────────────┐
   │ PostgreSQL        │ │ /data    │ │ /models    │ │ Accelerator  │
   │ (metadata)        │ │ assets/  │ │ HF hub     │ │ CUDA/ROCm/MPS│
   │                   │ │ outputs/ │ │ cache      │ │              │
   │                   │ │ loras/   │ └────────────┘ └──────────────┘
   └───────────────────┘ └──────────┘
```

### 4.1 Backend
- **Language/runtime:** Python 3.13.
- **Web framework:** FastAPI + Uvicorn. Pydantic v2 schemas.
- **Inference:** PyTorch (CUDA / ROCm / MPS builds), `diffusers`, `transformers`,
  `accelerate`, `safetensors`, `peft` (external LoRA loading), `Pillow`, plus
  `optimum-quanto`/`torchao` for fp8 and `nunchaku` for SVDQuant int4 (CUDA-only).
- **Persistence:** **PostgreSQL** via SQLModel/SQLAlchemy + Alembic migrations for
  metadata; binaries via the storage provider.
- **Job model:** a serialized async queue (one inference at a time per accelerator).
  v1 is an in-process FIFO behind a `JobQueue` interface so it can be swapped for
  Redis/RQ/Celery when scaling to cloud/multi-worker. Progress (per-step callback) is
  streamed over WebSocket; job status is persisted so jobs survive page reloads.

### 4.1a Future-proofing seams (cloud / multi-user)

v1 ships single-user and local, but these seams keep that door open without a rewrite:

- **`StorageProvider` interface** — `LocalFsStorage` now; `S3Storage`/object-store later.
  All binary access goes through it; the DB only stores keys, not absolute paths.
- **`AuthProvider` interface** — `NoAuthProvider` (a single implicit owner) now; token or
  OIDC providers later. Request context carries a `principal`.
- **Workspace/owner scoping in the schema** — every user-owned row carries a nullable
  `owner_id` / `workspace_id` (defaulted to the implicit single user in v1). Multi-user
  becomes a matter of populating and filtering on these, not migrating structure.
- **`JobQueue` interface** — in-process now; distributed broker later.
- **Stateless request handlers** — no reliance on in-process session state beyond the
  queue, so horizontal scaling stays viable.
- **Config-driven** — model ids, device, storage backend, DB URL, and integration keys all
  come from env/config, never hard-coded.

### 4.2 Frontend
- **Framework:** React (Vite SPA) + TypeScript, served as static assets by the backend.
- **Styling:** Tailwind CSS v4 + shadcn/ui (Radix primitives) for accessible components.
- **Motion:** Framer Motion for tasteful transitions.
- **State/data:** TanStack Query for server state; Zustand for local UI state.
- **Aesthetic (the "2026" look):**
  - Dark-first theme with optional light mode and a configurable accent.
  - Soft glass/elevated surfaces, generous spacing, rounded-2xl cards, subtle depth.
  - Fluid micro-interactions, skeleton loaders, optimistic UI.
  - Command palette (⌘K) for fast navigation/actions.
  - Drag-and-drop everywhere it helps (image upload, input reordering, library curation).
  - Keyboard shortcuts for the core loop (generate, reseed, send-to-input, save prompt).
- **Why not Gradio:** Gradio constrains layout and aesthetics; a bespoke SPA gives us the
  modern look, the slot/library UX, and the command-palette flow the brief calls for.

### 4.3 Multi-backend acceleration

A `device_manager` detects the available accelerator and reports its capabilities, which
feed the model advisor (§9):

| Backend | Detection | Notes |
|---------|-----------|-------|
| **NVIDIA / CUDA** | `torch.cuda.is_available()` + SM/compute capability | Full support incl. fp8 (Ada/Hopper, SM ≥ 8.9 for e4m3fn) and Nunchaku SVDQuant int4. |
| **AMD / ROCm** | `torch.version.hip` | bf16/fp16 + offloading; fp8 support depends on arch; **Nunchaku int4 is CUDA-only** so it's hidden on ROCm. |
| **Apple Silicon / MPS** | `torch.backends.mps.is_available()` | bf16/fp16 on `mps`; no CUDA-only quant paths; larger latents rely on unified memory. **Runs natively, not in Docker** (containers can't access the Mac GPU) — see §8. |

The advisor only offers precision options the detected backend actually supports, and
annotates each with backend-specific caveats.

---

## 5. Feature specifications

### 5.1 Images: ephemeral uploads + persistent library

Two scopes, clearly distinguished in the UI:

- **Run uploads (ephemeral):** dropped/added for the current composition only. Stored in a
  temp area, garbage-collected after the run (or after a TTL). This is the *default*.
- **Library assets (persistent):** explicitly saved images, kept indefinitely.

Three sources feed images into the app: **uploads** (ephemeral), **generated outputs**,
and **library assets** (persistent). Uploads and outputs both start ephemeral and can be
**promoted to the library**.

**Library asset features**
- Name, description, tags, and auto-generated thumbnail.
- Collections/folders for organization; search and tag filtering.
- **Promote-to-library** action — works on **both ephemeral uploads and generated
  outputs** — turning either into a permanent asset in one click (carrying along source
  metadata, e.g. the generation parameters for an output).
- Metadata: dimensions, format, file size, created/used timestamps, source
  (`upload` | `output` | `import`).
- Re-selectable from anywhere images are chosen (run composer, prompt bindings).
- De-dup by content hash to avoid storing the same image twice.

**Input handling**
- Supported input formats: **PNG, JPEG, WebP, GIF** (first frame for animated GIF), plus
  **HEIC/HEIF** (iPhone/iPad) via `pillow-heif`. TIFF as a nice-to-have.
- On import, images are normalized to RGB (handling palette/alpha/CMYK), EXIF orientation
  is auto-applied, and a working copy + thumbnail are generated.
- **Output formats:** PNG (default, lossless + embedded metadata), WebP, JPEG — selectable
  per run, with quality control for lossy formats.
- Drag-to-reorder the active input list (order matters to the model).
- Per-input thumbnail with remove/reorder/replace controls.

### 5.2 Prompts: library, slots, and bindings

**Prompt object**
```jsonc
{
  "id": "...",
  "name": "Furniture in living room",
  "text": "Produce a nicely lit composition that places the furniture in this living room.",
  "tags": ["staging", "furniture"],
  "loras": [ { "lora_id": "...", "weight": 0.8 } ],   // optional, see 5.3
  "images": [                                          // ordered, optional
    { "role": "pinned",  "asset_id": "livingroom-001" },
    { "role": "slot",    "slot_name": "furniture",
      "min": 1, "max": 3, "hint": "Upload the furniture piece(s)" }
  ],
  "defaults": {                                        // optional generation defaults
    "resolution_preset": "1024 / landscape / 3:2",
    "steps": 40, "true_cfg_scale": 4.0, "seed": null
  }
}
```

- **CRUD:** create, save-with-name, rename, **duplicate**, delete.
- **Organization:** tags + search; recently used; favorites.
- **LoRA binding (optional):** a saved prompt can carry its preferred LoRA(s) + weights,
  auto-applied when the prompt is loaded (user can override).
- **Image bindings (the key workflow):** an ordered image list where each entry is either:
  - **Pinned asset** — a specific library image always included (e.g. the living room).
  - **Open slot** — a named placeholder the user fills at run time (e.g. furniture).
    Slots declare min/max count and a hint string.
- This realizes the furniture example: keep the living-room photo pinned in the prompt,
  define a `furniture` slot, and at run time the user simply uploads chair/sofa/table
  images into the slot. The composed model input becomes
  `[livingroom, <uploaded furniture...>]` in the order defined by the prompt.
- **Template variables (optional, v1.1):** `{slot:furniture}` style tokens inside the
  prompt text, and simple variables (e.g. `{style}`) the user fills at run time.

### 5.3 LoRAs: install, select, weight

**Registry**
- A LoRA entry: name, file (`.safetensors`), description, trigger words, recommended
  weight, base-model compatibility tag (Qwen-Image vs Qwen-Image-Edit), applicable mode(s),
  thumbnail/sample, source + source ref, license, and enabled flag.
- Stored under `/data/loras/` with metadata in PostgreSQL.

**Install sources** (all supported in v1)
- **Local upload** of a `.safetensors` file.
- **Hugging Face** — in-app search of the Hub and import by repo id (auto-detecting the
  weight file and pulling card metadata/thumbnails). Uses an optional HF token for gated
  repos.
- **CivitAI** — in-app search and import by model/version id or page URL, pulling
  trigger words, preview images, and recommended weight. Uses a CivitAI API key. Accepts
  URLs from **`civitai.com`, `civitai.red`, and `civitai.green`** — these are filtered
  views over one database, so the importer strips the domain, extracts the model/version
  id, and resolves it through the canonical `civitai.com/api/v1` API (following any
  domain-color redirects; the API key authorizes mature content).
- **Direct URL** to a `.safetensors` file.
- All imports are **background download jobs** with progress, checksum verification, and a
  size guard; metadata is editable after import.

**Selection & application at run time**
- Multi-select LoRAs, each with an independent **weight slider** (0–~1.5).
- Backend applies them via `pipeline.load_lora_weights(...)` + PEFT `set_adapters` with
  per-adapter weights; adapters are cached and swapped efficiently between jobs.
- Validation: warn on base-model/mode mismatch (e.g. a Generate-only LoRA in Edit mode);
  surface trigger words as a one-click insert into the prompt.
- **Quant interaction:** flag that LoRA stacking adds VRAM, and that some quantized
  backends (notably Nunchaku int4) constrain how/whether LoRAs fuse — the model advisor
  (§9) accounts for active LoRAs in its headroom estimate.
- Note: 2511 already *integrates* some popular LoRAs into the base — surfaced in the UI so
  users don't double-apply.

**API keys** for HF/CivitAI live on the **Integrations settings page** (§5.7), stored
encrypted at rest and never returned to the client in plaintext.

### 5.4 Resolution presets

Output size is chosen from three independent controls that compose into a concrete `W×H`:

1. **Base size:** `Match source · 512 · 1024 · 1536 · 2048`
2. **Orientation:** `Square · Portrait · Landscape`
3. **Aspect ratio** (shown for Portrait/Landscape only):

| Orientation | Aspect ratios offered                 |
|-------------|----------------------------------------|
| Square      | 1:1                                    |
| Portrait    | 4:5 · 3:4 · 2:3 · 9:16                  |
| Landscape   | 5:4 · 4:3 · 3:2 · 16:9                  |

**Resolution rules**
- **Base size = the longer edge** (and the side length for Square). The shorter edge is
  derived from the chosen aspect ratio.
- Both dimensions are snapped to a multiple of **16** (latent-friendly) after derivation.
- **Match source** (Edit mode): uses the first input image's dimensions (snapped to ×16),
  optionally capped to a max long edge to protect VRAM. In Generate mode "Match source"
  is unavailable (no input), so it falls back to a default base size.
- Larger sizes are annotated with their VRAM cost in the advisor (§9), so the resolution
  picker and model picker stay consistent.
- The UI shows a **live preview chip** of the final `W×H` and a proportional thumbnail
  before generating.
- Example: `1024 / Landscape / 3:2` → `1024 × 688` (688 snapped from 682.7).

### 5.5 Generation (run composer) & output

**Mode switch** — a prominent **Generate / Edit** toggle at the top of the composer:
- **Edit:** input images are required (≥1); "Match source" resolution available.
- **Generate:** the input panel is hidden/disabled; everything else (prompt, LoRAs,
  resolution, settings) is shared.

**Composer**
- Left: input images (uploads + library + prompt-pinned + slots), drag-to-reorder.
  *(Edit mode only.)*
- Center: prompt editor (with library load/save, slot fills, variable fills).
- Right: settings — model/precision selector (§9), LoRAs+weights, resolution preset,
  steps, true_cfg_scale, negative prompt, seed (random/lock), batch count.
- A clear, prominent **Generate** action; live size + VRAM-headroom summary.

**Execution**
- Job enqueued → progress streamed over WebSocket: step N/40, ETA, **and a live latent
  preview** so the user watches the image resolve rather than staring at a timer. Previews
  are produced by a lightweight latent→RGB decode on a throttled cadence (e.g. every few
  steps) to keep the perf cost modest; cadence is configurable and can be turned off.
- Cancelable jobs.

**Output**
- Result gallery with before/after compare slider (Edit mode).
- **Download** any output directly, or **promote to library** (§5.1) — both uploads and
  outputs can become permanent assets.
- **Full reproducibility metadata** saved as a sidecar (and optionally embedded in the
  PNG): mode, prompt, negative, all input image refs/hashes, LoRAs+weights, resolution,
  steps, cfg, seed, **precision/quant**, model id/revision, backend/device.
- One-click **"send output to input"** (switches to Edit mode with the output loaded),
  **"reuse settings"**, and **"save as preset/prompt"**.
- **History** view: searchable, filterable by mode/prompt/LoRA/date; re-run any past job.

### 5.6 Batch / sweep mode

First-class batch support, tuned for the catalog/furniture use case:
- **Slot sweep (Edit):** fill an open slot (e.g. `furniture`) with N images and produce N
  outputs against the same pinned background + prompt — one staged render per piece.
- **Seed sweep:** N seeds for the same composition.
- **Param sweep:** small grids over CFG/steps for tuning.
- Batches are expanded into individual queued jobs (so progress, cancel, and history work
  per item) and grouped under a batch id in the UI.

### 5.7 Integrations & settings

A dedicated **Integrations** settings page manages external-service credentials:
- **Hugging Face token** — for gated/private repos and higher rate limits.
- **CivitAI API key** — for search/import.
- (Extensible to future providers.)

Keys are validated on save (a lightweight authenticated call), stored **encrypted at rest**
(app-level encryption with a key from env/secret), masked in the UI, and exposed only to
backend services. A broader **Settings** area also covers: default model/precision, device
selection, default output format/quality, data paths, and theme/accent.

### 5.8 Prompt enhancement (Qwen-VL rewriter)

The Qwen-Image team **strongly recommends prompt rewriting** for stable, high-quality
editing, and ships an official Prompt Enhancement Tool powered by Qwen-VL. QIE Studio
bakes this in as a first-class, **multimodal** prompt enhancer.

**Behavior**
- An **✨ Enhance** action in the prompt editor rewrites the user's instruction into a
  model-friendly prompt. In **Edit** mode it is genuinely multimodal — the VL model
  *sees the input image(s)* alongside the text, which is the whole point (it grounds the
  rewrite in what's actually in the picture). In **Generate** mode it enriches the text
  prompt.
- The result is shown as an editable **diff/preview** the user can accept, tweak, or
  reject — never a silent rewrite. An optional **auto-enhance per run** toggle (default
  on for Edit, given the team's guidance) can apply it automatically before generation.
- The original and enhanced prompts are both stored in job metadata for reproducibility.

**System prompts — reuse the official ones verbatim**
- We vendor the official rewriting templates from
  `QwenLM/Qwen-Image: src/examples/tools/prompt_utils.py` and `prompt_utils_2512.py`,
  used **verbatim**, and select the variant that matches the chosen edit-model version
  (e.g. the 2512 template with the 2512 base). This keeps us aligned with the model
  authors' tuning rather than inventing our own prompts.
- These are vendored under their upstream license with attribution, and pinned so a
  template update is a deliberate bump.

**Selectable rewriter models (with a VRAM/compat advisor, like §9)**
The enhancer model is **user-selectable**, mirroring the image-model advisor:

| Rewriter | Quant | Backend | Co-residence vs swap | Notes |
|----------|-------|---------|----------------------|-------|
| **Qwen3-VL-30B-A3B-Instruct** | **AWQ** (CUDA) or **Q4/Q5 GGUF** (portable) | AWQ: CUDA · GGUF: CUDA/ROCm/MPS via a GGUF runtime | Usually **requires a runtime swap** (unload image model → load VL → reload) on a single GPU | Best-quality rewrites; A3B MoE keeps active params modest, but total weights are large. |
| **Qwen3-VL-8B-Instruct** | **fp8** (CUDA Ada/Hopper) or GGUF | CUDA (fp8) · GGUF elsewhere | Often **co-resides** alongside the loaded image model → no swap, low latency | Strong fallback when the user wants instant enhancement without unload/reload churn. |

- A **rewriter advisor** estimates whether the chosen VL model can **co-reside** with the
  currently loaded image pipeline (given free VRAM after the image model + precision +
  LoRAs) or whether it needs a **swap**, and surfaces the trade-off: *"30B gives better
  rewrites but adds a load/unload (~Xs) each run; 8B fp8 fits alongside QIE for instant
  rewrites."* It only offers quant/backend combos the detected device supports (AWQ/fp8 are
  CUDA-centric; **GGUF is the portable path for ROCm/MPS**).
- Model swapping is managed by the same model manager and serialized through the GPU queue;
  the VL model can also be pinned **resident** (co-resident mode) to avoid per-run swaps
  when VRAM allows.
- Like the image models, rewriter model ids/versions and defaults are config-driven and
  overridable, with arbitrary HF repo ids allowed.

---

## 6. Data model (initial)

PostgreSQL via SQLModel/SQLAlchemy + Alembic. Every user-owned row carries a nullable
`owner_id` (the implicit single user in v1) so multi-user is additive, not a migration.
Binaries are addressed by a **storage key** resolved through the `StorageProvider` (local
FS now, object store later) — the DB never stores absolute paths.

```
# (owner_id present on all user-owned tables; omitted below for brevity)

Asset(id, scope[ephemeral|library], storage_key, thumb_key, name, description,
      tags[], width, height, format, bytes, sha256,
      source[upload|output|import], source_job_id?, created_at, last_used_at)

Prompt(id, name, text, tags[], mode[generate|edit|any], defaults_json,
       created_at, updated_at)
PromptImage(id, prompt_id, position, role[pinned|slot], asset_id?, slot_name?,
            slot_min?, slot_max?, hint?)
PromptLora(id, prompt_id, lora_id, weight)

Lora(id, name, storage_key, thumb_key, description, trigger_words[],
     recommended_weight, base_compat[qwen-image|qwen-image-edit], modes[],
     source[upload|hf|civitai|url], source_ref, license, enabled, created_at)

Job(id, batch_id?, mode[generate|edit], status[queued|running|done|error|canceled],
    precision[bf16|fp8|int4], device, model_id, model_revision,
    prompt, enhanced_prompt?, rewriter_model?,          # both prompts kept for repro
    params_json, progress, error?, created_at, started_at, ended_at)
JobInput(id, job_id, position, asset_id)        # resolved ordered inputs (edit mode)
JobOutput(id, job_id, position, storage_key, thumb_key, metadata_json, seed)

Integration(id, provider[huggingface|civitai], encrypted_key, label,
            status, last_validated_at)
Setting(key, value_json)                         # default model/precision, theme, etc.
```

Binaries live behind the storage provider; PostgreSQL holds metadata and references.

---

## 7. API surface (sketch)

```
# Assets
POST   /api/assets/upload            # ephemeral upload(s)
POST   /api/assets/{id}/promote      # ephemeral upload OR output -> library
GET    /api/assets?scope=&tag=&q=&source=
PATCH  /api/assets/{id}              # rename/tag/move
DELETE /api/assets/{id}

# Prompts
GET    /api/prompts?q=&tag=&mode=
POST   /api/prompts                  # create / save
PATCH  /api/prompts/{id}             # rename, edit, bindings
POST   /api/prompts/{id}/duplicate
DELETE /api/prompts/{id}

# LoRAs
GET    /api/loras
POST   /api/loras/upload             # install via file
POST   /api/loras/import             # install via {source: hf|civitai|url, ref}
GET    /api/loras/search?source=&q=  # search HF / CivitAI
PATCH  /api/loras/{id}
DELETE /api/loras/{id}

# Models / advisor
GET    /api/device                   # detected backend, SM/caps, total/free VRAM, RAM
POST   /api/models/advise            # {mode, resolution, batch, loras[]} -> ranked options
GET    /api/models                   # configured model ids + available precisions

# Prompt enhancement (Qwen-VL rewriter)
GET    /api/rewriter/models          # selectable VL models + quant/backend options
POST   /api/rewriter/advise          # {vl_model, image_model_state} -> co-reside vs swap
POST   /api/rewriter/enhance         # {mode, prompt, image_ids[]} -> enhanced prompt

# Integrations
GET    /api/integrations             # providers + status (keys masked)
PUT    /api/integrations/{provider}  # set/validate API key
DELETE /api/integrations/{provider}

# Presets
GET    /api/presets/resolution       # enumerations + a resolve helper
POST   /api/presets/resolution/resolve  # {base, orientation, aspect, source_dims?} -> {w,h}

# Jobs / generation
POST   /api/jobs                     # submit a run (mode, precision, composed params)
POST   /api/jobs/batch               # submit a batch/sweep -> batch_id + child jobs
GET    /api/jobs/{id}
POST   /api/jobs/{id}/cancel
GET    /api/jobs?status=&mode=&batch_id=&q=   # history
WS     /ws/jobs/{id}                 # progress stream
```

---

## 8. Containerization & deployment

v1 targets a **local workstation** but the layout is kept cloud-portable.

- **Backend images (per backend):**
  - **CUDA:** NVIDIA CUDA runtime base + Python 3.13 + PyTorch CUDA wheels (this is the
    primary, fully-featured image, incl. fp8 + Nunchaku int4).
  - **ROCm:** ROCm base + PyTorch ROCm wheels (no Nunchaku int4).
  - Multi-stage builds to keep runtimes lean; backend selected by build arg / image tag.
- **Apple Silicon (MPS):** GPU acceleration is **not available inside Docker** on macOS, so
  Apple-Silicon users run the backend **natively** (a documented `uv`/venv setup on Python
  3.13 using MPS) while still running **Postgres in Docker**. The app detects MPS the same
  way regardless of how it's launched.
- **GPU access:** `--gpus all` + NVIDIA Container Toolkit (CUDA); device groups for ROCm.
- **Services (Compose):**
  - `db` — **PostgreSQL** (with a persistent volume).
  - `backend` — FastAPI + accelerator, serving the built frontend as static assets.
  - optional `frontend` dev service for hot-reload during development.
- **Volumes:**
  - `pgdata` — PostgreSQL data.
  - `/data` — assets, outputs, loras (the storage provider's local root).
  - `/models` — Hugging Face hub cache (so the tens-of-GB weights aren't re-downloaded).
- **Config:** environment/`.env` — `DATABASE_URL`, model ids/revisions, device override,
  default precision, storage backend + root, data/model paths, port, and the secret used to
  encrypt integration keys. Nothing host-specific is hard-coded (keeps the cloud door open).
- **Migrations:** Alembic runs on startup.
- **Startup logging:** per project convention, the backend logs *before* each slow phase
  (DB connect/migrate, model download, weight load/quantize, warm-up) with elapsed-time
  context and flushed output, so the container never sits silent during long startup stalls.

---

## 9. Model precision & the VRAM-aware advisor

Even on 40GB+ cards, full bf16 leaves limited headroom once the DiT, text encoder, VAE,
LoRA stack, larger latents, and batches are all resident. Rather than force a single
choice, the app offers **selectable precision/quantization** and actively **recommends**
the best fit for the user's hardware.

### 9.1 Precision options offered

| Option | Approx. DiT weights | Backend(s) | Notes |
|--------|--------------------|------------|-------|
| **bf16** (full) | ~40 GB | CUDA, ROCm, MPS | Highest quality, highest VRAM. Best when you have ample headroom. |
| **fp8_e4m3fn (scaled)** | ~20 GB | CUDA (Ada/Hopper, SM ≥ 8.9), some ROCm | ~half the weight memory, near-bf16 quality. Great middle ground; leaves room for LoRAs/batches/large latents. |
| **SVDQuant int4 (Nunchaku)** | ~10–12 GB | **CUDA only** | Lowest VRAM, strong quality for int4, fastest on supported cards. Hidden on ROCm/MPS; LoRA fusion constraints noted at selection. |

(VAE + text encoder add a few GB on top; the advisor folds these in.)

### 9.2 The advisor

The `model_advisor` combines the `device_manager` capabilities with the planned run to
recommend an option and explain the trade-off:

**Inputs it considers**
- Detected backend and **compute capability / SM** (e.g. fp8_e4m3fn needs SM ≥ 8.9;
  Nunchaku needs CUDA + a supported arch).
- **Total and free VRAM** (and system RAM, relevant for offloading and MPS unified memory).
- The **planned run**: mode, target resolution (latent size scales with W×H), batch size,
  and number/size of active LoRAs.

**What it produces**
- A **recommended** option, plus each option marked **Recommended / Fits / Tight / Won't
  fit** with a one-line rationale and an estimated peak-VRAM figure and headroom.
- Backend-aware filtering (it never offers a CUDA-only option on ROCm/MPS).
- A live re-evaluation as the user changes resolution, batch, or LoRAs — so the model
  selector and resolution picker stay in sync.

**Example guidance strings**
- *24 GB Ada (SM 8.9):* "bf16 won't fit with LoRAs — **fp8_e4m3fn recommended** (~15 GB
  peak, comfortable headroom). int4 also available for max headroom/speed."
- *16 GB:* "**SVDQuant int4 recommended** (~11 GB peak). fp8 is tight at this resolution;
  bf16 won't fit."
- *80 GB H100:* "**bf16 recommended** — full headroom for large latents and batches."
- *Apple M-series:* "bf16/fp16 on MPS; int4 (Nunchaku) is CUDA-only and unavailable here."

### 9.3 Supporting strategies
- **Offloading:** `enable_model_cpu_offload()` / sequential offload, attention slicing, and
  VAE tiling — auto-suggested by the advisor when a chosen option is "Tight," and available
  as manual toggles.
- **Warm-up:** optional warm-up pass at startup to avoid first-run latency.
- **Caching:** keep the active pipeline resident between jobs; cache compiled/quantized
  weights and LoRA adapters; reuse the text encoder across modes where possible.
- **Single-accelerator serialization:** the job queue guarantees one inference at a time.
  v1 is single-GPU; per-GPU queues for multi-GPU are future work.
- **Selection is recorded** in each job's reproducibility metadata.

### 9.4 Delivered-v1 notes (validated on the reference A6000)

The v1 build was validated on the reference hardware (NVIDIA RTX A6000, 48 GB, Ampere
SM 8.6). Notes on how the precision paths behave as delivered:

- **Stack:** Python 3.13, PyTorch 2.12 (cu126), diffusers 0.38, in the CUDA **devel** image
  (`nvidia/cuda:12.6.3-cudnn-devel`). The devel base is required because both
  optimum-quanto (fp8) and Nunchaku JIT-compile CUDA kernels at load time (the runtime base
  lacks `nvcc`/headers).
- **bf16:** highest quality/VRAM. At 1024² on 48 GB it is *tight* once the text encoder is
  resident, so the advisor recommends **fp8** there; bf16 is best used with CPU offload or on
  larger cards. (Offload places the model via accelerate — the pipeline no longer calls
  `.to(cuda)` before enabling offload.)
- **fp8_e4m3fn (scaled, optimum-quanto):** the validated default on this card. Confirmed to
  **run end-to-end** (Generate→Edit), but it is **emulated** on SM 8.6 (native fp8 needs
  SM ≥ 8.9), so there is no speedup and the one-time quantization pass is slow (~13–16 min
  per model). ~37 GB resident at 1024² — comfortable on 48 GB without offload.
- **SVDQuant int4 (Nunchaku):** the PyPI package name `nunchaku` is an unrelated placeholder;
  the real SVDQuant runtime ships as arch/torch-specific wheels from the upstream project and
  was **not available** for this torch 2.12/cp313/cu126 combination at build time. The
  pipeline therefore degrades gracefully — int4 selection raises a clear, actionable error
  rather than crashing — and the advisor still surfaces int4 on supported CUDA arches. Wiring
  a matching Nunchaku wheel (or building it from source) is the remaining step to enable int4.

---

## 10. Functionality summary

**Core (in scope):** Generate + Edit modes, multi-image input, image library (uploads +
outputs + promote), prompt library with slots/bindings, LoRA registry + multi-source
import, resolution presets, VRAM-aware model advisor, live latent previews, batch/sweep,
generation history with full reproducibility, send-to-input, before/after compare.

**Committed round-out features (§13):** background removal, upscale/refine, Qwen-VL prompt
enhancer (§5.8), recipes, catalog export.

**Deferred / fast-followers:** C2PA provenance, input aspect-fit controls, session
autosave, queue-management UI, favorites/ratings, library import/export, per-project
workspaces (the schema's workspace scoping is ready for it), masking/regional edits, and
multi-GPU scheduling.

---

## 11. Open questions

### Resolved (rounds 1–2)
- **Modes:** Generate (text-to-image) + Edit.
- **Hardware/backends:** multi-backend CUDA/ROCm/MPS; 40GB+ reference.
- **Precision:** selectable bf16 / fp8_e4m3fn / SVDQuant int4 with a VRAM-aware advisor.
- **Users/auth:** single-user, no auth (pluggable for later).
- **Deployment:** local workstation v1; architecture kept cloud/multi-user-ready.
- **Metadata store:** PostgreSQL.
- **LoRA sources:** upload + HF + CivitAI + URL, with an Integrations/API-keys page.
- **Editing scope:** whole-image only in v1.
- **Resolution semantics:** base size = longer edge.
- **UI delivery:** browser web app, React/Tailwind/shadcn.
- **Image formats:** PNG/JPEG/WebP/GIF in (+ HEIC); PNG/WebP/JPEG out.

### Resolved (round 3)
- **Multi-GPU:** single-GPU in v1; multi-GPU scheduling is future work.
- **Model versions:** sensible defaults, user-selectable at run time, newer + older
  supported (arbitrary HF repo ids allowed).
- **Live latent previews:** in scope for v1.
- **Import guardrails:** no size or license caps.
- **CivitAI domains:** `.com` / `.red` / `.green` all accepted via the shared v1 API.
- **Project name:** **QIE Studio**.

### Still open
- Which **moderate-effort "round out" features** (§13) to commit to v1 vs defer.

---

## 12. Proposed build order

1. Backend skeleton: FastAPI app, config, **PostgreSQL** + Alembic, storage/auth/queue
   interfaces (local/no-auth/in-process impls), health + startup logging.
2. Device manager (CUDA/ROCm/MPS detection + caps) and model advisor.
3. Pipeline service: load Generate + Edit pipelines, bf16 first, then fp8 and int4;
   single- then multi-image; progress callback.
4. Resolution service + presets API.
5. Asset store (uploads + outputs + library, promote-to-library) and uploads.
6. Job queue + WebSocket progress + outputs/history; batch/sweep.
7. LoRA manager (upload + HF/CivitAI/URL search & import, weighted stacking) + Integrations.
8. Prompt store (CRUD, LoRA bindings, pinned images, slots).
9. Frontend: mode switch + composer, libraries, history, model advisor UI, settings/
   integrations, theming, command palette.
10. Round-out features: prompt enhancer (vendor official `prompt_utils*` + VL model
    manager/advisor), background removal, upscale/refine, recipes, catalog export.
11. Dockerfiles (CUDA/ROCm) + Compose (db + backend) + native MPS docs.
12. Polish: compare slider, library import/export, queue management UI.

---

## 13. "Round out" features (moderate effort, clear win)

Not "AI Photoshop" — these stay within the generate/edit/library workflow but make it
markedly more useful.

**Committed to v1**
1. **Background removal / cutout (rembg or SAM-based).** One-click isolate a subject from
   an uploaded image before compositing. Directly amplifies the furniture workflow (drop a
   chair photo, auto-cut it, place it in the pinned room) and is a small dependency.
2. **Output upscaling / refine pass.** Optional post-process (Real-ESRGAN, or a Qwen-Image
   img2img refine) to turn 1–2 MP results into deliverables.
3. **Prompt enhancer (Qwen-VL).** Full spec in **§5.8** — multimodal rewriting using the
   official `prompt_utils*` templates, with selectable Qwen3-VL-30B-A3B / 8B models and a
   co-reside-vs-swap advisor.
4. **Recipes (one-click run configs).** Save an entire composition — prompt + LoRAs +
   resolution + settings + slots — as a named recipe.
5. **Catalog/batch export.** Export a batch as a zip + contact-sheet grid + a CSV/JSON of
   per-image parameters. Natural companion to batch/sweep mode.

**Fast-followers (not committed)**
6. **C2PA / provenance metadata.** Embed content-credential provenance marking outputs as
   AI-generated. Low-effort, increasingly expected in 2026.
7. **Input aspect-fit controls (cover / contain / pad)** for mismatched inputs, with preview.
8. **Session autosave / restore** of the in-progress composition.
9. **Queue management UI** — reorder/pause/prioritize queued jobs.
10. **Favorites & ratings + quick filters** on outputs/library.

---

## Appendix — sources

- [Qwen/Qwen-Image-Edit-2511 · Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Qwen-Image-Edit-2511 intro — StableLearn](https://stable-learn.com/en/qwen-image-edit-2511-intro/)
- [QwenLM/Qwen-Image — GitHub](https://github.com/QwenLM/Qwen-Image)
- [Qwen-Image-Edit-2511 ComfyUI workflow — ComfyUI docs](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511)
- [Civitai REST API Reference](https://github.com/civitai/civitai/wiki/REST-API-Reference)
- [Two Front Doors: Civitai.com, Civitai.red — Civitai](https://civitai.com/articles/28369/two-front-doors-civitaicom-civitaired-and-whats-next)
- [QwenLM/Qwen-Image — official prompt rewriting utilities (`src/examples/tools/prompt_utils.py`, `prompt_utils_2512.py`)](https://github.com/QwenLM/Qwen-Image/tree/main/src/examples/tools)
