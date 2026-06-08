// TypeScript types mirroring the QIE Studio backend API (DESIGN §6 data model, §7 API).
// The backend is the source of truth; these shapes match the Pydantic schemas exactly.

export type Mode = "generate" | "edit";
export type Precision = "bf16" | "fp8" | "int4";
export type AssetScope = "ephemeral" | "library";
export type AssetSource = "upload" | "output" | "import";
export type JobStatus = "queued" | "running" | "done" | "error" | "canceled";
export type LoraSource = "upload" | "hf" | "civitai" | "url";
export type BaseCompat = "qwen-image" | "qwen-image-edit";
export type IntegrationProvider = "huggingface" | "civitai";

// ---- Assets -----------------------------------------------------------------
// Backend: app/schemas/assets.py::AssetRead

export interface Asset {
  id: string;
  scope: AssetScope;
  source: AssetSource;
  source_job_id: string | null;
  storage_key: string;
  thumb_key: string | null;
  name: string;
  description: string;
  tags: string[];
  collection: string | null;
  width: number;
  height: number;
  format: string;
  bytes: number;
  sha256: string;
  created_at: string;
  last_used_at: string | null;
}

// ---- Prompts ----------------------------------------------------------------
// Backend: app/schemas/prompts.py

export type PromptImageRole = "pinned" | "slot";

export interface PromptImage {
  id?: string;
  position?: number;
  role: PromptImageRole;
  asset_id?: string | null;
  slot_name?: string | null;
  slot_min?: number | null;
  slot_max?: number | null;
  hint?: string | null;
}

export interface PromptLora {
  lora_id: string;
  weight: number;
}

export interface Prompt {
  id: string;
  name: string;
  text: string;
  tags: string[];
  mode: Mode | "any";
  favorite: boolean;
  defaults_json: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  last_used_at: string | null;
  images: PromptImage[];
  loras: PromptLora[];
}

// Body for POST /api/prompts (PromptCreate). PATCH accepts the same fields, all optional.
export interface PromptCreate {
  name: string;
  text?: string;
  tags?: string[];
  mode?: Mode | "any";
  favorite?: boolean;
  defaults_json?: Record<string, unknown>;
  images?: PromptImage[];
  loras?: PromptLora[];
}

// ---- LoRAs ------------------------------------------------------------------
// Backend: app/schemas/loras.py

export interface Lora {
  id: string;
  name: string;
  storage_key: string;
  thumb_key: string | null;
  description: string;
  trigger_words: string[];
  recommended_weight: number;
  base_compat: string;
  modes: string[];
  source: string;
  source_ref: string;
  license: string;
  sha256: string;
  bytes: number;
  enabled: boolean;
  created_at: string;
}

export interface LoraSearchResult {
  source: string; // hf | civitai
  name: string;
  ref: string;
  base_model: string;
  trigger_words: string[];
}

export interface LoraImportRequest {
  source: "hf" | "civitai" | "url";
  ref: string;
  name?: string | null;
  filename?: string | null;
}

// ---- Device / models / advisor ---------------------------------------------
// Backend: app/schemas/advisor.py

export type Backend = "cuda" | "rocm" | "mps" | "cpu";

export interface DeviceInfo {
  backend: Backend;
  device_str: string;
  name: string;
  compute_capability: [number, number] | null;
  total_vram_mb: number;
  free_vram_mb: number;
  total_ram_mb: number;
  supports_bf16: boolean;
  supports_fp8_native: boolean;
  supports_int4_nunchaku: boolean;
}

export interface ModelEntry {
  mode: Mode;
  model_id: string;
  available_precisions: Precision[];
}

export interface ModelCatalog {
  device: DeviceInfo;
  models: ModelEntry[];
  default_precision: Precision;
}

// PrecisionOption.status is a free-form string on the backend; these are the known values.
export type FitVerdict =
  | "recommended"
  | "fits"
  | "tight"
  | "wont_fit"
  | "unavailable"
  | (string & {});

export interface PrecisionOption {
  precision: Precision;
  available: boolean;
  status: FitVerdict;
  est_peak_vram_mb: number;
  headroom_mb: number;
  rationale: string;
  caveats: string[];
}

// Body for POST /api/models/advise (AdviseRequest). Provide longer_edge OR resolution.
export interface AdviceRequest {
  mode: Mode;
  longer_edge?: number;
  resolution?: ResolveRequest;
  batch: number;
  loras: { lora_id: string; weight: number }[];
}

export interface Advice {
  device: DeviceInfo;
  options: PrecisionOption[];
  recommended: Precision | null;
}

// ---- Rewriter (Qwen-VL) -----------------------------------------------------
// Backend: app/schemas/rewriter.py

export interface RewriterQuantOption {
  quant: string;
  label: string;
  backends: string[];
  requires_runtime: string | null;
}

export interface RewriterModel {
  model_id: string;
  label: string;
  est_vram_mb: number;
  is_default: boolean;
  co_resides_typically: boolean;
  notes: string;
  quant_options: RewriterQuantOption[];
}

export interface RewriterModelsResponse {
  models: RewriterModel[];
  default_model: string;
}

// Body for POST /api/rewriter/advise (AdviseRewriterRequest).
export interface RewriterAdviceRequest {
  vl_model: string;
  image_model_resident_mb?: number;
}

export interface RewriterAdvice {
  vl_model: string;
  can_co_reside: boolean;
  decision: string;
  est_vl_vram_mb: number;
  image_model_resident_mb: number;
  free_after_image_mb: number;
  rationale: string;
}

// Body for POST /api/rewriter/enhance (EnhanceRequest).
export interface EnhanceRequest {
  mode: Mode;
  prompt: string;
  image_ids: string[];
  vl_model?: string;
  edit_model_id?: string;
}

export interface EnhanceResult {
  enhanced_prompt: string;
  original_prompt: string;
}

// ---- Resolution -------------------------------------------------------------
// Backend: app/schemas/advisor.py (PresetsResponse, ResolveRequest, ResolveResponse)

export type Orientation = "square" | "portrait" | "landscape";
export type BaseSize = "match" | "512" | "1024" | "1536" | "2048";

export interface ResolutionEnums {
  base_sizes: number[];
  match_supported: boolean;
  default_base_size: number;
  orientations: string[];
  aspect_ratios: Record<string, string[]>;
  snap_multiple: number;
}

// Body for POST /api/presets/resolution/resolve (ResolveRequest).
export interface ResolveRequest {
  base: number | string; // int | "match"
  orientation?: string;
  aspect?: string | null;
  source_dims?: [number, number] | null; // (width, height)
  max_long_edge?: number | null;
}

// ResolveResponse: { w, h }.
export interface ResolvedResolution {
  w: number;
  h: number;
}

// ---- Integrations -----------------------------------------------------------
// Backend: app/schemas/integrations.py

export type IntegrationStatusValue = "unconfigured" | "valid" | "invalid";

export interface Integration {
  provider: string;
  label: string;
  status: IntegrationStatusValue;
  configured: boolean;
  last_validated_at: string | null;
  masked_hint: string | null;
}

// Body for PUT /api/integrations/{provider} (IntegrationSet).
export interface IntegrationSet {
  key: string;
  label?: string;
}

// ---- Jobs -------------------------------------------------------------------
// Backend: app/schemas/jobs.py + app/routers/jobs.py::_serialize

export interface JobOutput {
  id: string;
  position: number;
  seed: number | null;
  file_url: string;
  thumb_url: string | null;
  metadata: Record<string, unknown>;
}

export interface Job {
  id: string;
  batch_id: string | null;
  mode: Mode;
  status: JobStatus;
  precision: Precision;
  device: string;
  model_id: string;
  model_revision: string | null;
  prompt: string;
  enhanced_prompt: string | null;
  rewriter_model: string | null;
  params: Record<string, unknown>;
  progress: number;
  progress_step: number;
  progress_total: number;
  error: string | null;
  created_at: string | null;
  started_at: string | null;
  ended_at: string | null;
  outputs: JobOutput[];
}

// Resolution spec embedded in a JobSubmit (app/schemas/jobs.py::ResolutionSpec).
export interface ResolutionSpec {
  base?: number | string | null; // int | "match"
  orientation?: string | null;
  aspect?: string | null;
  width?: number | null;
  height?: number | null;
  max_long_edge?: number | null;
}

// Body for POST /api/jobs (JobSubmit) — FLAT fields, not nested params.
export interface JobSubmit {
  mode: Mode;
  prompt?: string;
  enhanced_prompt?: string | null;
  rewriter_model?: string | null;
  negative_prompt?: string;
  model_id?: string | null;
  model_revision?: string | null;
  precision?: Precision;
  prompt_id?: string | null;
  input_asset_ids?: string[];
  slot_fills?: Record<string, string[]>;
  loras?: PromptLora[];
  resolution?: ResolutionSpec;
  num_inference_steps?: number;
  true_cfg_scale?: number;
  guidance_scale?: number;
  seed?: number | null;
  batch?: number;
  output_format?: "png" | "webp" | "jpeg";
  output_quality?: number;
  preview_every_n_steps?: number | null;
  enable_model_cpu_offload?: boolean;
  enable_sequential_cpu_offload?: boolean;
  enable_attention_slicing?: boolean;
  enable_vae_tiling?: boolean;
}

export type SweepKind = "slot" | "seed" | "param";

export interface SweepSpec {
  kind: SweepKind;
  slot_name?: string | null;
  slot_asset_ids?: string[];
  seeds?: number[];
  steps_grid?: number[];
  cfg_grid?: number[];
}

// Body for POST /api/jobs/batch (BatchSubmit).
export interface BatchSubmit {
  base: JobSubmit;
  sweep: SweepSpec;
}

export interface JobSubmitResponse {
  job_id: string;
}

export interface BatchSubmitResponse {
  batch_id: string;
  job_ids: string[];
}

// ---- Recipes ----------------------------------------------------------------
// Backend: app/schemas/recipes.py

export interface Recipe {
  id: string;
  name: string;
  description: string;
  tags: string[];
  mode: Mode;
  config_json: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface RecipeCreate {
  name: string;
  description?: string;
  tags?: string[];
  mode?: Mode;
  config_json?: Record<string, unknown>;
}

// ---- Tools ------------------------------------------------------------------
// Backend: app/schemas/tools.py — POST /api/tools/* returns { asset }.

export interface ToolResult {
  asset: Asset;
}

// ---- WebSocket progress messages -------------------------------------------
// Backend: app/services/job_service.py publishes {type: snapshot|progress|status}.

export interface JobSnapshotMessage {
  type: "snapshot";
  status: JobStatus;
  progress: number;
  step: number;
  total: number;
}

export interface JobProgressMessage {
  type: "progress";
  step: number;
  total: number;
  progress: number;
  /** data:image/png;base64,... preview of the throttled latent->RGB decode */
  preview?: string;
}

export interface JobStatusMessage {
  type: "status";
  status: JobStatus;
  device?: string;
  outputs?: { id: string; position: number; seed: number | null }[];
  error?: string;
}

export type JobWsMessage =
  | JobSnapshotMessage
  | JobProgressMessage
  | JobStatusMessage;
