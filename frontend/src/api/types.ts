// TypeScript types mirroring the QIE Studio backend API (DESIGN §6 data model, §7 API).

export type Mode = "generate" | "edit";
export type Precision = "bf16" | "fp8" | "int4";
export type AssetScope = "ephemeral" | "library";
export type AssetSource = "upload" | "output" | "import";
export type JobStatus = "queued" | "running" | "done" | "error" | "canceled";
export type LoraSource = "upload" | "hf" | "civitai" | "url";
export type BaseCompat = "qwen-image" | "qwen-image-edit";
export type IntegrationProvider = "huggingface" | "civitai";

// ---- Assets -----------------------------------------------------------------

export interface Asset {
  id: string;
  scope: AssetScope;
  storage_key: string;
  thumb_key: string | null;
  name: string | null;
  description: string | null;
  tags: string[];
  width: number;
  height: number;
  format: string;
  bytes: number;
  sha256: string;
  source: AssetSource;
  source_job_id: string | null;
  created_at: string;
  last_used_at: string | null;
}

// ---- Prompts ----------------------------------------------------------------

export type PromptImageRole = "pinned" | "slot";

export interface PromptImage {
  id?: string;
  position: number;
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

export interface PromptDefaults {
  resolution_preset?: string;
  steps?: number;
  true_cfg_scale?: number;
  seed?: number | null;
}

export interface Prompt {
  id: string;
  name: string;
  text: string;
  tags: string[];
  mode: Mode | "any";
  favorite?: boolean;
  loras: PromptLora[];
  images: PromptImage[];
  defaults: PromptDefaults;
  created_at: string;
  updated_at: string;
}

export type PromptCreate = Omit<Prompt, "id" | "created_at" | "updated_at">;

// ---- LoRAs ------------------------------------------------------------------

export interface Lora {
  id: string;
  name: string;
  storage_key: string;
  thumb_key: string | null;
  description: string | null;
  trigger_words: string[];
  recommended_weight: number;
  base_compat: BaseCompat;
  modes: Mode[];
  source: LoraSource;
  source_ref: string | null;
  license: string | null;
  enabled: boolean;
  created_at: string;
}

export interface LoraSearchResult {
  source: LoraSource;
  ref: string;
  name: string;
  description: string | null;
  thumb_url: string | null;
  trigger_words: string[];
  recommended_weight: number | null;
  base_compat: BaseCompat | null;
}

export interface LoraImportRequest {
  source: "hf" | "civitai" | "url";
  ref: string;
}

// ---- Device / models / advisor ---------------------------------------------

export type Backend = "cuda" | "rocm" | "mps" | "cpu";

export interface DeviceInfo {
  backend: Backend;
  device_name: string;
  compute_capability: string | null;
  total_vram_bytes: number;
  free_vram_bytes: number;
  total_ram_bytes: number;
  supports_fp8: boolean;
  supports_int4: boolean;
}

export interface ModelVersion {
  id: string;
  revision: string | null;
  label: string;
  mode: Mode;
  is_default: boolean;
}

export interface ModelCatalog {
  generate: ModelVersion[];
  edit: ModelVersion[];
  precisions: Precision[];
}

export type FitVerdict = "recommended" | "fits" | "tight" | "wont_fit";

export interface PrecisionOption {
  precision: Precision;
  verdict: FitVerdict;
  peak_vram_bytes: number;
  headroom_bytes: number;
  rationale: string;
  available: boolean;
}

export interface AdviceRequest {
  mode: Mode;
  model_id: string;
  resolution: { width: number; height: number };
  batch: number;
  loras: { lora_id: string; weight: number }[];
}

export interface Advice {
  recommended: Precision;
  options: PrecisionOption[];
}

// ---- Rewriter (Qwen-VL) -----------------------------------------------------

export interface RewriterModel {
  id: string;
  label: string;
  quant: string;
  backends: Backend[];
  is_default: boolean;
}

export interface RewriterAdviceRequest {
  vl_model: string;
  image_model_id: string;
  precision: Precision;
}

export interface RewriterAdvice {
  co_resident: boolean;
  swap_seconds_estimate: number | null;
  rationale: string;
}

export interface EnhanceRequest {
  mode: Mode;
  prompt: string;
  image_ids: string[];
  vl_model?: string;
}

export interface EnhanceResult {
  original: string;
  enhanced: string;
}

// ---- Resolution -------------------------------------------------------------

export type Orientation = "square" | "portrait" | "landscape";
export type BaseSize = "match" | "512" | "1024" | "1536" | "2048";

export interface ResolutionEnums {
  base_sizes: BaseSize[];
  orientations: Orientation[];
  aspects: Record<Orientation, string[]>;
}

export interface ResolveRequest {
  base: BaseSize;
  orientation: Orientation;
  aspect: string;
  source_dims?: { width: number; height: number } | null;
}

export interface ResolvedResolution {
  width: number;
  height: number;
}

// ---- Integrations -----------------------------------------------------------

export interface Integration {
  provider: IntegrationProvider;
  label: string | null;
  status: "unset" | "valid" | "invalid";
  masked_key: string | null;
  last_validated_at: string | null;
}

export interface IntegrationSet {
  key: string;
  label?: string;
}

// ---- Jobs -------------------------------------------------------------------

export interface JobInput {
  position: number;
  asset_id: string;
}

export interface JobOutput {
  position: number;
  storage_key: string;
  thumb_key: string | null;
  seed: number;
  metadata_json: Record<string, unknown>;
}

export interface JobParams {
  resolution: ResolvedResolution;
  steps: number;
  true_cfg_scale: number;
  guidance_scale: number;
  negative_prompt: string;
  seed: number | null;
  batch: number;
  precision: Precision;
  loras: PromptLora[];
  output_format: "png" | "webp" | "jpeg";
  output_quality?: number;
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
  params_json: JobParams;
  progress: number;
  error: string | null;
  inputs: JobInput[];
  outputs: JobOutput[];
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
}

export interface JobSubmit {
  mode: Mode;
  model_id: string;
  model_revision?: string | null;
  prompt: string;
  enhanced_prompt?: string | null;
  rewriter_model?: string | null;
  precision: Precision;
  input_asset_ids: string[];
  params: JobParams;
}

export type SweepType = "slot" | "seed" | "param";

export interface BatchSubmit extends JobSubmit {
  sweep: {
    type: SweepType;
    // slot: asset_ids per output; seed: list of seeds; param: grid
    slot_asset_ids?: string[];
    seeds?: number[];
    param_grid?: { steps?: number[]; true_cfg_scale?: number[] };
  };
}

export interface BatchResult {
  batch_id: string;
  job_ids: string[];
}

// ---- WebSocket progress messages -------------------------------------------

export interface JobProgressMessage {
  type: "progress";
  job_id: string;
  status: JobStatus;
  step: number;
  total_steps: number;
  eta_seconds: number | null;
  /** data URL or storage key for the throttled latent->RGB preview */
  preview_url?: string | null;
}

export interface JobDoneMessage {
  type: "done";
  job_id: string;
  job: Job;
}

export interface JobErrorMessage {
  type: "error";
  job_id: string;
  error: string;
}

export type JobWsMessage =
  | JobProgressMessage
  | JobDoneMessage
  | JobErrorMessage;
