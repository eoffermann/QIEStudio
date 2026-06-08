// Thin endpoint wrappers grouped by resource (DESIGN §7).
import { api } from "./client";
import type {
  Advice,
  AdviceRequest,
  Asset,
  BatchSubmit,
  BatchSubmitResponse,
  DeviceInfo,
  EnhanceRequest,
  EnhanceResult,
  Integration,
  IntegrationProvider,
  IntegrationSet,
  Job,
  JobStatus,
  JobSubmit,
  JobSubmitResponse,
  Lora,
  LoraImportRequest,
  LoraSearchResult,
  LoraSource,
  Mode,
  ModelCatalog,
  Prompt,
  PromptCreate,
  QueueState,
  Recipe,
  RecipeCreate,
  ResolutionEnums,
  ResolveRequest,
  ResolvedResolution,
  RewriterAdvice,
  RewriterAdviceRequest,
  RewriterModelsResponse,
  ToolResult,
} from "./types";

// ---- Assets -----------------------------------------------------------------
export const assetsApi = {
  list: (q?: { scope?: string; tag?: string; q?: string; source?: string }) =>
    api<Asset[]>("/assets", { query: q }),
  get: (id: string) => api<Asset>(`/assets/${id}`),
  upload: (files: File[]) => {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    // Backend returns { assets: AssetRead[] }.
    return api<{ assets: Asset[] }>("/assets/upload", {
      method: "POST",
      formData: fd,
    }).then((r) => r.assets);
  },
  promote: (id: string) =>
    api<Asset>(`/assets/${id}/promote`, { method: "POST" }),
  patch: (
    id: string,
    body: Partial<Pick<Asset, "name" | "description" | "tags" | "collection">>,
  ) => api<Asset>(`/assets/${id}`, { method: "PATCH", body }),
  remove: (id: string) => api<void>(`/assets/${id}`, { method: "DELETE" }),
};

// ---- Prompts ----------------------------------------------------------------
export const promptsApi = {
  list: (q?: { q?: string; tag?: string; mode?: Mode | "any" }) =>
    api<Prompt[]>("/prompts", { query: q }),
  create: (body: PromptCreate) =>
    api<Prompt>("/prompts", { method: "POST", body }),
  patch: (id: string, body: Partial<PromptCreate>) =>
    api<Prompt>(`/prompts/${id}`, { method: "PATCH", body }),
  duplicate: (id: string) =>
    api<Prompt>(`/prompts/${id}/duplicate`, { method: "POST" }),
  remove: (id: string) => api<void>(`/prompts/${id}`, { method: "DELETE" }),
};

// ---- LoRAs ------------------------------------------------------------------
export const lorasApi = {
  list: () => api<Lora[]>("/loras"),
  upload: (file: File, meta?: { name?: string }) => {
    const fd = new FormData();
    fd.append("file", file);
    if (meta?.name) fd.append("name", meta.name);
    return api<Lora>("/loras/upload", { method: "POST", formData: fd });
  },
  import: (body: LoraImportRequest) =>
    api<Lora>("/loras/import", { method: "POST", body }),
  search: (source: LoraSource, q: string) =>
    api<LoraSearchResult[]>("/loras/search", { query: { source, q } }),
  patch: (id: string, body: Partial<Lora>) =>
    api<Lora>(`/loras/${id}`, { method: "PATCH", body }),
  remove: (id: string) => api<void>(`/loras/${id}`, { method: "DELETE" }),
};

// ---- Device / models --------------------------------------------------------
export const deviceApi = {
  get: () => api<DeviceInfo>("/device"),
};

export const modelsApi = {
  catalog: () => api<ModelCatalog>("/models"),
  advise: (body: AdviceRequest) =>
    api<Advice>("/models/advise", { method: "POST", body }),
};

// ---- Rewriter ---------------------------------------------------------------
export const rewriterApi = {
  models: () => api<RewriterModelsResponse>("/rewriter/models"),
  advise: (body: RewriterAdviceRequest) =>
    api<RewriterAdvice>("/rewriter/advise", { method: "POST", body }),
  enhance: (body: EnhanceRequest) =>
    api<EnhanceResult>("/rewriter/enhance", { method: "POST", body }),
};

// ---- Resolution -------------------------------------------------------------
export const resolutionApi = {
  enums: () => api<ResolutionEnums>("/presets/resolution"),
  resolve: (body: ResolveRequest) =>
    api<ResolvedResolution>("/presets/resolution/resolve", {
      method: "POST",
      body,
    }),
};

// ---- Integrations -----------------------------------------------------------
export const integrationsApi = {
  list: () => api<Integration[]>("/integrations"),
  set: (provider: IntegrationProvider, body: IntegrationSet) =>
    api<Integration>(`/integrations/${provider}`, { method: "PUT", body }),
  remove: (provider: IntegrationProvider) =>
    api<void>(`/integrations/${provider}`, { method: "DELETE" }),
};

// ---- Recipes ----------------------------------------------------------------
export const recipesApi = {
  list: (q?: { q?: string; tag?: string; mode?: Mode }) =>
    api<Recipe[]>("/recipes", { query: q }),
  create: (body: RecipeCreate) =>
    api<Recipe>("/recipes", { method: "POST", body }),
  get: (id: string) => api<Recipe>(`/recipes/${id}`),
  patch: (id: string, body: Partial<RecipeCreate>) =>
    api<Recipe>(`/recipes/${id}`, { method: "PATCH", body }),
  remove: (id: string) => api<void>(`/recipes/${id}`, { method: "DELETE" }),
  instantiate: (id: string) =>
    api<JobSubmit>(`/recipes/${id}/instantiate`, { method: "POST" }),
};

// ---- Tools ------------------------------------------------------------------
export const toolsApi = {
  backgroundRemoval: (assetId: string) =>
    api<ToolResult>("/tools/background-removal/json", {
      method: "POST",
      body: { asset_id: assetId },
    }),
  upscale: (assetId: string, opts?: { scale?: number; max_long_edge?: number }) =>
    api<ToolResult>("/tools/upscale/json", {
      method: "POST",
      body: { asset_id: assetId, scale: opts?.scale, max_long_edge: opts?.max_long_edge },
    }),
};

// ---- Jobs -------------------------------------------------------------------
export const jobsApi = {
  submit: (body: JobSubmit) =>
    api<JobSubmitResponse>("/jobs", { method: "POST", body }),
  submitBatch: (body: BatchSubmit) =>
    api<BatchSubmitResponse>("/jobs/batch", { method: "POST", body }),
  get: (id: string) => api<Job>(`/jobs/${id}`),
  cancel: (id: string) =>
    api<{ canceled: boolean }>(`/jobs/${id}/cancel`, { method: "POST" }),
  remove: (id: string) =>
    api<{ deleted: boolean }>(`/jobs/${id}`, { method: "DELETE" }),
  queue: () => api<QueueState>("/jobs/queue"),
  reorder: (job_ids: string[]) =>
    api<QueueState>("/jobs/reorder", { method: "POST", body: { job_ids } }),
  history: (q?: {
    status?: JobStatus;
    mode?: Mode;
    batch_id?: string;
    q?: string;
  }) => api<Job[]>("/jobs", { query: q }),
};
