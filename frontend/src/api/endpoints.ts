// Thin endpoint wrappers grouped by resource (DESIGN §7).
import { api } from "./client";
import type {
  Advice,
  AdviceRequest,
  Asset,
  BatchResult,
  BatchSubmit,
  DeviceInfo,
  EnhanceRequest,
  EnhanceResult,
  Integration,
  IntegrationProvider,
  IntegrationSet,
  Job,
  JobStatus,
  JobSubmit,
  Lora,
  LoraImportRequest,
  LoraSearchResult,
  LoraSource,
  Mode,
  ModelCatalog,
  Prompt,
  PromptCreate,
  ResolutionEnums,
  ResolveRequest,
  ResolvedResolution,
  RewriterAdvice,
  RewriterAdviceRequest,
  RewriterModel,
} from "./types";

// ---- Assets -----------------------------------------------------------------
export const assetsApi = {
  list: (q?: { scope?: string; tag?: string; q?: string; source?: string }) =>
    api<Asset[]>("/assets", { query: q }),
  upload: (files: File[]) => {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    return api<Asset[]>("/assets/upload", { method: "POST", formData: fd });
  },
  promote: (id: string) =>
    api<Asset>(`/assets/${id}/promote`, { method: "POST" }),
  patch: (id: string, body: Partial<Pick<Asset, "name" | "description" | "tags" | "scope">>) =>
    api<Asset>(`/assets/${id}`, { method: "PATCH", body }),
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
  models: () => api<RewriterModel[]>("/rewriter/models"),
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

// ---- Jobs -------------------------------------------------------------------
export const jobsApi = {
  submit: (body: JobSubmit) => api<Job>("/jobs", { method: "POST", body }),
  submitBatch: (body: BatchSubmit) =>
    api<BatchResult>("/jobs/batch", { method: "POST", body }),
  get: (id: string) => api<Job>(`/jobs/${id}`),
  cancel: (id: string) => api<Job>(`/jobs/${id}/cancel`, { method: "POST" }),
  history: (q?: {
    status?: JobStatus;
    mode?: Mode;
    batch_id?: string;
    q?: string;
  }) => api<Job[]>("/jobs", { query: q }),
};
