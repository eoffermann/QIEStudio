// TanStack Query hooks over the endpoint wrappers.
import {
  useMutation,
  useQuery,
  useQueryClient,
  keepPreviousData,
} from "@tanstack/react-query";
import {
  assetsApi,
  deviceApi,
  integrationsApi,
  jobsApi,
  lorasApi,
  modelsApi,
  promptsApi,
  resolutionApi,
  rewriterApi,
} from "./endpoints";
import { qk } from "./queryKeys";
import type {
  AdviceRequest,
  BatchSubmit,
  IntegrationProvider,
  IntegrationSet,
  JobStatus,
  JobSubmit,
  LoraImportRequest,
  Mode,
  PromptCreate,
} from "./types";

// ---- Device / models --------------------------------------------------------
export function useDevice() {
  return useQuery({ queryKey: qk.device, queryFn: deviceApi.get, staleTime: 30_000 });
}

export function useModels() {
  return useQuery({ queryKey: qk.models, queryFn: modelsApi.catalog, staleTime: Infinity });
}

export function useAdvice(req: AdviceRequest, enabled = true) {
  return useQuery({
    queryKey: qk.advice(req),
    queryFn: () => modelsApi.advise(req),
    enabled,
    placeholderData: keepPreviousData,
  });
}

// ---- Rewriter ---------------------------------------------------------------
export function useRewriterModels() {
  return useQuery({
    queryKey: qk.rewriterModels,
    queryFn: rewriterApi.models,
    staleTime: Infinity,
  });
}

export function useEnhance() {
  return useMutation({ mutationFn: rewriterApi.enhance });
}

// ---- Resolution -------------------------------------------------------------
export function useResolutionEnums() {
  return useQuery({
    queryKey: qk.resolutionEnums,
    queryFn: resolutionApi.enums,
    staleTime: Infinity,
  });
}

// ---- Assets -----------------------------------------------------------------
export function useAssets(filter?: {
  scope?: string;
  tag?: string;
  q?: string;
  source?: string;
}) {
  return useQuery({
    queryKey: qk.assets(filter),
    queryFn: () => assetsApi.list(filter),
    placeholderData: keepPreviousData,
  });
}

export function useUploadAssets() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (files: File[]) => assetsApi.upload(files),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["assets"] }),
  });
}

export function usePromoteAsset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => assetsApi.promote(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["assets"] }),
  });
}

export function usePatchAsset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: Parameters<typeof assetsApi.patch>[1] }) =>
      assetsApi.patch(id, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["assets"] }),
  });
}

export function useDeleteAsset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => assetsApi.remove(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["assets"] }),
  });
}

// ---- Prompts ----------------------------------------------------------------
export function usePrompts(filter?: { q?: string; tag?: string; mode?: Mode | "any" }) {
  return useQuery({
    queryKey: qk.prompts(filter),
    queryFn: () => promptsApi.list(filter),
    placeholderData: keepPreviousData,
  });
}

export function useCreatePrompt() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: PromptCreate) => promptsApi.create(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["prompts"] }),
  });
}

export function usePatchPrompt() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: Partial<PromptCreate> }) =>
      promptsApi.patch(id, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["prompts"] }),
  });
}

export function useDuplicatePrompt() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => promptsApi.duplicate(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["prompts"] }),
  });
}

export function useDeletePrompt() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => promptsApi.remove(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["prompts"] }),
  });
}

// ---- LoRAs ------------------------------------------------------------------
export function useLoras() {
  return useQuery({ queryKey: qk.loras, queryFn: lorasApi.list });
}

export function useUploadLora() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ file, name }: { file: File; name?: string }) =>
      lorasApi.upload(file, { name }),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.loras }),
  });
}

export function useImportLora() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: LoraImportRequest) => lorasApi.import(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.loras }),
  });
}

export function usePatchLora() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: Parameters<typeof lorasApi.patch>[1] }) =>
      lorasApi.patch(id, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.loras }),
  });
}

export function useDeleteLora() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => lorasApi.remove(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.loras }),
  });
}

// ---- Integrations -----------------------------------------------------------
export function useIntegrations() {
  return useQuery({ queryKey: qk.integrations, queryFn: integrationsApi.list });
}

export function useSetIntegration() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ provider, body }: { provider: IntegrationProvider; body: IntegrationSet }) =>
      integrationsApi.set(provider, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.integrations }),
  });
}

export function useDeleteIntegration() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (provider: IntegrationProvider) => integrationsApi.remove(provider),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.integrations }),
  });
}

// ---- Jobs -------------------------------------------------------------------
export function useJobHistory(filter?: {
  status?: JobStatus;
  mode?: Mode;
  batch_id?: string;
  q?: string;
}) {
  return useQuery({
    queryKey: qk.jobs(filter),
    queryFn: () => jobsApi.history(filter),
    placeholderData: keepPreviousData,
  });
}

export function useJob(id: string | null) {
  return useQuery({
    queryKey: qk.job(id ?? ""),
    queryFn: () => jobsApi.get(id as string),
    enabled: !!id,
  });
}

export function useSubmitJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: JobSubmit) => jobsApi.submit(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export function useSubmitBatch() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: BatchSubmit) => jobsApi.submitBatch(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => jobsApi.cancel(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export function useDeleteJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => jobsApi.remove(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

/** Live queue snapshot (running + pending order); polls while there is work. */
export function useQueue() {
  return useQuery({
    queryKey: qk.queue,
    queryFn: jobsApi.queue,
    refetchInterval: (query) => {
      const d = query.state.data;
      return d && (d.running || d.pending.length) ? 1500 : false;
    },
  });
}

export function useReorderQueue() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (job_ids: string[]) => jobsApi.reorder(job_ids),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}
