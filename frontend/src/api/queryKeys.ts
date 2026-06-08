// Centralized TanStack Query keys.
import type { AdviceRequest, Mode } from "./types";

export const qk = {
  device: ["device"] as const,
  models: ["models"] as const,
  advice: (req: AdviceRequest) => ["advice", req] as const,
  rewriterModels: ["rewriter", "models"] as const,
  resolutionEnums: ["resolution", "enums"] as const,
  integrations: ["integrations"] as const,
  assets: (filter?: { scope?: string; tag?: string; q?: string; source?: string }) =>
    ["assets", filter ?? {}] as const,
  prompts: (filter?: { q?: string; tag?: string; mode?: Mode | "any" }) =>
    ["prompts", filter ?? {}] as const,
  loras: ["loras"] as const,
  jobs: (filter?: Record<string, unknown>) => ["jobs", filter ?? {}] as const,
  job: (id: string) => ["job", id] as const,
  queue: ["jobs", "queue"] as const,
};
