// Re-apply a finished job's settings back onto the composer store.
// Reads the flat `params` dict the backend records (job_service.create_job).
import { useComposer } from "@/store/composer";
import type { Job, PromptLora } from "@/api/types";

function num(v: unknown, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

/** Push a job's mode/prompt/precision and recorded run params into the composer. */
export function applyJobSettings(job: Job): void {
  const c = useComposer.getState();
  const p = job.params ?? {};
  c.setMode(job.mode);
  c.setPrompt(job.prompt);
  c.setEnhancedPrompt(job.enhanced_prompt);
  c.setPrecision(job.precision);
  c.setSteps(num(p.num_inference_steps, job.progress_total || 40));
  c.setTrueCfgScale(num(p.true_cfg_scale, 4.0));
  if (typeof p.negative_prompt === "string") c.setNegativePrompt(p.negative_prompt);
  if (Array.isArray(p.loras)) c.setLoras(p.loras as PromptLora[]);
}
