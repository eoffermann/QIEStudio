import { useCallback, useEffect } from "react";
import { Sparkles, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ModeToggle } from "./ModeToggle";
import { InputImagesPanel } from "./InputImagesPanel";
import { PromptEditor } from "./PromptEditor";
import { SettingsRail } from "./SettingsRail";
import { JobProgressPanel } from "./JobProgressPanel";
import { useComposer, effectiveSeed } from "@/store/composer";
import { useUi } from "@/store/ui";
import { useSubmitJob, useSubmitBatch, useDevice } from "@/api/hooks";
import { useResolvedResolution } from "./useResolvedResolution";
import { gbFromMb } from "@/lib/utils";
import { toast } from "sonner";
import type { JobSubmit, BatchSubmit } from "@/api/types";

export function ComposerPage() {
  const mode = useComposer((s) => s.mode);
  const submitJob = useSubmitJob();
  const submitBatch = useSubmitBatch();
  const setActiveJob = useUi((s) => s.setActiveJob);
  const setActiveBatch = useUi((s) => s.setActiveBatch);
  const { data: device } = useDevice();

  const resolution = useComposer((s) => s.resolution);
  const inputs = useComposer((s) => s.inputs);
  const firstDims = inputs[0]
    ? { width: inputs[0].asset.width, height: inputs[0].asset.height }
    : null;
  const wh = useResolvedResolution(resolution, firstDims);

  const generate = useCallback(async () => {
    const s = useComposer.getState();
    if (!s.modelId) {
      toast.error("Select a model");
      return;
    }
    if (!s.prompt.trim()) {
      toast.error("Write a prompt");
      return;
    }
    if (s.mode === "edit" && s.inputs.length === 0) {
      toast.error("Edit mode needs at least one input image");
      return;
    }

    const seed = effectiveSeed(s);
    // JobSubmit is FLAT (DESIGN §7 / app/schemas/jobs.py) — no nested params object.
    const base: JobSubmit = {
      mode: s.mode,
      model_id: s.modelId,
      prompt: s.prompt,
      enhanced_prompt: s.enhancedPrompt,
      negative_prompt: s.negativePrompt,
      precision: s.precision,
      input_asset_ids: s.inputs.map((i) => i.asset.id),
      loras: s.loras,
      resolution: { width: wh.w, height: wh.h },
      num_inference_steps: s.steps,
      true_cfg_scale: s.trueCfgScale,
      guidance_scale: s.guidanceScale,
      seed: s.lockSeed ? seed : null,
      batch: s.batch,
      output_format: s.outputFormat,
    };

    try {
      if (s.batch > 1) {
        const batchBody: BatchSubmit = {
          base,
          sweep: {
            kind: "seed",
            seeds: Array.from({ length: s.batch }, () =>
              Math.floor(Math.random() * 2 ** 31),
            ),
          },
        };
        const res = await submitBatch.mutateAsync(batchBody);
        setActiveBatch(res.batch_id);
        setActiveJob(res.job_ids[0] ?? null);
        toast.success(`Batch of ${res.job_ids.length} queued`);
      } else {
        const res = await submitJob.mutateAsync(base);
        setActiveJob(res.job_id);
        toast.success("Job queued");
      }
    } catch (e) {
      toast.error(`Submit failed: ${(e as Error).message}`);
    }
  }, [wh, submitJob, submitBatch, setActiveJob, setActiveBatch]);

  // G / ⌘↩ shortcut + command palette dispatch this event.
  useEffect(() => {
    const handler = () => void generate();
    window.addEventListener("qie:generate", handler);
    return () => window.removeEventListener("qie:generate", handler);
  }, [generate]);

  const submitting = submitJob.isPending || submitBatch.isPending;
  const freeVram = device ? gbFromMb(device.free_vram_mb) : "—";

  return (
    <div className="flex h-full flex-col">
      {/* Header bar */}
      <div className="flex items-center justify-between gap-4 border-b px-6 py-4">
        <ModeToggle />
        <div className="flex items-center gap-3">
          <div className="hidden text-right text-xs text-muted-foreground sm:block">
            <div className="font-mono text-foreground">
              {wh.w} × {wh.h}
            </div>
            <div>{freeVram} VRAM free</div>
          </div>
          <Button size="lg" variant="glow" onClick={() => generate()} disabled={submitting}>
            {submitting ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Sparkles className="h-5 w-5" />
            )}
            Generate
            <Badge variant="secondary" className="ml-1 font-mono">
              G
            </Badge>
          </Button>
        </div>
      </div>

      {/* Body */}
      <div className="grid min-h-0 flex-1 gap-4 overflow-hidden p-6 lg:grid-cols-[minmax(260px,320px)_minmax(0,1fr)_minmax(300px,360px)]">
        {/* Left: inputs (edit only) */}
        {mode === "edit" ? (
          <div className="min-h-0 overflow-hidden">
            <InputImagesPanel />
          </div>
        ) : (
          <div className="hidden lg:block" />
        )}

        {/* Center: prompt + progress */}
        <div className="flex min-h-0 flex-col gap-4 overflow-y-auto">
          <div className="min-h-[280px] flex-1">
            <PromptEditor />
          </div>
          <JobProgressPanel />
        </div>

        {/* Right: settings */}
        <div className="min-h-0 overflow-hidden">
          <SettingsRail />
        </div>
      </div>
    </div>
  );
}
