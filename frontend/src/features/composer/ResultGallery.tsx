import { useState } from "react";
import { Download, Star, ArrowRightToLine, RotateCcw, SplitSquareHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { fileUrl, thumbUrl } from "@/api/client";
import { usePromoteAsset } from "@/api/hooks";
import { useComposer, newUid } from "@/store/composer";
import { useUi } from "@/store/ui";
import { CompareSlider } from "./CompareSlider";
import { toast } from "sonner";
import type { Job } from "@/api/types";

export function ResultGallery({ job }: { job: Job }) {
  const [compare, setCompare] = useState(false);
  const promote = usePromoteAsset();
  const setActiveJob = useUi((s) => s.setActiveJob);

  const beforeKey =
    job.mode === "edit" && job.inputs.length
      ? undefined // resolved server-side; use job input asset thumbs not available here
      : undefined;

  const sendToInput = (storageKey: string, w: number, h: number) => {
    // Switch to edit mode and load this output as an input.
    const c = useComposer.getState();
    c.setMode("edit");
    c.addInputs([
      {
        uid: newUid(),
        origin: "upload",
        asset: {
          id: `out_${storageKey}`,
          scope: "ephemeral",
          storage_key: storageKey,
          thumb_key: null,
          name: "output",
          description: null,
          tags: [],
          width: w,
          height: h,
          format: "png",
          bytes: 0,
          sha256: "",
          source: "output",
          source_job_id: job.id,
          created_at: new Date().toISOString(),
          last_used_at: null,
        },
      },
    ]);
    setActiveJob(null);
    toast.success("Sent to input");
  };

  const reuseSettings = () => {
    const c = useComposer.getState();
    c.setMode(job.mode);
    c.setPrompt(job.prompt);
    c.setEnhancedPrompt(job.enhanced_prompt);
    c.setPrecision(job.precision);
    c.setSteps(job.params_json.steps);
    c.setTrueCfgScale(job.params_json.true_cfg_scale);
    c.setNegativePrompt(job.params_json.negative_prompt);
    c.setLoras(job.params_json.loras);
    toast.success("Settings reused");
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Badge variant="muted">{job.mode}</Badge>
        <Badge variant="muted">{job.precision}</Badge>
        <Badge variant="muted">
          {job.params_json.resolution.width}×{job.params_json.resolution.height}
        </Badge>
        {job.mode === "edit" && job.outputs.length > 0 && (
          <Button
            size="sm"
            variant="ghost"
            className="ml-auto"
            onClick={() => setCompare((v) => !v)}
          >
            <SplitSquareHorizontal className="h-3.5 w-3.5" />
            {compare ? "Grid" : "Compare"}
          </Button>
        )}
      </div>

      <div className={compare ? "" : "grid grid-cols-2 gap-3"}>
        {job.outputs.map((out) => {
          const url = fileUrl(out.storage_key);
          return (
            <div key={out.position} className="group relative overflow-hidden rounded-xl border">
              {compare && beforeKey ? (
                <CompareSlider before={fileUrl(beforeKey)} after={url} />
              ) : (
                <img src={thumbUrl(out.thumb_key) ?? url} alt="output" className="w-full" />
              )}
              <div className="absolute inset-x-0 bottom-0 flex items-center gap-1 bg-gradient-to-t from-black/70 to-transparent p-2 opacity-0 transition-opacity group-hover:opacity-100">
                <a href={url} download className="contents">
                  <Button size="icon" variant="secondary" className="h-7 w-7" title="Download">
                    <Download className="h-3.5 w-3.5" />
                  </Button>
                </a>
                <Button
                  size="icon"
                  variant="secondary"
                  className="h-7 w-7"
                  title="Promote to library"
                  onClick={() =>
                    promote
                      .mutateAsync(`out_${out.storage_key}`)
                      .then(() => toast.success("Promoted to library"))
                      .catch((e) => toast.error((e as Error).message))
                  }
                >
                  <Star className="h-3.5 w-3.5" />
                </Button>
                <Button
                  size="icon"
                  variant="secondary"
                  className="h-7 w-7"
                  title="Send to input"
                  onClick={() =>
                    sendToInput(
                      out.storage_key,
                      job.params_json.resolution.width,
                      job.params_json.resolution.height,
                    )
                  }
                >
                  <ArrowRightToLine className="h-3.5 w-3.5" />
                </Button>
                <span className="ml-auto rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-white">
                  seed {out.seed}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <Button variant="outline" size="sm" onClick={reuseSettings}>
        <RotateCcw className="h-3.5 w-3.5" /> Reuse settings
      </Button>
    </div>
  );
}
