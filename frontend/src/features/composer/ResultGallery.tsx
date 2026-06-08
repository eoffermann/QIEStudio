import { useState } from "react";
import { Download, ArrowRightToLine, RotateCcw, SplitSquareHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useUi } from "@/store/ui";
import { applyJobSettings } from "@/lib/jobSettings";
import { CompareSlider } from "./CompareSlider";
import { toast } from "sonner";
import type { Job } from "@/api/types";

export function ResultGallery({ job }: { job: Job }) {
  const [compare, setCompare] = useState(false);
  const setActiveJob = useUi((s) => s.setActiveJob);

  // The "before" image (Edit mode source) is not exposed in the job read model;
  // compare falls back to a grid until a source URL is available.
  const beforeUrl: string | undefined = undefined;

  const w = typeof job.params.width === "number" ? job.params.width : 0;
  const h = typeof job.params.height === "number" ? job.params.height : 0;

  const reuseSettings = () => {
    applyJobSettings(job);
    toast.success("Settings reused");
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Badge variant="muted">{job.mode}</Badge>
        <Badge variant="muted">{job.precision}</Badge>
        {w > 0 && h > 0 && (
          <Badge variant="muted">
            {w}×{h}
          </Badge>
        )}
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
          const url = out.file_url;
          return (
            <div key={out.id} className="group relative overflow-hidden rounded-xl border">
              {compare && beforeUrl ? (
                <CompareSlider before={beforeUrl} after={url} />
              ) : (
                <img src={out.thumb_url ?? url} alt="output" className="w-full" />
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
                  title="Send to input"
                  onClick={() => {
                    setActiveJob(null);
                    toast.info("Open the output from the library to reuse as input");
                  }}
                >
                  <ArrowRightToLine className="h-3.5 w-3.5" />
                </Button>
                {out.seed != null && (
                  <span className="ml-auto rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-white">
                    seed {out.seed}
                  </span>
                )}
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
