import { Ruler } from "lucide-react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { useComposer } from "@/store/composer";
import { useResolvedResolution } from "./useResolvedResolution";
import type { BaseSize, Orientation } from "@/api/types";

const BASE_SIZES: { value: BaseSize; label: string; editOnly?: boolean }[] = [
  { value: "match", label: "Match source", editOnly: true },
  { value: "512", label: "512" },
  { value: "1024", label: "1024" },
  { value: "1536", label: "1536" },
  { value: "2048", label: "2048" },
];

const ORIENTATIONS: Orientation[] = ["square", "portrait", "landscape"];

const ASPECTS: Record<Orientation, string[]> = {
  square: ["1:1"],
  portrait: ["4:5", "3:4", "2:3", "9:16"],
  landscape: ["5:4", "4:3", "3:2", "16:9"],
};

export function ResolutionPicker() {
  const mode = useComposer((s) => s.mode);
  const resolution = useComposer((s) => s.resolution);
  const setResolution = useComposer((s) => s.setResolution);
  const inputs = useComposer((s) => s.inputs);

  const firstDims = inputs[0]
    ? { width: inputs[0].asset.width, height: inputs[0].asset.height }
    : null;
  const wh = useResolvedResolution(resolution, firstDims);

  const aspectOptions = ASPECTS[resolution.orientation];
  const isMatch = resolution.base === "match";

  // Proportional thumbnail (max 64px on the long edge).
  const long = Math.max(wh.w, wh.h);
  const thumbW = (wh.w / long) * 64;
  const thumbH = (wh.h / long) * 64;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        <Ruler className="h-3.5 w-3.5" /> Resolution
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <Label>Base</Label>
          <Select
            value={resolution.base}
            onValueChange={(v) => setResolution({ base: v as BaseSize })}
          >
            <SelectTrigger className="h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {BASE_SIZES.filter((b) => mode === "edit" || !b.editOnly).map((b) => (
                <SelectItem key={b.value} value={b.value}>
                  {b.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1">
          <Label>Orientation</Label>
          <Select
            value={resolution.orientation}
            disabled={isMatch}
            onValueChange={(v) =>
              setResolution({
                orientation: v as Orientation,
                aspect: ASPECTS[v as Orientation][0],
              })
            }
          >
            <SelectTrigger className="h-9 capitalize">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ORIENTATIONS.map((o) => (
                <SelectItem key={o} value={o} className="capitalize">
                  {o}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {resolution.orientation !== "square" && !isMatch && (
        <div className="space-y-1">
          <Label>Aspect ratio</Label>
          <div className="flex flex-wrap gap-1.5">
            {aspectOptions.map((a) => (
              <button
                key={a}
                onClick={() => setResolution({ aspect: a })}
                className={
                  "rounded-lg border px-2.5 py-1 text-xs transition-colors " +
                  (resolution.aspect === a
                    ? "border-primary bg-primary/10 text-foreground"
                    : "border-border text-muted-foreground hover:border-primary/40")
                }
              >
                {a}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex items-center gap-3 rounded-xl border bg-card/60 p-3">
        <div
          className="shrink-0 rounded-md bg-primary/20 ring-1 ring-primary/40"
          style={{ width: `${thumbW}px`, height: `${thumbH}px` }}
        />
        <div>
          <Badge variant="default" className="font-mono">
            {wh.w} × {wh.h}
          </Badge>
          <div className="mt-1 text-[11px] text-muted-foreground">
            {isMatch ? "matched to first input" : "snapped to ×16"}
          </div>
        </div>
      </div>
    </div>
  );
}
