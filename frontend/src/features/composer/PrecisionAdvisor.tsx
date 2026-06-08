import { Loader2, Cpu } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn, gbFromMb } from "@/lib/utils";
import { useAdvice } from "@/api/hooks";
import { useComposer } from "@/store/composer";
import { useResolvedResolution } from "./useResolvedResolution";
import type { FitVerdict, Precision, PrecisionOption } from "@/api/types";

const VERDICT: Record<
  FitVerdict,
  { label: string; variant: "success" | "default" | "warning" | "danger" | "muted" }
> = {
  recommended: { label: "Recommended", variant: "success" },
  fits: { label: "Fits", variant: "default" },
  tight: { label: "Tight", variant: "warning" },
  wont_fit: { label: "Won't fit", variant: "danger" },
  unavailable: { label: "Unavailable", variant: "muted" },
};

const PRECISION_LABEL: Record<Precision, string> = {
  bf16: "bf16 (full)",
  fp8: "fp8 e4m3fn",
  int4: "SVDQuant int4",
};

export function PrecisionAdvisor() {
  const mode = useComposer((s) => s.mode);
  const modelId = useComposer((s) => s.modelId);
  const precision = useComposer((s) => s.precision);
  const setPrecision = useComposer((s) => s.setPrecision);
  const loras = useComposer((s) => s.loras);
  const batch = useComposer((s) => s.batch);
  const resolution = useComposer((s) => s.resolution);
  const inputs = useComposer((s) => s.inputs);

  const firstDims = inputs[0]
    ? { width: inputs[0].asset.width, height: inputs[0].asset.height }
    : null;
  const wh = useResolvedResolution(resolution, firstDims);

  const advice = useAdvice(
    {
      mode,
      longer_edge: Math.max(wh.w, wh.h),
      batch,
      loras,
    },
    !!modelId,
  );

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        <Cpu className="h-3.5 w-3.5" /> Precision
        {advice.isFetching && <Loader2 className="h-3 w-3 animate-spin" />}
      </div>

      {!modelId && (
        <p className="text-xs text-muted-foreground">Select a model first.</p>
      )}

      <div className="space-y-1.5">
        {(advice.data?.options ?? FALLBACK_OPTIONS).map((opt) => {
          const verdict = VERDICT[opt.status] ?? VERDICT.fits;
          const active = precision === opt.precision;
          const disabled = !opt.available;
          return (
            <button
              key={opt.precision}
              disabled={disabled}
              onClick={() => setPrecision(opt.precision)}
              className={cn(
                "w-full rounded-xl border p-2.5 text-left transition-colors",
                active
                  ? "border-primary bg-primary/10"
                  : "border-border hover:border-primary/40 hover:bg-accent/40",
                disabled && "cursor-not-allowed opacity-40",
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium">
                  {PRECISION_LABEL[opt.precision]}
                </span>
                <Badge variant={verdict.variant}>{verdict.label}</Badge>
              </div>
              <div className="mt-1 flex items-center justify-between text-xs text-muted-foreground">
                <span>{opt.rationale}</span>
              </div>
              {opt.est_peak_vram_mb > 0 && (
                <div className="mt-1 text-[11px] text-muted-foreground">
                  ~{gbFromMb(opt.est_peak_vram_mb)} peak · {gbFromMb(opt.headroom_mb)} headroom
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// Shown before the advisor responds / when no device info is available.
const FALLBACK_OPTIONS: PrecisionOption[] = [
  {
    precision: "bf16",
    status: "fits",
    est_peak_vram_mb: 0,
    headroom_mb: 0,
    rationale: "Highest quality, highest VRAM.",
    available: true,
    caveats: [],
  },
  {
    precision: "fp8",
    status: "fits",
    est_peak_vram_mb: 0,
    headroom_mb: 0,
    rationale: "~half the weights, near-bf16 quality.",
    available: true,
    caveats: [],
  },
  {
    precision: "int4",
    status: "fits",
    est_peak_vram_mb: 0,
    headroom_mb: 0,
    rationale: "Lowest VRAM (CUDA only).",
    available: true,
    caveats: [],
  },
];
