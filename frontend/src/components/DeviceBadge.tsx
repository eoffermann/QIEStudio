import { Cpu } from "lucide-react";
import { useDevice } from "@/api/hooks";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { gbFromMb } from "@/lib/utils";

const BACKEND_LABEL: Record<string, string> = {
  cuda: "CUDA",
  rocm: "ROCm",
  mps: "MPS",
  cpu: "CPU",
};

export function DeviceBadge() {
  const { data, isLoading, isError } = useDevice();

  if (isLoading) {
    return (
      <Badge variant="muted" className="gap-1.5">
        <Cpu className="h-3.5 w-3.5" /> detecting…
      </Badge>
    );
  }
  if (isError || !data) {
    return (
      <Badge variant="danger" className="gap-1.5">
        <Cpu className="h-3.5 w-3.5" /> no device
      </Badge>
    );
  }

  const freePct = data.total_vram_mb
    ? Math.round((data.free_vram_mb / data.total_vram_mb) * 100)
    : 0;
  const cc = data.compute_capability
    ? `${data.compute_capability[0]}.${data.compute_capability[1]}`
    : null;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant="secondary" className="gap-1.5">
          <Cpu className="h-3.5 w-3.5 text-primary" />
          {BACKEND_LABEL[data.backend] ?? data.backend}
          <span className="text-muted-foreground">·</span>
          {gbFromMb(data.free_vram_mb)} free
        </Badge>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">
        <div className="space-y-0.5">
          <div className="font-medium">{data.name}</div>
          {cc && <div>Compute {cc}</div>}
          <div>
            VRAM {gbFromMb(data.free_vram_mb)} / {gbFromMb(data.total_vram_mb)} ({freePct}% free)
          </div>
          <div>RAM {gbFromMb(data.total_ram_mb)}</div>
          <div className="pt-1 text-muted-foreground">
            fp8 {data.supports_fp8_native ? "✓" : "✗ (emulated)"} · int4{" "}
            {data.supports_int4_nunchaku ? "✓" : "✗"}
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
