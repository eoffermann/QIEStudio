import { Cpu } from "lucide-react";
import { useDevice } from "@/api/hooks";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { gb } from "@/lib/utils";

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

  const freePct = data.total_vram_bytes
    ? Math.round((data.free_vram_bytes / data.total_vram_bytes) * 100)
    : 0;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant="secondary" className="gap-1.5">
          <Cpu className="h-3.5 w-3.5 text-primary" />
          {BACKEND_LABEL[data.backend] ?? data.backend}
          <span className="text-muted-foreground">·</span>
          {gb(data.free_vram_bytes)} free
        </Badge>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">
        <div className="space-y-0.5">
          <div className="font-medium">{data.device_name}</div>
          {data.compute_capability && <div>Compute {data.compute_capability}</div>}
          <div>
            VRAM {gb(data.free_vram_bytes)} / {gb(data.total_vram_bytes)} ({freePct}% free)
          </div>
          <div>RAM {gb(data.total_ram_bytes)}</div>
          <div className="pt-1 text-muted-foreground">
            fp8 {data.supports_fp8 ? "✓" : "✗"} · int4 {data.supports_int4 ? "✓" : "✗"}
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
