import { useQuery } from "@tanstack/react-query";
import { resolutionApi } from "@/api/endpoints";
import type { ResolutionState } from "@/store/composer";

/** Aspect ratio "W:H" -> numeric. */
function parseAspect(aspect: string): number {
  const [w, h] = aspect.split(":").map(Number);
  if (!w || !h) return 1;
  return w / h;
}

/** Local fallback resolver (mirrors backend rules) used before the API responds. */
export function resolveLocal(
  res: ResolutionState,
  sourceDims?: { width: number; height: number } | null,
): { width: number; height: number } {
  if (res.base === "match") {
    if (sourceDims) {
      return {
        width: snap16(sourceDims.width),
        height: snap16(sourceDims.height),
      };
    }
    return { width: 1024, height: 1024 };
  }
  const base = Number(res.base);
  if (res.orientation === "square") {
    return { width: snap16(base), height: snap16(base) };
  }
  const ratio = parseAspect(res.aspect); // long:short for landscape, etc.
  // base = longer edge
  const longer = base;
  const shorter = base / ratio;
  if (res.orientation === "landscape") {
    return { width: snap16(longer), height: snap16(shorter) };
  }
  // portrait: longer edge is height
  return { width: snap16(shorter), height: snap16(longer) };
}

function snap16(n: number): number {
  return Math.max(16, Math.round(n / 16) * 16);
}

/**
 * Resolve a resolution preset via the backend, falling back to a local
 * computation so the W×H chip is always live.
 */
export function useResolvedResolution(
  res: ResolutionState,
  sourceDims?: { width: number; height: number } | null,
) {
  const local = resolveLocal(res, sourceDims);
  const query = useQuery({
    queryKey: ["resolve", res, sourceDims ?? null],
    queryFn: () =>
      resolutionApi.resolve({
        base: res.base,
        orientation: res.orientation,
        aspect: res.aspect,
        source_dims: sourceDims ?? null,
      }),
    placeholderData: (prev) => prev,
    staleTime: 60_000,
  });
  return query.data ?? local;
}
