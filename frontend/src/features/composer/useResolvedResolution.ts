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
): { w: number; h: number } {
  if (res.base === "match") {
    if (sourceDims) {
      return {
        w: snap16(sourceDims.width),
        h: snap16(sourceDims.height),
      };
    }
    return { w: 1024, h: 1024 };
  }
  const base = Number(res.base);
  if (res.orientation === "square") {
    return { w: snap16(base), h: snap16(base) };
  }
  const ratio = parseAspect(res.aspect); // long:short for landscape, etc.
  // base = longer edge
  const longer = base;
  const shorter = base / ratio;
  if (res.orientation === "landscape") {
    return { w: snap16(longer), h: snap16(shorter) };
  }
  // portrait: longer edge is height
  return { w: snap16(shorter), h: snap16(longer) };
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
        base: res.base === "match" ? "match" : Number(res.base),
        orientation: res.orientation,
        aspect: res.aspect,
        source_dims: sourceDims ? [sourceDims.width, sourceDims.height] : null,
      }),
    placeholderData: (prev) => prev,
    staleTime: 60_000,
  });
  return query.data ?? local;
}
