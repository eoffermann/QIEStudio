// Typed fetch client for the QIE Studio backend.
// All endpoints are relative; Vite proxies /api and /ws to :8000 in dev,
// and in production the backend serves both the SPA and the API on one origin.

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, message: string, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined | null>;
  signal?: AbortSignal;
  /** Pass a FormData body directly (skips JSON serialization). */
  formData?: FormData;
}

function buildQuery(
  query?: Record<string, string | number | boolean | undefined | null>,
): string {
  if (!query) return "";
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  }
  const s = params.toString();
  return s ? `?${s}` : "";
}

export async function api<T>(
  path: string,
  opts: RequestOptions = {},
): Promise<T> {
  const { method = "GET", body, query, signal, formData } = opts;
  const url = `/api${path}${buildQuery(query)}`;

  const headers: Record<string, string> = {};
  let payload: BodyInit | undefined;

  if (formData) {
    payload = formData;
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }

  const res = await fetch(url, { method, headers, body: payload, signal });

  if (!res.ok) {
    let detail: unknown;
    let message = `${res.status} ${res.statusText}`;
    try {
      detail = await res.json();
      if (
        detail &&
        typeof detail === "object" &&
        "detail" in detail &&
        typeof (detail as { detail: unknown }).detail === "string"
      ) {
        message = (detail as { detail: string }).detail;
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, message, detail);
  }

  if (res.status === 204) return undefined as T;
  const ct = res.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) return (await res.json()) as T;
  return (await res.text()) as unknown as T;
}

/** Backend URL for an asset's full binary (DESIGN §7: GET /api/assets/{id}/file). */
export function assetFileUrl(id: string): string {
  return `/api/assets/${encodeURIComponent(id)}/file`;
}

/** Backend URL for an asset's WebP thumbnail (GET /api/assets/{id}/thumb).
 *  Pass the asset's `thumb_key` (or any truthy flag); returns undefined if absent. */
export function assetThumbUrl(
  id: string,
  hasThumb: string | boolean | null | undefined,
): string | undefined {
  if (!hasThumb) return undefined;
  return `/api/assets/${encodeURIComponent(id)}/thumb`;
}

/** Build the WebSocket URL for job progress, honoring the current origin. */
export function jobWsUrl(jobId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/jobs/${encodeURIComponent(jobId)}`;
}
