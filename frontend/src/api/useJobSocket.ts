// WebSocket hook subscribing to WS /ws/jobs/{id} for progress + live latent previews.
// Backend (app/services/job_service.py) publishes {type: "snapshot"|"progress"|"status"}.
import { useEffect, useRef, useState } from "react";
import { jobWsUrl } from "./client";
import type { JobStatus, JobWsMessage } from "./types";

export interface JobSocketState {
  connected: boolean;
  status: JobStatus | null;
  step: number;
  totalSteps: number;
  progress: number;
  /** data:image/png;base64,... live latent preview */
  previewUrl: string | null;
  /** True once a terminal "done" status arrived (outputs ready to refetch). */
  done: boolean;
  device: string | null;
  error: string | null;
}

const INITIAL: JobSocketState = {
  connected: false,
  status: null,
  step: 0,
  totalSteps: 0,
  progress: 0,
  previewUrl: null,
  done: false,
  device: null,
  error: null,
};

const TERMINAL: JobStatus[] = ["done", "error", "canceled"];

/**
 * Subscribe to a job's progress stream. Pass `null` to disconnect.
 * Auto-reconnects (with backoff) while the job is not terminal.
 */
export function useJobSocket(jobId: string | null): JobSocketState {
  const [state, setState] = useState<JobSocketState>(INITIAL);
  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<number>(0);
  const closedRef = useRef<boolean>(false);

  useEffect(() => {
    if (!jobId) {
      setState(INITIAL);
      return;
    }
    closedRef.current = false;
    setState(INITIAL);

    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      if (closedRef.current) return;
      const ws = new WebSocket(jobWsUrl(jobId));
      socketRef.current = ws;

      ws.onopen = () => {
        retryRef.current = 0;
        setState((s) => ({ ...s, connected: true }));
      };

      ws.onmessage = (ev) => {
        let msg: JobWsMessage;
        try {
          msg = JSON.parse(ev.data as string) as JobWsMessage;
        } catch {
          return;
        }
        setState((s) => {
          switch (msg.type) {
            case "snapshot":
              return {
                ...s,
                status: msg.status,
                step: msg.step,
                totalSteps: msg.total,
                progress: msg.progress,
                done: s.done || msg.status === "done",
              };
            case "progress":
              return {
                ...s,
                status: s.status ?? "running",
                step: msg.step,
                totalSteps: msg.total,
                progress: msg.progress,
                previewUrl: msg.preview ?? s.previewUrl,
              };
            case "status": {
              const terminal = TERMINAL.includes(msg.status);
              if (terminal) closedRef.current = true;
              return {
                ...s,
                status: msg.status,
                device: msg.device ?? s.device,
                done: s.done || msg.status === "done",
                error: msg.error ?? s.error,
                step: msg.status === "done" ? s.totalSteps || s.step : s.step,
              };
            }
            default:
              return s;
          }
        });
      };

      ws.onclose = () => {
        setState((s) => ({ ...s, connected: false }));
        if (closedRef.current) return;
        // Reconnect with capped backoff while the job is still in flight.
        retryRef.current = Math.min(retryRef.current + 1, 5);
        const delay = 500 * 2 ** (retryRef.current - 1);
        reconnectTimer = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connect();

    return () => {
      closedRef.current = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [jobId]);

  return state;
}
