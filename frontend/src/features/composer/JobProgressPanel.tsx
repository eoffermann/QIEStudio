import { motion } from "framer-motion";
import { Loader2, X, ImageIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useJobSocket } from "@/api/useJobSocket";
import { useCancelJob } from "@/api/hooks";
import { useUi } from "@/store/ui";
import { ResultGallery } from "./ResultGallery";

export function JobProgressPanel() {
  const activeJobId = useUi((s) => s.activeJobId);
  const setActiveJob = useUi((s) => s.setActiveJob);
  const cancel = useCancelJob();
  const sock = useJobSocket(activeJobId);

  if (!activeJobId) return null;

  const pct = sock.totalSteps ? Math.round((sock.step / sock.totalSteps) * 100) : 0;
  const running = sock.status === "running" || sock.status === "queued" || sock.status === null;
  const done = sock.status === "done" && sock.finalJob;

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="flex items-center gap-2 text-sm">
          {running ? (
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
          ) : (
            <ImageIcon className="h-4 w-4 text-primary" />
          )}
          {sock.status === "error"
            ? "Job failed"
            : done
              ? "Result"
              : sock.status === "queued"
                ? "Queued…"
                : "Generating…"}
        </CardTitle>
        <div className="flex items-center gap-2">
          {running && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => activeJobId && cancel.mutate(activeJobId)}
            >
              <X className="h-3.5 w-3.5" /> Cancel
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={() => setActiveJob(null)}>
            Dismiss
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {sock.status === "error" && (
          <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            {sock.error ?? "Unknown error"}
          </div>
        )}

        {running && (
          <>
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>
                Step {sock.step} / {sock.totalSteps || "?"}
              </span>
              <span>
                {sock.etaSeconds != null ? `~${Math.ceil(sock.etaSeconds)}s left` : "—"}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
              <motion.div
                className="h-full bg-primary"
                animate={{ width: `${pct}%` }}
                transition={{ ease: "easeOut" }}
              />
            </div>
            {sock.previewUrl ? (
              <motion.img
                key={sock.previewUrl}
                initial={{ opacity: 0.4 }}
                animate={{ opacity: 1 }}
                src={sock.previewUrl}
                alt="live latent preview"
                className="mx-auto max-h-[420px] rounded-xl object-contain"
              />
            ) : (
              <Skeleton className="aspect-square w-full max-w-md mx-auto" />
            )}
            <p className="text-center text-[11px] text-muted-foreground">
              Live latent preview (throttled)
            </p>
          </>
        )}

        {done && sock.finalJob && <ResultGallery job={sock.finalJob} />}
      </CardContent>
    </Card>
  );
}
