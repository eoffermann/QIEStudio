import { useState } from "react";
import {
  Search,
  History as HistoryIcon,
  RotateCcw,
  Eye,
  Trash2,
  X,
  ArrowUp,
  ArrowDown,
  ListOrdered,
} from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState } from "@/components/EmptyState";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  useJobHistory,
  useQueue,
  useCancelJob,
  useDeleteJob,
  useReorderQueue,
} from "@/api/hooks";
import { useUi } from "@/store/ui";
import { useNavigate } from "react-router-dom";
import { timeAgo } from "@/lib/utils";
import { applyJobSettings } from "@/lib/jobSettings";
import { toast } from "sonner";
import type { Job, JobStatus, Mode } from "@/api/types";

const STATUS_VARIANT: Record<JobStatus, "success" | "default" | "warning" | "danger" | "muted"> = {
  done: "success",
  running: "default",
  queued: "warning",
  error: "danger",
  canceled: "muted",
};

export function HistoryPage() {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<Mode | "all">("all");
  const { data, isLoading } = useJobHistory({
    q: q || undefined,
    mode: mode === "all" ? undefined : mode,
  });
  const queue = useQueue();
  const navigate = useNavigate();
  const setActiveJob = useUi((s) => s.setActiveJob);
  const cancel = useCancelJob();
  const del = useDeleteJob();
  const reorder = useReorderQueue();

  const rerun = (job: Job) => {
    applyJobSettings(job);
    navigate("/compose");
    toast.success("Loaded settings — press Generate to re-run");
  };

  const remove = (job: Job) => {
    if (!window.confirm("Delete this job and its outputs? This cannot be undone.")) return;
    del.mutate(job.id, { onSuccess: () => toast.success("Job deleted") });
  };

  const byId = new Map((data ?? []).map((j) => [j.id, j]));
  const running = queue.data?.running ? byId.get(queue.data.running) : undefined;
  const pending = (queue.data?.pending ?? []).map((id) => byId.get(id)).filter(Boolean) as Job[];

  const movePending = (index: number, dir: -1 | 1) => {
    const ids = pending.map((j) => j.id);
    const target = index + dir;
    if (target < 0 || target >= ids.length) return;
    [ids[index], ids[target]] = [ids[target], ids[index]];
    reorder.mutate(ids);
  };

  return (
    <div className="space-y-5 p-6">
      <PageHeader
        title="Queue & History"
        description="Manage the run queue (reorder, cancel, delete) and review / re-run / delete past jobs."
      />

      {/* --- Queue (running + pending) --- */}
      {(running || pending.length > 0) && (
        <div className="space-y-2 rounded-2xl border bg-card/50 p-4">
          <div className="flex items-center gap-2 text-sm font-medium">
            <ListOrdered className="h-4 w-4 text-primary" /> Queue
            <Badge variant="muted" className="text-[10px]">
              {(running ? 1 : 0) + pending.length} active
            </Badge>
          </div>
          {running && (
            <div className="flex items-center gap-3 rounded-xl border border-primary/40 bg-primary/5 p-2.5">
              <Badge variant="default">running</Badge>
              <span className="min-w-0 flex-1 truncate text-sm">{running.prompt}</span>
              <Button
                size="icon"
                variant="ghost"
                className="h-8 w-8"
                title="Cancel"
                onClick={() => cancel.mutate(running.id, { onSuccess: () => toast.success("Cancelling…") })}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          )}
          {pending.map((job, i) => (
            <div key={job.id} className="flex items-center gap-2 rounded-xl border p-2.5">
              <span className="w-6 text-center font-mono text-xs text-muted-foreground">
                {i + 1}
              </span>
              <span className="min-w-0 flex-1 truncate text-sm">{job.prompt}</span>
              <Badge variant="muted" className="text-[10px]">{job.mode}</Badge>
              <Button
                size="icon"
                variant="ghost"
                className="h-7 w-7"
                title="Move up"
                disabled={i === 0 || reorder.isPending}
                onClick={() => movePending(i, -1)}
              >
                <ArrowUp className="h-3.5 w-3.5" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                className="h-7 w-7"
                title="Move down"
                disabled={i === pending.length - 1 || reorder.isPending}
                onClick={() => movePending(i, 1)}
              >
                <ArrowDown className="h-3.5 w-3.5" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                className="h-7 w-7"
                title="Cancel"
                onClick={() => cancel.mutate(job.id, { onSuccess: () => toast.success("Canceled") })}
              >
                <X className="h-3.5 w-3.5" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                className="h-7 w-7 text-destructive"
                title="Delete"
                onClick={() => remove(job)}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center gap-3">
        <Tabs value={mode} onValueChange={(v) => setMode(v as Mode | "all")}>
          <TabsList>
            <TabsTrigger value="all">All</TabsTrigger>
            <TabsTrigger value="generate">Generate</TabsTrigger>
            <TabsTrigger value="edit">Edit</TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="relative max-w-sm flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            className="pl-9"
            placeholder="Search prompts…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-20" />
          ))}
        </div>
      ) : !data?.length ? (
        <EmptyState
          icon={HistoryIcon}
          title="No history yet"
          description="Generated and edited jobs will appear here."
        />
      ) : (
        <div className="space-y-2">
          {data.map((job) => (
            <div
              key={job.id}
              className="flex items-center gap-4 rounded-2xl border bg-card p-3 elevated"
            >
              {job.outputs[0] ? (
                <img
                  src={job.outputs[0].thumb_url ?? job.outputs[0].file_url}
                  alt="output"
                  className="h-16 w-16 shrink-0 rounded-xl object-cover"
                />
              ) : (
                <div className="grid h-16 w-16 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                  <HistoryIcon className="h-5 w-5" />
                </div>
              )}
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <Badge variant="muted">{job.mode}</Badge>
                  <Badge variant={STATUS_VARIANT[job.status]}>{job.status}</Badge>
                  <Badge variant="outline" className="text-[10px]">
                    {job.precision}
                  </Badge>
                  {job.batch_id && (
                    <Badge variant="outline" className="text-[10px]">
                      batch
                    </Badge>
                  )}
                  <span className="ml-auto text-xs text-muted-foreground">
                    {job.created_at ? timeAgo(job.created_at) : ""}
                  </span>
                </div>
                <p className="mt-1 line-clamp-1 text-sm text-muted-foreground">
                  {job.prompt}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                {(job.status === "queued" || job.status === "running") && (
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-8 w-8"
                    title="Cancel"
                    onClick={() => cancel.mutate(job.id, { onSuccess: () => toast.success("Canceled") })}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8"
                  title="View"
                  onClick={() => {
                    setActiveJob(job.id);
                    navigate("/compose");
                  }}
                >
                  <Eye className="h-4 w-4" />
                </Button>
                <Button size="sm" variant="outline" onClick={() => rerun(job)}>
                  <RotateCcw className="h-3.5 w-3.5" /> Re-run
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8 text-destructive"
                  title="Delete"
                  onClick={() => remove(job)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
