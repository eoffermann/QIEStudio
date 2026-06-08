import { useState } from "react";
import { Search, History as HistoryIcon, RotateCcw, Eye } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState } from "@/components/EmptyState";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useJobHistory } from "@/api/hooks";
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
  const navigate = useNavigate();
  const setActiveJob = useUi((s) => s.setActiveJob);

  const rerun = (job: Job) => {
    applyJobSettings(job);
    navigate("/compose");
    toast.success("Loaded settings — press Generate to re-run");
  };

  return (
    <div className="space-y-5 p-6">
      <PageHeader
        title="History"
        description="Past jobs with full reproducibility metadata. Search, filter, and re-run."
      />

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
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
