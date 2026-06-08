import { useRef, useState } from "react";
import { Upload, Search, Star, Trash2, ArrowRightToLine, Images as ImagesIcon } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState } from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  useAssets,
  useUploadAssets,
  usePromoteAsset,
  useDeleteAsset,
} from "@/api/hooks";
import { assetThumbUrl, assetFileUrl } from "@/api/client";
import { useComposer, newUid } from "@/store/composer";
import { useNavigate } from "react-router-dom";
import { formatBytes } from "@/lib/utils";
import { toast } from "sonner";
import type { Asset } from "@/api/types";

export function ImagesPage() {
  const [scope, setScope] = useState<"library" | "ephemeral">("library");
  const [q, setQ] = useState("");
  const { data, isLoading } = useAssets({ scope, q });
  const upload = useUploadAssets();
  const promote = usePromoteAsset();
  const remove = useDeleteAsset();
  const fileRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const addInputs = useComposer((s) => s.addInputs);

  const sendToComposer = (a: Asset) => {
    useComposer.getState().setMode("edit");
    addInputs([{ uid: newUid(), asset: a, origin: "library" }]);
    navigate("/compose");
    toast.success("Added to composer");
  };

  return (
    <div className="space-y-5 p-6">
      <PageHeader
        title="Images"
        description="Persistent library assets and ephemeral uploads. Promote either into the library."
        actions={
          <Button onClick={() => fileRef.current?.click()}>
            <Upload className="h-4 w-4" /> Upload
          </Button>
        }
      />
      <input
        ref={fileRef}
        type="file"
        accept="image/*,.heic,.heif"
        multiple
        hidden
        onChange={(e) =>
          e.target.files &&
          upload
            .mutateAsync(Array.from(e.target.files))
            .then(() => toast.success("Uploaded"))
            .catch((err) => toast.error((err as Error).message))
        }
      />

      <div className="flex items-center gap-3">
        <Tabs value={scope} onValueChange={(v) => setScope(v as typeof scope)}>
          <TabsList>
            <TabsTrigger value="library">Library</TabsTrigger>
            <TabsTrigger value="ephemeral">Uploads</TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            className="pl-9"
            placeholder="Search by name or tag…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          {Array.from({ length: 10 }).map((_, i) => (
            <Skeleton key={i} className="aspect-square" />
          ))}
        </div>
      ) : !data?.length ? (
        <EmptyState
          icon={ImagesIcon}
          title="No images yet"
          description="Upload images or generate outputs and promote them here."
          action={
            <Button onClick={() => fileRef.current?.click()}>
              <Upload className="h-4 w-4" /> Upload
            </Button>
          }
        />
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          {data.map((a) => (
            <div
              key={a.id}
              className="group relative overflow-hidden rounded-2xl border bg-card elevated"
            >
              <img
                src={assetThumbUrl(a.id, a.thumb_key) ?? assetFileUrl(a.id)}
                alt={a.name || a.id}
                className="aspect-square w-full object-cover"
              />
              <div className="p-2">
                <div className="truncate text-sm font-medium">
                  {a.name ?? a.id.slice(0, 8)}
                </div>
                <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                  <span>{a.width}×{a.height}</span>
                  <span>·</span>
                  <span>{formatBytes(a.bytes)}</span>
                  <Badge variant="muted" className="ml-auto text-[10px]">
                    {a.source}
                  </Badge>
                </div>
              </div>
              <div className="absolute inset-x-0 top-0 flex justify-end gap-1 p-2 opacity-0 transition-opacity group-hover:opacity-100">
                <Button
                  size="icon"
                  variant="secondary"
                  className="h-7 w-7"
                  title="Send to composer"
                  onClick={() => sendToComposer(a)}
                >
                  <ArrowRightToLine className="h-3.5 w-3.5" />
                </Button>
                {a.scope === "ephemeral" && (
                  <Button
                    size="icon"
                    variant="secondary"
                    className="h-7 w-7"
                    title="Promote to library"
                    onClick={() =>
                      promote
                        .mutateAsync(a.id)
                        .then(() => toast.success("Promoted"))
                        .catch((e) => toast.error((e as Error).message))
                    }
                  >
                    <Star className="h-3.5 w-3.5" />
                  </Button>
                )}
                <Button
                  size="icon"
                  variant="secondary"
                  className="h-7 w-7"
                  title="Delete"
                  onClick={() =>
                    remove
                      .mutateAsync(a.id)
                      .then(() => toast.success("Deleted"))
                      .catch((e) => toast.error((e as Error).message))
                  }
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
