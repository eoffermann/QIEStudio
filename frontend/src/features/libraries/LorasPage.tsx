import { useRef, useState } from "react";
import { Upload, Plus, Trash2, Layers, Link2 } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState } from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import { loraThumbUrl } from "@/api/client";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  useLoras,
  useUploadLora,
  useImportLora,
  useDeleteLora,
  usePatchLora,
} from "@/api/hooks";
import { toast } from "sonner";

export function LorasPage() {
  const { data, isLoading } = useLoras();
  const upload = useUploadLora();
  const importLora = useImportLora();
  const remove = useDeleteLora();
  const patch = usePatchLora();
  const fileRef = useRef<HTMLInputElement>(null);
  const [importOpen, setImportOpen] = useState(false);
  const [ref, setRef] = useState({ hf: "", civitai: "", url: "" });

  const doImport = (source: "hf" | "civitai" | "url", value: string) => {
    if (!value.trim()) return;
    importLora
      .mutateAsync({ source, ref: value.trim() })
      .then(() => {
        toast.success("Import started");
        setImportOpen(false);
        setRef({ hf: "", civitai: "", url: "" });
      })
      .catch((e) => toast.error((e as Error).message));
  };

  return (
    <div className="space-y-5 p-6">
      <PageHeader
        title="LoRAs"
        description="Installed adapters. Import from upload, Hugging Face, CivitAI, or a direct URL."
        actions={
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => fileRef.current?.click()}>
              <Upload className="h-4 w-4" /> Upload
            </Button>
            <Button onClick={() => setImportOpen(true)}>
              <Plus className="h-4 w-4" /> Import
            </Button>
          </div>
        }
      />
      <input
        ref={fileRef}
        type="file"
        accept=".safetensors"
        hidden
        onChange={(e) =>
          e.target.files?.[0] &&
          upload
            .mutateAsync({ file: e.target.files[0] })
            .then(() => toast.success("LoRA uploaded"))
            .catch((err) => toast.error((err as Error).message))
        }
      />

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      ) : !data?.length ? (
        <EmptyState
          icon={Layers}
          title="No LoRAs installed"
          description="Upload a .safetensors file or import from HF / CivitAI / URL."
          action={
            <Button onClick={() => setImportOpen(true)}>
              <Plus className="h-4 w-4" /> Import
            </Button>
          }
        />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {data.map((l) => (
            <div key={l.id} className="flex gap-3 rounded-2xl border bg-card p-4 elevated">
              {l.thumb_key ? (
                <img
                  src={loraThumbUrl(l.id, l.thumb_key)}
                  alt={l.name}
                  className="h-20 w-20 shrink-0 rounded-xl border object-cover"
                />
              ) : (
                <div className="grid h-20 w-20 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                  <Layers className="h-6 w-6" />
                </div>
              )}
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-medium">{l.name}</span>
                  <Switch
                    checked={l.enabled}
                    onCheckedChange={(v) => patch.mutate({ id: l.id, body: { enabled: v } })}
                  />
                </div>
                <div className="mt-1 flex flex-wrap gap-1">
                  <Badge variant="muted" className="text-[10px]">
                    {l.base_compat}
                  </Badge>
                  <Badge variant="outline" className="text-[10px]">
                    {l.source}
                  </Badge>
                  <Badge variant="outline" className="text-[10px]">
                    w {l.recommended_weight}
                  </Badge>
                </div>
                {l.trigger_words.length > 0 && (
                  <p className="mt-1 line-clamp-1 text-[11px] text-muted-foreground">
                    {l.trigger_words.join(", ")}
                  </p>
                )}
                <Button
                  size="sm"
                  variant="ghost"
                  className="mt-1 h-7 px-2 text-destructive"
                  onClick={() => remove.mutate(l.id)}
                >
                  <Trash2 className="h-3.5 w-3.5" /> Remove
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Import dialog */}
      <Dialog open={importOpen} onOpenChange={setImportOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Import a LoRA</DialogTitle>
          </DialogHeader>
          <Tabs defaultValue="hf">
            <TabsList className="w-full">
              <TabsTrigger value="hf" className="flex-1">
                Hugging Face
              </TabsTrigger>
              <TabsTrigger value="civitai" className="flex-1">
                CivitAI
              </TabsTrigger>
              <TabsTrigger value="url" className="flex-1">
                URL
              </TabsTrigger>
            </TabsList>
            <TabsContent value="hf" className="space-y-2">
              <Input
                placeholder="repo id, e.g. user/qwen-lora"
                value={ref.hf}
                onChange={(e) => setRef((r) => ({ ...r, hf: e.target.value }))}
              />
              <Button className="w-full" onClick={() => doImport("hf", ref.hf)}>
                Import from HF
              </Button>
            </TabsContent>
            <TabsContent value="civitai" className="space-y-2">
              <Input
                placeholder="civitai.com/.red/.green URL or model/version id"
                value={ref.civitai}
                onChange={(e) => setRef((r) => ({ ...r, civitai: e.target.value }))}
              />
              <p className="text-xs text-muted-foreground">
                .com / .red / .green URLs all resolve via the shared v1 API.
              </p>
              <Button className="w-full" onClick={() => doImport("civitai", ref.civitai)}>
                Import from CivitAI
              </Button>
            </TabsContent>
            <TabsContent value="url" className="space-y-2">
              <Input
                placeholder="https://…/adapter.safetensors"
                value={ref.url}
                onChange={(e) => setRef((r) => ({ ...r, url: e.target.value }))}
              />
              <Button className="w-full" onClick={() => doImport("url", ref.url)}>
                <Link2 className="h-4 w-4" /> Import from URL
              </Button>
            </TabsContent>
          </Tabs>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setImportOpen(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
