import { useRef, useState } from "react";
import { Reorder } from "framer-motion";
import { Upload, X, GripVertical, Images as ImagesIcon, Pin } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useComposer, newUid, type InputImage } from "@/store/composer";
import { useUploadAssets } from "@/api/hooks";
import { assetThumbUrl, assetFileUrl } from "@/api/client";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

const ORIGIN_BADGE: Record<InputImage["origin"], string> = {
  upload: "Upload",
  library: "Library",
  pinned: "Pinned",
  slot: "Slot",
};

export function InputImagesPanel() {
  const inputs = useComposer((s) => s.inputs);
  const addInputs = useComposer((s) => s.addInputs);
  const removeInput = useComposer((s) => s.removeInput);
  const setInputs = useComposer.setState;
  const upload = useUploadAssets();
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = async (files: FileList | File[]) => {
    const arr = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (!arr.length) return;
    try {
      const assets = await upload.mutateAsync(arr);
      addInputs(
        assets.map((a) => ({ uid: newUid(), asset: a, origin: "upload" as const })),
      );
      toast.success(`Added ${assets.length} image${assets.length > 1 ? "s" : ""}`);
    } catch (e) {
      toast.error(`Upload failed: ${(e as Error).message}`);
    }
  };

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="flex items-center gap-2 text-sm">
          <ImagesIcon className="h-4 w-4 text-primary" />
          Input images
          <Badge variant="muted">{inputs.length}</Badge>
        </CardTitle>
        <Button size="sm" variant="outline" onClick={() => fileRef.current?.click()}>
          <Upload className="h-3.5 w-3.5" /> Add
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*,.heic,.heif"
          multiple
          hidden
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
      </CardHeader>

      <CardContent className="min-h-0 flex-1 overflow-y-auto">
        <p className="mb-3 text-xs text-muted-foreground">
          Order is meaningful to the model — drag to reorder.
        </p>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
          }}
          className={cn(
            "rounded-xl border border-dashed p-3 transition-colors",
            dragOver ? "border-primary bg-primary/5" : "border-border",
          )}
        >
          {inputs.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-8 text-center text-sm text-muted-foreground">
              <Upload className="h-6 w-6" />
              Drop images here or click Add
            </div>
          ) : (
            <Reorder.Group
              axis="y"
              values={inputs}
              onReorder={(next) => setInputs({ inputs: next })}
              className="flex flex-col gap-2"
            >
              {inputs.map((img, idx) => (
                <Reorder.Item
                  key={img.uid}
                  value={img}
                  className="flex items-center gap-3 rounded-xl border bg-card p-2"
                >
                  <GripVertical className="h-4 w-4 shrink-0 cursor-grab text-muted-foreground" />
                  <div className="grid h-6 w-6 shrink-0 place-items-center rounded-md bg-primary/15 text-xs font-semibold text-primary">
                    {idx + 1}
                  </div>
                  <img
                    src={assetThumbUrl(img.asset.id, img.asset.thumb_key) ?? assetFileUrl(img.asset.id)}
                    alt={img.asset.name || "input"}
                    className="h-12 w-12 shrink-0 rounded-lg object-cover"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm">
                      {img.asset.name ?? img.asset.id.slice(0, 8)}
                    </div>
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      {img.origin === "pinned" && <Pin className="h-3 w-3" />}
                      {ORIGIN_BADGE[img.origin]}
                      {img.slotName && <span>· {img.slotName}</span>}
                      <span>· {img.asset.width}×{img.asset.height}</span>
                    </div>
                  </div>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0"
                    onClick={() => removeInput(img.uid)}
                  >
                    <X className="h-3.5 w-3.5" />
                  </Button>
                </Reorder.Item>
              ))}
            </Reorder.Group>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
