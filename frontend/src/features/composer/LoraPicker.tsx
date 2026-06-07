import { Layers, Plus } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useLoras } from "@/api/hooks";
import { useComposer } from "@/store/composer";
import { cn } from "@/lib/utils";

export function LoraPicker() {
  const { data: loras } = useLoras();
  const mode = useComposer((s) => s.mode);
  const selected = useComposer((s) => s.loras);
  const toggleLora = useComposer((s) => s.toggleLora);
  const setLoraWeight = useComposer((s) => s.setLoraWeight);

  const available = (loras ?? []).filter((l) => l.enabled);
  const unselected = available.filter(
    (l) => !selected.some((s) => s.lora_id === l.id),
  );

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        <Layers className="h-3.5 w-3.5" /> LoRAs
        {selected.length > 0 && <Badge variant="muted">{selected.length}</Badge>}
      </div>

      <Select
        value=""
        onValueChange={(id) => {
          const l = available.find((x) => x.id === id);
          if (l) toggleLora(l.id, l.recommended_weight ?? 0.8);
        }}
      >
        <SelectTrigger className="h-9">
          <span className="flex items-center gap-2 text-muted-foreground">
            <Plus className="h-3.5 w-3.5" />
            <SelectValue placeholder="Add a LoRA…" />
          </span>
        </SelectTrigger>
        <SelectContent>
          {unselected.length === 0 ? (
            <div className="px-3 py-2 text-xs text-muted-foreground">
              No more LoRAs available
            </div>
          ) : (
            unselected.map((l) => {
              const mismatch =
                (mode === "edit" && l.base_compat !== "qwen-image-edit") ||
                (mode === "generate" && l.base_compat !== "qwen-image");
              return (
                <SelectItem key={l.id} value={l.id}>
                  <span className={cn(mismatch && "text-amber-400")}>
                    {l.name}
                    {mismatch && " (base mismatch)"}
                  </span>
                </SelectItem>
              );
            })
          )}
        </SelectContent>
      </Select>

      <div className="space-y-2">
        {selected.map((sel) => {
          const lora = available.find((l) => l.id === sel.lora_id);
          return (
            <div key={sel.lora_id} className="rounded-xl border bg-card/60 p-2.5">
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-sm font-medium">
                  {lora?.name ?? sel.lora_id.slice(0, 8)}
                </span>
                <button
                  className="text-xs text-muted-foreground hover:text-destructive"
                  onClick={() => toggleLora(sel.lora_id, sel.weight)}
                >
                  remove
                </button>
              </div>
              <div className="mt-2 flex items-center gap-3">
                <Slider
                  value={[sel.weight]}
                  min={0}
                  max={1.5}
                  step={0.05}
                  onValueChange={([v]) => setLoraWeight(sel.lora_id, v)}
                  className="flex-1"
                />
                <span className="w-10 text-right font-mono text-xs">
                  {sel.weight.toFixed(2)}
                </span>
              </div>
              {lora?.trigger_words.length ? (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {lora.trigger_words.slice(0, 4).map((t) => (
                    <Badge key={t} variant="outline" className="text-[10px]">
                      {t}
                    </Badge>
                  ))}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
      {available.length === 0 && (
        <p className="text-xs text-muted-foreground">
          No LoRAs installed. Add some in the LoRAs library.
        </p>
      )}
      <Label className="sr-only">lora selection</Label>
    </div>
  );
}
