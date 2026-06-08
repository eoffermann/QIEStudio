import { useState } from "react";
import { Sparkles, Save, FolderOpen, Check, Undo2, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useComposer } from "@/store/composer";
import { useEnhance, useCreatePrompt, usePrompts } from "@/api/hooks";
import { toast } from "sonner";
import { PromptPicker } from "./PromptPicker";

export function PromptEditor() {
  const mode = useComposer((s) => s.mode);
  const prompt = useComposer((s) => s.prompt);
  const setPrompt = useComposer((s) => s.setPrompt);
  const enhanced = useComposer((s) => s.enhancedPrompt);
  const setEnhanced = useComposer((s) => s.setEnhancedPrompt);
  const inputs = useComposer((s) => s.inputs);

  const enhance = useEnhance();
  const createPrompt = useCreatePrompt();
  const prompts = usePrompts();

  const [diffOpen, setDiffOpen] = useState(false);
  const [draftEnhanced, setDraftEnhanced] = useState("");
  const [saveOpen, setSaveOpen] = useState(false);
  const [pickOpen, setPickOpen] = useState(false);
  const [saveName, setSaveName] = useState("");

  const runEnhance = async () => {
    if (!prompt.trim()) {
      toast.error("Write a prompt first");
      return;
    }
    try {
      const res = await enhance.mutateAsync({
        mode,
        prompt,
        image_ids: inputs.map((i) => i.asset.id),
      });
      setDraftEnhanced(res.enhanced_prompt);
      setDiffOpen(true);
    } catch (e) {
      toast.error(`Enhance failed: ${(e as Error).message}`);
    }
  };

  const acceptEnhanced = () => {
    setEnhanced(draftEnhanced);
    setDiffOpen(false);
    toast.success("Enhanced prompt applied");
  };

  const save = async () => {
    if (!saveName.trim()) return;
    try {
      await createPrompt.mutateAsync({
        name: saveName.trim(),
        text: prompt,
        tags: [],
        mode,
        loras: useComposer.getState().loras,
        images: [],
        defaults_json: {},
        favorite: false,
      });
      toast.success("Prompt saved");
      setSaveOpen(false);
      setSaveName("");
    } catch (e) {
      toast.error(`Save failed: ${(e as Error).message}`);
    }
  };

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="text-sm">Prompt</CardTitle>
        <div className="flex items-center gap-1.5">
          <Button size="sm" variant="ghost" onClick={() => setPickOpen(true)}>
            <FolderOpen className="h-3.5 w-3.5" /> Load
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setSaveOpen(true)}>
            <Save className="h-3.5 w-3.5" /> Save
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex min-h-0 flex-1 flex-col gap-3">
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={
            mode === "edit"
              ? "Describe the edit, e.g. 'place the chair in the living room near the window…'"
              : "Describe the image you want to generate…"
          }
          className="min-h-[140px] flex-1"
        />

        {enhanced && (
          <div className="rounded-xl border border-primary/30 bg-primary/5 p-3">
            <div className="mb-1 flex items-center gap-2">
              <Badge>
                <Sparkles className="h-3 w-3" /> Enhanced
              </Badge>
              <Button
                size="sm"
                variant="ghost"
                className="ml-auto h-7"
                onClick={() => setEnhanced(null)}
              >
                <Undo2 className="h-3.5 w-3.5" /> Use original
              </Button>
            </div>
            <p className="text-sm text-foreground/90">{enhanced}</p>
          </div>
        )}

        <div className="flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            {mode === "edit"
              ? "Enhance sees your input images (multimodal)."
              : "Enhance enriches your text prompt."}
          </p>
          <Button
            variant="glow"
            onClick={runEnhance}
            disabled={enhance.isPending}
          >
            {enhance.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            Enhance
          </Button>
        </div>
      </CardContent>

      {/* Enhance diff/preview dialog */}
      <Dialog open={diffOpen} onOpenChange={setDiffOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" /> Review enhanced prompt
            </DialogTitle>
          </DialogHeader>
          <div className="grid gap-3">
            <div>
              <div className="mb-1 text-xs font-medium uppercase text-muted-foreground">
                Original
              </div>
              <div className="rounded-xl border bg-muted/40 p-3 text-sm text-muted-foreground">
                {prompt}
              </div>
            </div>
            <div>
              <div className="mb-1 text-xs font-medium uppercase text-muted-foreground">
                Enhanced (editable)
              </div>
              <Textarea
                value={draftEnhanced}
                onChange={(e) => setDraftEnhanced(e.target.value)}
                className="min-h-[120px]"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setDiffOpen(false)}>
              Reject
            </Button>
            <Button onClick={acceptEnhanced}>
              <Check className="h-4 w-4" /> Apply
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Save dialog */}
      <Dialog open={saveOpen} onOpenChange={setSaveOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Save prompt</DialogTitle>
          </DialogHeader>
          <Input
            placeholder="Prompt name"
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && save()}
            autoFocus
          />
          <DialogFooter>
            <Button variant="ghost" onClick={() => setSaveOpen(false)}>
              Cancel
            </Button>
            <Button onClick={save} disabled={createPrompt.isPending}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Load picker */}
      <PromptPicker
        open={pickOpen}
        onOpenChange={setPickOpen}
        prompts={prompts.data ?? []}
        onPick={(p) => {
          setPrompt(p.text);
          setEnhanced(null);
          useComposer.getState().loadPromptId(p.id);
          if (p.loras.length) useComposer.getState().setLoras(p.loras);
          setPickOpen(false);
          toast.success(`Loaded "${p.name}"`);
        }}
      />
    </Card>
  );
}
