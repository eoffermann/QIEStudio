import { useState } from "react";
import {
  Search,
  Plus,
  Copy,
  Trash2,
  Star,
  MessageSquareText,
  ArrowRightToLine,
  Pin,
  SquareDashed,
} from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState } from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  usePrompts,
  useCreatePrompt,
  useDuplicatePrompt,
  useDeletePrompt,
  usePatchPrompt,
} from "@/api/hooks";
import { useComposer } from "@/store/composer";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import type { Prompt } from "@/api/types";

export function PromptsPage() {
  const [q, setQ] = useState("");
  const { data, isLoading } = usePrompts({ q });
  const create = useCreatePrompt();
  const duplicate = useDuplicatePrompt();
  const remove = useDeletePrompt();
  const patch = usePatchPrompt();
  const navigate = useNavigate();

  const [editing, setEditing] = useState<Prompt | null>(null);
  const [newOpen, setNewOpen] = useState(false);
  const [draft, setDraft] = useState({ name: "", text: "" });

  const load = (p: Prompt) => {
    const c = useComposer.getState();
    c.setPrompt(p.text);
    c.setEnhancedPrompt(null);
    c.loadPromptId(p.id);
    if (p.mode !== "any") c.setMode(p.mode);
    if (p.loras.length) c.setLoras(p.loras);
    navigate("/compose");
    toast.success(`Loaded "${p.name}"`);
  };

  const saveNew = async () => {
    if (!draft.name.trim()) return;
    await create.mutateAsync({
      name: draft.name.trim(),
      text: draft.text,
      tags: [],
      mode: "any",
      loras: [],
      images: [],
      defaults: {},
      favorite: false,
    });
    setNewOpen(false);
    setDraft({ name: "", text: "" });
    toast.success("Prompt created");
  };

  return (
    <div className="space-y-5 p-6">
      <PageHeader
        title="Prompts"
        description="Saved prompts with tags, favorites, and image slots/bindings."
        actions={
          <Button onClick={() => setNewOpen(true)}>
            <Plus className="h-4 w-4" /> New prompt
          </Button>
        }
      />

      <div className="relative max-w-sm">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          className="pl-9"
          placeholder="Search prompts…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      </div>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-40" />
          ))}
        </div>
      ) : !data?.length ? (
        <EmptyState
          icon={MessageSquareText}
          title="No prompts yet"
          description="Save reusable prompts with pinned images and open slots."
          action={
            <Button onClick={() => setNewOpen(true)}>
              <Plus className="h-4 w-4" /> New prompt
            </Button>
          }
        />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {data.map((p) => (
            <div
              key={p.id}
              className="flex flex-col gap-2 rounded-2xl border bg-card p-4 elevated"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{p.name}</span>
                  <Badge variant="muted">{p.mode}</Badge>
                </div>
                <button
                  onClick={() =>
                    patch.mutate({ id: p.id, body: { favorite: !p.favorite } })
                  }
                  title="Favorite"
                >
                  <Star
                    className={
                      "h-4 w-4 " +
                      (p.favorite
                        ? "fill-amber-400 text-amber-400"
                        : "text-muted-foreground")
                    }
                  />
                </button>
              </div>
              <p className="line-clamp-3 text-sm text-muted-foreground">{p.text}</p>

              {(p.images.length > 0 || p.loras.length > 0) && (
                <div className="flex flex-wrap gap-1.5">
                  {p.images.map((img, i) => (
                    <Badge key={i} variant="outline" className="gap-1 text-[10px]">
                      {img.role === "pinned" ? (
                        <Pin className="h-3 w-3" />
                      ) : (
                        <SquareDashed className="h-3 w-3" />
                      )}
                      {img.role === "pinned" ? "pinned" : (img.slot_name ?? "slot")}
                    </Badge>
                  ))}
                  {p.loras.length > 0 && (
                    <Badge variant="outline" className="text-[10px]">
                      {p.loras.length} LoRA{p.loras.length > 1 ? "s" : ""}
                    </Badge>
                  )}
                </div>
              )}

              <div className="mt-auto flex items-center gap-1 pt-2">
                <Button size="sm" onClick={() => load(p)}>
                  <ArrowRightToLine className="h-3.5 w-3.5" /> Load
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setEditing(p)}>
                  Edit
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="ml-auto h-8 w-8"
                  title="Duplicate"
                  onClick={() => duplicate.mutate(p.id)}
                >
                  <Copy className="h-3.5 w-3.5" />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8"
                  title="Delete"
                  onClick={() => remove.mutate(p.id)}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* New prompt */}
      <Dialog open={newOpen} onOpenChange={setNewOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New prompt</DialogTitle>
          </DialogHeader>
          <Input
            placeholder="Name"
            value={draft.name}
            onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))}
            autoFocus
          />
          <Textarea
            placeholder="Prompt text"
            value={draft.text}
            onChange={(e) => setDraft((d) => ({ ...d, text: e.target.value }))}
            className="min-h-[120px]"
          />
          <DialogFooter>
            <Button variant="ghost" onClick={() => setNewOpen(false)}>
              Cancel
            </Button>
            <Button onClick={saveNew}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit prompt (name/text); slots/bindings editor is a TODO stub) */}
      <Dialog open={!!editing} onOpenChange={(o) => !o && setEditing(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Edit prompt</DialogTitle>
          </DialogHeader>
          {editing && (
            <>
              <Input
                value={editing.name}
                onChange={(e) => setEditing({ ...editing, name: e.target.value })}
              />
              <Textarea
                value={editing.text}
                onChange={(e) => setEditing({ ...editing, text: e.target.value })}
                className="min-h-[140px]"
              />
              <div className="rounded-xl border border-dashed p-3 text-xs text-muted-foreground">
                TODO: pinned-asset + open-slot bindings editor (DESIGN §5.2).
              </div>
              <DialogFooter>
                <Button variant="ghost" onClick={() => setEditing(null)}>
                  Cancel
                </Button>
                <Button
                  onClick={() =>
                    patch
                      .mutateAsync({
                        id: editing.id,
                        body: { name: editing.name, text: editing.text },
                      })
                      .then(() => {
                        setEditing(null);
                        toast.success("Saved");
                      })
                  }
                >
                  Save
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
