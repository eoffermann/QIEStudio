import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import type { Prompt } from "@/api/types";
import { MessageSquareText } from "lucide-react";

export function PromptPicker({
  open,
  onOpenChange,
  prompts,
  onPick,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  prompts: Prompt[];
  onPick: (p: Prompt) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Load a saved prompt</DialogTitle>
        </DialogHeader>
        <div className="max-h-[60vh] space-y-2 overflow-y-auto">
          {prompts.length === 0 && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              No saved prompts yet.
            </div>
          )}
          {prompts.map((p) => (
            <button
              key={p.id}
              onClick={() => onPick(p)}
              className="flex w-full items-start gap-3 rounded-xl border bg-card p-3 text-left transition-colors hover:border-primary/50 hover:bg-accent/40"
            >
              <MessageSquareText className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{p.name}</span>
                  <Badge variant="muted">{p.mode}</Badge>
                </div>
                <p className="mt-0.5 line-clamp-2 text-sm text-muted-foreground">
                  {p.text}
                </p>
              </div>
            </button>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
