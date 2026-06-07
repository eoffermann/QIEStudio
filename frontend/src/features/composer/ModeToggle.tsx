import { motion } from "framer-motion";
import { Wand2, ImagePlus } from "lucide-react";
import { cn } from "@/lib/utils";
import { useComposer } from "@/store/composer";
import type { Mode } from "@/api/types";

const MODES: { id: Mode; label: string; icon: typeof Wand2; hint: string }[] = [
  { id: "generate", label: "Generate", icon: Wand2, hint: "Text → image" },
  { id: "edit", label: "Edit", icon: ImagePlus, hint: "Images + instruction" },
];

export function ModeToggle() {
  const mode = useComposer((s) => s.mode);
  const setMode = useComposer((s) => s.setMode);

  return (
    <div className="inline-flex rounded-2xl border bg-card/60 p-1 backdrop-blur">
      {MODES.map(({ id, label, icon: Icon, hint }) => {
        const active = mode === id;
        return (
          <button
            key={id}
            onClick={() => setMode(id)}
            className={cn(
              "relative flex items-center gap-2.5 rounded-xl px-5 py-2.5 text-sm font-medium transition-colors",
              active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.div
                layoutId="mode-active"
                className="absolute inset-0 -z-10 rounded-xl bg-primary"
                transition={{ type: "spring", stiffness: 420, damping: 34 }}
              />
            )}
            <Icon className="h-4 w-4" />
            <div className="text-left leading-tight">
              <div>{label}</div>
              <div
                className={cn(
                  "text-[10px] font-normal",
                  active ? "text-primary-foreground/80" : "text-muted-foreground/70",
                )}
              >
                {hint}
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
