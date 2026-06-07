import { Moon, Sun, Command as CommandIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/store/theme";
import { useUi } from "@/store/ui";
import { DeviceBadge } from "@/components/DeviceBadge";

export function Topbar() {
  const { theme, toggleTheme } = useTheme();
  const toggleCommand = useUi((s) => s.toggleCommand);

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b bg-card/30 px-4 backdrop-blur">
      <button
        onClick={() => toggleCommand()}
        className="group flex h-9 w-72 items-center gap-2 rounded-xl border bg-background/50 px-3 text-sm text-muted-foreground transition-colors hover:border-primary/50"
      >
        <CommandIcon className="h-4 w-4" />
        <span>Search or run a command…</span>
        <kbd className="ml-auto rounded bg-muted px-1.5 py-0.5 font-mono text-[11px]">
          ⌘K
        </kbd>
      </button>

      <div className="flex items-center gap-2">
        <DeviceBadge />
        <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label="Toggle theme">
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </div>
    </header>
  );
}
