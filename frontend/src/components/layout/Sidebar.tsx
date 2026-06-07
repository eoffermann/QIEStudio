import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Wand2,
  Images,
  MessageSquareText,
  Layers,
  History,
  Settings,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/compose", label: "Compose", icon: Wand2 },
  { to: "/images", label: "Images", icon: Images },
  { to: "/prompts", label: "Prompts", icon: MessageSquareText },
  { to: "/loras", label: "LoRAs", icon: Layers },
  { to: "/history", label: "History", icon: History },
  { to: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  return (
    <aside className="flex h-full w-60 shrink-0 flex-col gap-2 border-r bg-card/40 p-3 backdrop-blur">
      <div className="flex items-center gap-2.5 px-2 py-3">
        <div className="grid h-9 w-9 place-items-center rounded-xl bg-primary text-primary-foreground shadow-[0_0_24px_-6px_var(--color-primary)]">
          <Sparkles className="h-5 w-5" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold">QIE Studio</div>
          <div className="text-[11px] text-muted-foreground">Qwen-Image</div>
        </div>
      </div>

      <nav className="mt-2 flex flex-col gap-1">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent/50",
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.div
                    layoutId="nav-active"
                    className="absolute inset-0 -z-10 rounded-xl bg-accent"
                    transition={{ type: "spring", stiffness: 400, damping: 32 }}
                  />
                )}
                <Icon className="h-4.5 w-4.5 shrink-0" />
                <span>{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto px-3 py-2 text-[11px] text-muted-foreground">
        Press{" "}
        <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono">⌘K</kbd> for
        commands
      </div>
    </aside>
  );
}
