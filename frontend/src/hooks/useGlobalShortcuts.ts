import { useEffect } from "react";
import { useUi } from "@/store/ui";
import { useComposer } from "@/store/composer";

/**
 * Global keyboard shortcuts for the core loop.
 *  ⌘K / Ctrl+K  — command palette
 *  R            — reseed (when not typing)
 *  G            — generate (dispatches a window event the composer listens for)
 */
export function useGlobalShortcuts() {
  const toggleCommand = useUi((s) => s.toggleCommand);
  const reseed = useComposer((s) => s.reseed);

  useEffect(() => {
    const isEditable = (el: EventTarget | null) => {
      const node = el as HTMLElement | null;
      if (!node) return false;
      const tag = node.tagName;
      return (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        node.isContentEditable
      );
    };

    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        toggleCommand();
        return;
      }
      if (isEditable(e.target)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) {
        if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
          e.preventDefault();
          window.dispatchEvent(new CustomEvent("qie:generate"));
        }
        return;
      }
      if (e.key.toLowerCase() === "r") {
        reseed();
      }
      if (e.key.toLowerCase() === "g") {
        window.dispatchEvent(new CustomEvent("qie:generate"));
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleCommand, reseed]);
}
