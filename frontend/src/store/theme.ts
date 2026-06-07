import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ThemeMode = "dark" | "light";

interface ThemeState {
  theme: ThemeMode;
  accent: string; // oklch lightness/chroma/hue string for --primary
  setTheme: (t: ThemeMode) => void;
  toggleTheme: () => void;
  setAccent: (a: string) => void;
}

/** Apply theme + accent to the document root. */
export function applyTheme(theme: ThemeMode, accent: string) {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  root.classList.toggle("light", theme === "light");
  root.style.setProperty("--primary", accent);
  root.style.setProperty("--ring", accent);
}

export const useTheme = create<ThemeState>()(
  persist(
    (set, get) => ({
      theme: "dark",
      accent: "oklch(0.68 0.19 290)",
      setTheme: (theme) => {
        set({ theme });
        applyTheme(theme, get().accent);
      },
      toggleTheme: () => {
        const next = get().theme === "dark" ? "light" : "dark";
        set({ theme: next });
        applyTheme(next, get().accent);
      },
      setAccent: (accent) => {
        set({ accent });
        applyTheme(get().theme, accent);
      },
    }),
    { name: "qie-theme" },
  ),
);
