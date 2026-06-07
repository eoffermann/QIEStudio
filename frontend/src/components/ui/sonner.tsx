import { Toaster as Sonner } from "sonner";
import { useTheme } from "@/store/theme";

export function Toaster() {
  const theme = useTheme((s) => s.theme);
  return (
    <Sonner
      theme={theme}
      position="bottom-right"
      richColors
      closeButton
      toastOptions={{
        classNames: {
          toast:
            "group rounded-xl border border-border bg-card text-card-foreground elevated",
        },
      }}
    />
  );
}
