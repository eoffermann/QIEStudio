import { Routes, Route, Navigate } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { CommandPalette } from "@/components/CommandPalette";
import { useGlobalShortcuts } from "@/hooks/useGlobalShortcuts";
import { ComposerPage } from "@/features/composer/ComposerPage";
import { ImagesPage } from "@/features/libraries/ImagesPage";
import { PromptsPage } from "@/features/libraries/PromptsPage";
import { LorasPage } from "@/features/libraries/LorasPage";
import { HistoryPage } from "@/features/history/HistoryPage";
import { SettingsPage } from "@/features/settings/SettingsPage";

export default function App() {
  useGlobalShortcuts();
  return (
    <AppShell>
      <CommandPalette />
      <Routes>
        <Route path="/" element={<Navigate to="/compose" replace />} />
        <Route path="/compose" element={<ComposerPage />} />
        <Route path="/images" element={<ImagesPage />} />
        <Route path="/prompts" element={<PromptsPage />} />
        <Route path="/loras" element={<LorasPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/compose" replace />} />
      </Routes>
    </AppShell>
  );
}
