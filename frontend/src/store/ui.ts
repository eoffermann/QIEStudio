import { create } from "zustand";

interface UiState {
  commandOpen: boolean;
  /** Currently watched job id (drives the live progress panel). */
  activeJobId: string | null;
  /** Batch id if the active run is a batch. */
  activeBatchId: string | null;
  setCommandOpen: (open: boolean) => void;
  toggleCommand: () => void;
  setActiveJob: (id: string | null) => void;
  setActiveBatch: (id: string | null) => void;
}

export const useUi = create<UiState>((set, get) => ({
  commandOpen: false,
  activeJobId: null,
  activeBatchId: null,
  setCommandOpen: (commandOpen) => set({ commandOpen }),
  toggleCommand: () => set({ commandOpen: !get().commandOpen }),
  setActiveJob: (activeJobId) => set({ activeJobId }),
  setActiveBatch: (activeBatchId) => set({ activeBatchId }),
}));
