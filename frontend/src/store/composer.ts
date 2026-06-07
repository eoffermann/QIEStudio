import { create } from "zustand";
import type {
  Asset,
  BaseSize,
  Mode,
  Orientation,
  Precision,
  PromptLora,
} from "@/api/types";

/** An ordered input image in the Edit composer. Origin tracks how it got there. */
export interface InputImage {
  /** stable client id for drag/reorder */
  uid: string;
  asset: Asset;
  origin: "upload" | "library" | "pinned" | "slot";
  slotName?: string;
}

export interface ResolutionState {
  base: BaseSize;
  orientation: Orientation;
  aspect: string;
}

export interface ComposerState {
  mode: Mode;
  modelId: string | null;
  precision: Precision;
  prompt: string;
  enhancedPrompt: string | null;
  negativePrompt: string;
  inputs: InputImage[];
  loras: PromptLora[];
  resolution: ResolutionState;
  steps: number;
  trueCfgScale: number;
  guidanceScale: number;
  seed: number | null; // null = random
  lockSeed: boolean;
  batch: number;
  outputFormat: "png" | "webp" | "jpeg";
  loadedPromptId: string | null;

  setMode: (m: Mode) => void;
  setModelId: (id: string) => void;
  setPrecision: (p: Precision) => void;
  setPrompt: (t: string) => void;
  setEnhancedPrompt: (t: string | null) => void;
  setNegativePrompt: (t: string) => void;
  setResolution: (r: Partial<ResolutionState>) => void;
  setSteps: (n: number) => void;
  setTrueCfgScale: (n: number) => void;
  setSeed: (n: number | null) => void;
  setLockSeed: (b: boolean) => void;
  setBatch: (n: number) => void;
  setOutputFormat: (f: "png" | "webp" | "jpeg") => void;

  addInputs: (imgs: InputImage[]) => void;
  removeInput: (uid: string) => void;
  reorderInputs: (from: number, to: number) => void;
  clearInputs: () => void;

  setLoras: (l: PromptLora[]) => void;
  toggleLora: (loraId: string, weight: number) => void;
  setLoraWeight: (loraId: string, weight: number) => void;

  reseed: () => void;
  loadPromptId: (id: string | null) => void;
  reset: () => void;
}

let uidCounter = 0;
export function newUid(): string {
  uidCounter += 1;
  return `in_${Date.now()}_${uidCounter}`;
}

const DEFAULT_RESOLUTION: ResolutionState = {
  base: "1024",
  orientation: "landscape",
  aspect: "3:2",
};

export const useComposer = create<ComposerState>((set) => ({
  mode: "edit",
  modelId: null,
  precision: "bf16",
  prompt: "",
  enhancedPrompt: null,
  negativePrompt: " ",
  inputs: [],
  loras: [],
  resolution: DEFAULT_RESOLUTION,
  steps: 40,
  trueCfgScale: 4.0,
  guidanceScale: 1.0,
  seed: null,
  lockSeed: false,
  batch: 1,
  outputFormat: "png",
  loadedPromptId: null,

  setMode: (mode) =>
    set((s) => ({
      mode,
      // Match-source only valid in edit mode; fall back for generate.
      resolution:
        mode === "generate" && s.resolution.base === "match"
          ? { ...s.resolution, base: "1024" }
          : s.resolution,
    })),
  setModelId: (modelId) => set({ modelId }),
  setPrecision: (precision) => set({ precision }),
  setPrompt: (prompt) => set({ prompt }),
  setEnhancedPrompt: (enhancedPrompt) => set({ enhancedPrompt }),
  setNegativePrompt: (negativePrompt) => set({ negativePrompt }),
  setResolution: (r) => set((s) => ({ resolution: { ...s.resolution, ...r } })),
  setSteps: (steps) => set({ steps }),
  setTrueCfgScale: (trueCfgScale) => set({ trueCfgScale }),
  setSeed: (seed) => set({ seed }),
  setLockSeed: (lockSeed) => set({ lockSeed }),
  setBatch: (batch) => set({ batch: Math.max(1, Math.min(64, batch)) }),
  setOutputFormat: (outputFormat) => set({ outputFormat }),

  addInputs: (imgs) => set((s) => ({ inputs: [...s.inputs, ...imgs] })),
  removeInput: (uid) =>
    set((s) => ({ inputs: s.inputs.filter((i) => i.uid !== uid) })),
  reorderInputs: (from, to) =>
    set((s) => {
      const next = [...s.inputs];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return { inputs: next };
    }),
  clearInputs: () => set({ inputs: [] }),

  setLoras: (loras) => set({ loras }),
  toggleLora: (loraId, weight) =>
    set((s) => {
      const exists = s.loras.find((l) => l.lora_id === loraId);
      if (exists) return { loras: s.loras.filter((l) => l.lora_id !== loraId) };
      return { loras: [...s.loras, { lora_id: loraId, weight }] };
    }),
  setLoraWeight: (loraId, weight) =>
    set((s) => ({
      loras: s.loras.map((l) => (l.lora_id === loraId ? { ...l, weight } : l)),
    })),

  reseed: () => set({ seed: Math.floor(Math.random() * 2 ** 31) }),
  loadPromptId: (loadedPromptId) => set({ loadedPromptId }),

  reset: () =>
    set({
      prompt: "",
      enhancedPrompt: null,
      inputs: [],
      loras: [],
      loadedPromptId: null,
      seed: null,
      batch: 1,
    }),
}));

/** Compute the effective seed used for a run (random if not locked / null). */
export function effectiveSeed(state: ComposerState): number {
  if (state.lockSeed && state.seed !== null) return state.seed;
  if (state.seed !== null) return state.seed;
  return Math.floor(Math.random() * 2 ** 31);
}
