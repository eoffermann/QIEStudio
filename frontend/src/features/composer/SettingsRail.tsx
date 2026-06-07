import { useEffect, useMemo } from "react";
import { Boxes, Dices, Lock, LockOpen } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useModels } from "@/api/hooks";
import { useComposer } from "@/store/composer";
import { PrecisionAdvisor } from "./PrecisionAdvisor";
import { ResolutionPicker } from "./ResolutionPicker";
import { LoraPicker } from "./LoraPicker";

export function SettingsRail() {
  const { data: models } = useModels();
  const mode = useComposer((s) => s.mode);
  const modelId = useComposer((s) => s.modelId);
  const setModelId = useComposer((s) => s.setModelId);
  const steps = useComposer((s) => s.steps);
  const setSteps = useComposer((s) => s.setSteps);
  const cfg = useComposer((s) => s.trueCfgScale);
  const setCfg = useComposer((s) => s.setTrueCfgScale);
  const negative = useComposer((s) => s.negativePrompt);
  const setNegative = useComposer((s) => s.setNegativePrompt);
  const seed = useComposer((s) => s.seed);
  const setSeed = useComposer((s) => s.setSeed);
  const lockSeed = useComposer((s) => s.lockSeed);
  const setLockSeed = useComposer((s) => s.setLockSeed);
  const reseed = useComposer((s) => s.reseed);
  const batch = useComposer((s) => s.batch);
  const setBatch = useComposer((s) => s.setBatch);

  const modeModels = useMemo(() => (models ? models[mode] : []), [models, mode]);

  // Default the model to this mode's default when it changes.
  useEffect(() => {
    if (!modeModels.length) return;
    const stillValid = modeModels.some((m) => m.id === modelId);
    if (!stillValid) {
      const def = modeModels.find((m) => m.is_default) ?? modeModels[0];
      setModelId(def.id);
    }
  }, [mode, modeModels, modelId, setModelId]);

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="space-y-0 pb-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Boxes className="h-4 w-4 text-primary" /> Settings
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 space-y-5 overflow-y-auto">
        {/* Model */}
        <div className="space-y-1">
          <Label>Model</Label>
          <Select value={modelId ?? ""} onValueChange={setModelId}>
            <SelectTrigger className="h-9">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              {modeModels.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  {m.label}
                  {m.is_default ? " (default)" : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <PrecisionAdvisor />
        <ResolutionPicker />
        <LoraPicker />

        {/* Steps */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <Label>Steps</Label>
            <span className="font-mono text-xs">{steps}</span>
          </div>
          <Slider
            value={[steps]}
            min={1}
            max={100}
            step={1}
            onValueChange={([v]) => setSteps(v)}
          />
        </div>

        {/* CFG */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <Label>true_cfg_scale</Label>
            <span className="font-mono text-xs">{cfg.toFixed(1)}</span>
          </div>
          <Slider
            value={[cfg]}
            min={1}
            max={10}
            step={0.5}
            onValueChange={([v]) => setCfg(v)}
          />
        </div>

        {/* Negative */}
        <div className="space-y-1">
          <Label>Negative prompt</Label>
          <Input
            value={negative}
            onChange={(e) => setNegative(e.target.value)}
            placeholder="(a single space by default)"
          />
        </div>

        {/* Seed */}
        <div className="space-y-1">
          <Label>Seed</Label>
          <div className="flex gap-2">
            <Input
              type="number"
              value={seed ?? ""}
              placeholder="random"
              onChange={(e) =>
                setSeed(e.target.value === "" ? null : Number(e.target.value))
              }
            />
            <Button
              variant="outline"
              size="icon"
              onClick={() => setLockSeed(!lockSeed)}
              title={lockSeed ? "Seed locked" : "Seed random"}
            >
              {lockSeed ? <Lock className="h-4 w-4" /> : <LockOpen className="h-4 w-4" />}
            </Button>
            <Button variant="outline" size="icon" onClick={reseed} title="Reseed (R)">
              <Dices className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Batch */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <Label>Batch count</Label>
            <span className="font-mono text-xs">{batch}</span>
          </div>
          <Slider
            value={[batch]}
            min={1}
            max={16}
            step={1}
            onValueChange={([v]) => setBatch(v)}
          />
        </div>
      </CardContent>
    </Card>
  );
}
