import { Palette, Monitor } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTheme } from "@/store/theme";
import { useComposer } from "@/store/composer";
import { cn } from "@/lib/utils";

const ACCENTS: { name: string; value: string }[] = [
  { name: "Violet", value: "oklch(0.68 0.19 290)" },
  { name: "Blue", value: "oklch(0.66 0.16 250)" },
  { name: "Cyan", value: "oklch(0.74 0.13 200)" },
  { name: "Emerald", value: "oklch(0.72 0.16 160)" },
  { name: "Amber", value: "oklch(0.78 0.15 75)" },
  { name: "Rose", value: "oklch(0.68 0.2 15)" },
];

export function PreferencesTab() {
  const { theme, setTheme, accent, setAccent } = useTheme();
  const outputFormat = useComposer((s) => s.outputFormat);
  const setOutputFormat = useComposer((s) => s.setOutputFormat);

  return (
    <div className="space-y-4 pt-2">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Palette className="h-4 w-4 text-primary" /> Appearance
          </CardTitle>
          <CardDescription>Theme and accent color.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label>Theme</Label>
            <div className="flex gap-2">
              {(["dark", "light"] as const).map((t) => (
                <Button
                  key={t}
                  variant={theme === t ? "default" : "outline"}
                  size="sm"
                  className="capitalize"
                  onClick={() => setTheme(t)}
                >
                  {t}
                </Button>
              ))}
            </div>
          </div>
          <div className="space-y-1.5">
            <Label>Accent</Label>
            <div className="flex flex-wrap gap-2">
              {ACCENTS.map((a) => (
                <button
                  key={a.value}
                  title={a.name}
                  onClick={() => setAccent(a.value)}
                  className={cn(
                    "h-8 w-8 rounded-full ring-2 ring-offset-2 ring-offset-background transition-transform hover:scale-110",
                    accent === a.value ? "ring-foreground" : "ring-transparent",
                  )}
                  style={{ backgroundColor: a.value }}
                />
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Monitor className="h-4 w-4 text-primary" /> Output
          </CardTitle>
          <CardDescription>Default output format for new runs.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-w-xs space-y-1.5">
            <Label>Format</Label>
            <Select
              value={outputFormat}
              onValueChange={(v) => setOutputFormat(v as "png" | "webp" | "jpeg")}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="png">PNG (lossless + metadata)</SelectItem>
                <SelectItem value="webp">WebP</SelectItem>
                <SelectItem value="jpeg">JPEG</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Defaults</CardTitle>
          <CardDescription>
            Default model/precision and device selection.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-xl border border-dashed p-3 text-xs text-muted-foreground">
            TODO: wire default-model/precision and device override to the backend
            settings endpoint (DESIGN §5.7).
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
