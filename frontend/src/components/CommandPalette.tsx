import { useNavigate } from "react-router-dom";
import {
  Wand2,
  Images,
  MessageSquareText,
  Layers,
  History,
  Settings,
  Dices,
  Sparkles,
  SunMoon,
} from "lucide-react";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { useUi } from "@/store/ui";
import { useComposer } from "@/store/composer";
import { useTheme } from "@/store/theme";

export function CommandPalette() {
  const { commandOpen, setCommandOpen } = useUi();
  const navigate = useNavigate();
  const reseed = useComposer((s) => s.reseed);
  const toggleTheme = useTheme((s) => s.toggleTheme);

  const run = (fn: () => void) => {
    setCommandOpen(false);
    fn();
  };

  return (
    <CommandDialog open={commandOpen} onOpenChange={setCommandOpen}>
      <CommandInput placeholder="Type a command or search…" />
      <CommandList>
        <CommandEmpty>No results.</CommandEmpty>
        <CommandGroup heading="Navigate">
          <CommandItem onSelect={() => run(() => navigate("/compose"))}>
            <Wand2 className="h-4 w-4" /> Composer
          </CommandItem>
          <CommandItem onSelect={() => run(() => navigate("/images"))}>
            <Images className="h-4 w-4" /> Images library
          </CommandItem>
          <CommandItem onSelect={() => run(() => navigate("/prompts"))}>
            <MessageSquareText className="h-4 w-4" /> Prompts library
          </CommandItem>
          <CommandItem onSelect={() => run(() => navigate("/loras"))}>
            <Layers className="h-4 w-4" /> LoRAs
          </CommandItem>
          <CommandItem onSelect={() => run(() => navigate("/history"))}>
            <History className="h-4 w-4" /> History
          </CommandItem>
          <CommandItem onSelect={() => run(() => navigate("/settings"))}>
            <Settings className="h-4 w-4" /> Settings
          </CommandItem>
        </CommandGroup>
        <CommandGroup heading="Actions">
          <CommandItem
            onSelect={() =>
              run(() => window.dispatchEvent(new CustomEvent("qie:generate")))
            }
          >
            <Sparkles className="h-4 w-4" /> Generate now
            <span className="ml-auto text-xs text-muted-foreground">G</span>
          </CommandItem>
          <CommandItem onSelect={() => run(reseed)}>
            <Dices className="h-4 w-4" /> Reseed
            <span className="ml-auto text-xs text-muted-foreground">R</span>
          </CommandItem>
          <CommandItem onSelect={() => run(toggleTheme)}>
            <SunMoon className="h-4 w-4" /> Toggle theme
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
