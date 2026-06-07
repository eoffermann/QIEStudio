import { useState } from "react";
import { KeyRound, Check, X, Loader2, Trash2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  useIntegrations,
  useSetIntegration,
  useDeleteIntegration,
} from "@/api/hooks";
import { toast } from "sonner";
import type { Integration, IntegrationProvider } from "@/api/types";

const PROVIDERS: { id: IntegrationProvider; label: string; help: string }[] = [
  {
    id: "huggingface",
    label: "Hugging Face",
    help: "For gated/private repos and higher rate limits.",
  },
  {
    id: "civitai",
    label: "CivitAI",
    help: "For search/import and mature-content access.",
  },
];

export function IntegrationsTab() {
  const { data } = useIntegrations();
  const setInt = useSetIntegration();
  const delInt = useDeleteIntegration();
  const byProvider = (p: IntegrationProvider): Integration | undefined =>
    data?.find((i) => i.provider === p);

  return (
    <div className="space-y-4 pt-2">
      {PROVIDERS.map((p) => (
        <ProviderRow
          key={p.id}
          provider={p.id}
          label={p.label}
          help={p.help}
          current={byProvider(p.id)}
          onSave={(key) =>
            setInt
              .mutateAsync({ provider: p.id, body: { key } })
              .then(() => toast.success(`${p.label} key saved & validated`))
              .catch((e) => toast.error((e as Error).message))
          }
          onDelete={() =>
            delInt
              .mutateAsync(p.id)
              .then(() => toast.success(`${p.label} key removed`))
          }
          saving={setInt.isPending}
        />
      ))}
    </div>
  );
}

function ProviderRow({
  provider,
  label,
  help,
  current,
  onSave,
  onDelete,
  saving,
}: {
  provider: IntegrationProvider;
  label: string;
  help: string;
  current?: Integration;
  onSave: (key: string) => void;
  onDelete: () => void;
  saving: boolean;
}) {
  const [key, setKey] = useState("");
  const status = current?.status ?? "unset";

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <div>
          <CardTitle className="flex items-center gap-2 text-base">
            <KeyRound className="h-4 w-4 text-primary" /> {label}
          </CardTitle>
          <CardDescription>{help}</CardDescription>
        </div>
        {status === "valid" && (
          <Badge variant="success">
            <Check className="h-3 w-3" /> Connected
          </Badge>
        )}
        {status === "invalid" && (
          <Badge variant="danger">
            <X className="h-3 w-3" /> Invalid
          </Badge>
        )}
        {status === "unset" && <Badge variant="muted">Not set</Badge>}
      </CardHeader>
      <CardContent className="flex items-center gap-2">
        <Input
          type="password"
          placeholder={current?.masked_key ?? `Enter ${label} key`}
          value={key}
          onChange={(e) => setKey(e.target.value)}
          data-provider={provider}
        />
        <Button onClick={() => key && onSave(key)} disabled={saving || !key}>
          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : "Save"}
        </Button>
        {status !== "unset" && (
          <Button variant="ghost" size="icon" onClick={onDelete} title="Remove">
            <Trash2 className="h-4 w-4" />
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
