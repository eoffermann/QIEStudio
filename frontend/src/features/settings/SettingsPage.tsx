import { PageHeader } from "@/components/PageHeader";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { IntegrationsTab } from "./IntegrationsTab";
import { PreferencesTab } from "./PreferencesTab";

export function SettingsPage() {
  return (
    <div className="space-y-5 p-6">
      <PageHeader
        title="Settings"
        description="Integrations, defaults, and appearance."
      />
      <Tabs defaultValue="integrations" className="max-w-2xl">
        <TabsList>
          <TabsTrigger value="integrations">Integrations</TabsTrigger>
          <TabsTrigger value="preferences">Preferences</TabsTrigger>
        </TabsList>
        <TabsContent value="integrations">
          <IntegrationsTab />
        </TabsContent>
        <TabsContent value="preferences">
          <PreferencesTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
