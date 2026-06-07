import type { ComponentType, ReactNode } from "react";
import type { LucideProps } from "lucide-react";

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: ComponentType<LucideProps>;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed py-16 text-center">
      <div className="grid h-12 w-12 place-items-center rounded-2xl bg-muted text-muted-foreground">
        <Icon className="h-6 w-6" />
      </div>
      <div>
        <div className="font-medium">{title}</div>
        {description && (
          <div className="mt-1 max-w-sm text-sm text-muted-foreground">{description}</div>
        )}
      </div>
      {action}
    </div>
  );
}
