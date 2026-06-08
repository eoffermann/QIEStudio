import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}
interface State {
  error: Error | null;
}

/**
 * Catches render/runtime errors anywhere in the tree and shows a readable fallback instead
 * of a blank (black) screen — and logs the error to the console for debugging. Previously a
 * single API-contract mismatch (e.g. an undefined .map) would unmount the whole app.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Surfaced in the browser console for diagnosis.
    console.error("QIE UI crashed:", error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="flex h-screen w-full flex-col items-center justify-center gap-4 bg-background p-8 text-center">
          <h1 className="text-xl font-semibold text-foreground">Something went wrong</h1>
          <p className="max-w-lg text-sm text-muted-foreground">
            The UI hit an unexpected error and stopped rendering. The details are in the
            browser console.
          </p>
          <pre className="max-h-48 max-w-2xl overflow-auto rounded-lg border bg-card p-3 text-left text-xs text-destructive">
            {this.state.error.message}
          </pre>
          <button
            onClick={() => window.location.reload()}
            className="rounded-xl border border-primary bg-primary/10 px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-primary/20"
          >
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
