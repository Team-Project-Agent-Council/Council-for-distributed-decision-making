"use client";

interface ToolBadgeProps {
  tool: string;
}

export function ToolBadge({ tool }: ToolBadgeProps) {
  return (
    <span className="inline-flex items-center rounded-full border border-border bg-muted/60 px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
      {tool}
    </span>
  );
}
