"use client";

import type { CollaborationStep, AgentProfile } from "@/services/api/types";

interface CollaborationFlowProps {
  steps: CollaborationStep[];
  // Reserved for future use — the previous version rendered agent emojis
  // inside each step, but the collaboration overview now stays text-only.
  agents?: AgentProfile[];
}

export function CollaborationFlow({ steps }: CollaborationFlowProps) {
  return (
    <div className="rounded-2xl p-6 sm:p-8" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
      <h2 className="text-2xl font-bold mb-8">How the Council Works</h2>

      <div className="hidden lg:block">
        <div className="flex items-start gap-0">
          {steps.map((step, i) => (
            <div key={step.stepNumber} className="flex-1 flex flex-col items-center text-center px-2">
              <div className="flex items-center w-full mb-4">
                <div className={`flex-1 h-0.5 ${i === 0 ? "invisible" : "bg-border"}`} />
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-bold shadow-sm">
                  {step.stepNumber}
                </div>
                <div className={`flex-1 h-0.5 ${i === steps.length - 1 ? "invisible" : "bg-border"}`} />
              </div>
              <h3 className="font-semibold text-sm mb-1">{step.title}</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">{step.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="lg:hidden space-y-6">
        {steps.map((step, i) => (
          <div key={step.stepNumber} className="flex gap-4">
            <div className="flex flex-col items-center">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-bold">
                {step.stepNumber}
              </div>
              {i < steps.length - 1 && <div className="flex-1 w-0.5 bg-border mt-2 min-h-[2rem]" />}
            </div>
            <div className="pb-2">
              <h3 className="font-semibold text-sm">{step.title}</h3>
              <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{step.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
