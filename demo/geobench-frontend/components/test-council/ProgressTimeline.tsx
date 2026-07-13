"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Check, SkipForward } from "lucide-react";
import type { DemoPhase } from "@/stores/demoStore";

/**
 * View-step keys map to the Progressive Narrowing pipeline phases.
 *
 *   "phase1"           — 5 specialists' initial assessments
 *   "region_consensus" — judge.check_region_consensus (always runs)
 *   "region_eval"      — region hypothesis evaluation (Path B only)
 *   "country_assess"   — region-constrained re-assessment (Path B only)
 *   "country_eval"     — country hypothesis evaluation (both paths)
 *   "final"            — judge.decide_country
 *
 * Path A (region consensus reached) skips region_eval + country_assess.
 * The stepper still renders them as "skipped" so the user can see what was
 * bypassed.
 */
export type DemoViewStep =
  | "phase1"
  | "region_consensus"
  | "region_eval"
  | "country_assess"
  | "country_eval"
  | "final";

interface ProgressTimelineProps {
  /** Live pipeline phase from the store. Drives which steps are unlocked. */
  phase: DemoPhase;
  /**
   * Whether the run took Path A (region consensus reached). When true, the
   * region_eval and country_assess steps are rendered as "skipped" stripes.
   * When null (not yet known), they render as locked.
   */
  pathAConsensus: boolean | null;
  /** Currently focused step (controls highlighting). */
  viewStep: DemoViewStep;
  /** Whether the user can scrub between steps (true once past consensus). */
  scrubbable: boolean;
  /** Click handler — only fires for unlocked, non-skipped steps. */
  onStepClick: (step: DemoViewStep) => void;
}

interface Step {
  key: DemoViewStep;
  label: string;
  description: string;
}

const STEPS: Step[] = [
  { key: "phase1", label: "Phase 1", description: "Specialist assessment" },
  { key: "region_consensus", label: "Consensus", description: "Region consensus check" },
  { key: "region_eval", label: "Region", description: "Region hypothesis eval" },
  { key: "country_assess", label: "Constrained", description: "Region-bound assessment" },
  { key: "country_eval", label: "Country", description: "Country hypothesis eval" },
  { key: "final", label: "Final", description: "Country determination" },
];

export function ProgressTimeline({
  phase,
  pathAConsensus,
  viewStep,
  scrubbable,
  onStepClick,
}: ProgressTimelineProps) {
  // Track hovered step in React state instead of mutating button.style — direct
  // DOM mutation during render-time is prone to hydration mismatches in
  // App-Router + Turbopack.
  const [hoveredStep, setHoveredStep] = useState<DemoViewStep | null>(null);

  const maxUnlockedIdx = computeMaxUnlocked(phase);
  const activeIdx = STEPS.findIndex((s) => s.key === viewStep);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "stretch",
        gap: 0,
        padding: "12px 16px",
        background: "var(--card, var(--background))",
        border: "1px solid var(--border)",
        borderRadius: 12,
      }}
    >
      {STEPS.map((step, i) => {
        const isPathASkipped =
          pathAConsensus === true &&
          (step.key === "region_eval" || step.key === "country_assess");
        const isUnlocked = i <= maxUnlockedIdx && !isPathASkipped;
        const isCompleted = isPathASkipped
          ? i <= maxUnlockedIdx // skipped steps "complete" once we've passed them
          : i < maxUnlockedIdx ||
            (i === maxUnlockedIdx && phase === "done");
        const isActive = i === activeIdx;
        const isClickable = scrubbable && isUnlocked;

        const accent = isPathASkipped
          ? "var(--muted-foreground)"
          : isActive
          ? "#c6ef38"
          : isCompleted
          ? "#c6ef38"
          : isUnlocked
          ? "var(--foreground)"
          : "var(--muted-foreground)";

        return (
          <div
            key={step.key}
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              gap: 10,
              minWidth: 0,
            }}
          >
            <button
              type="button"
              disabled={!isClickable}
              onClick={() => isClickable && onStepClick(step.key)}
              onMouseEnter={() =>
                isClickable ? setHoveredStep(step.key) : undefined
              }
              onMouseLeave={() =>
                hoveredStep === step.key ? setHoveredStep(null) : undefined
              }
              title={
                isPathASkipped
                  ? `${step.label} — skipped (region consensus reached)`
                  : isClickable
                  ? `Show ${step.label}`
                  : isUnlocked
                  ? step.label
                  : "Not yet reached"
              }
              style={{
                appearance: "none",
                border: "none",
                outline: "none",
                font: "inherit",
                color: "inherit",
                textAlign: "left",
                background:
                  isClickable && hoveredStep === step.key
                    ? "rgba(198,239,56,0.07)"
                    : "transparent",
                display: "flex",
                alignItems: "center",
                gap: 10,
                cursor: isClickable
                  ? "pointer"
                  : isPathASkipped
                  ? "not-allowed"
                  : "default",
                opacity: isPathASkipped ? 0.45 : isUnlocked ? 1 : 0.55,
                flex: 1,
                minWidth: 0,
                padding: "2px 4px",
                borderRadius: 8,
                transition: "background 0.2s, opacity 0.2s",
              }}
            >
              <motion.div
                initial={false}
                animate={{
                  scale: isActive && phase !== "done" ? [1, 1.1, 1] : 1,
                  background: isPathASkipped
                    ? "var(--border)"
                    : isActive
                    ? "rgba(198,239,56,0.18)"
                    : isCompleted
                    ? "#c6ef38"
                    : "var(--border)",
                }}
                transition={{
                  scale: {
                    repeat: isActive && phase !== "done" ? Infinity : 0,
                    duration: 1.2,
                  },
                  background: { duration: 0.3 },
                }}
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: "50%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  border: `2px solid ${
                    isPathASkipped
                      ? "var(--border)"
                      : isActive || isCompleted
                      ? "#c6ef38"
                      : "var(--border)"
                  }`,
                  color: isCompleted && !isActive ? "#111" : "#fff",
                  flexShrink: 0,
                  fontWeight: 700,
                  fontSize: 12,
                }}
              >
                {isPathASkipped ? (
                  <SkipForward size={12} />
                ) : isCompleted && !isActive ? (
                  <Check size={14} />
                ) : (
                  i + 1
                )}
              </motion.div>

              <div style={{ minWidth: 0, flex: 1, textAlign: "left" }}>
                <div
                  style={{
                    fontWeight: 700,
                    fontSize: 12,
                    color: accent,
                    letterSpacing: "0.04em",
                    textTransform: "uppercase",
                  }}
                >
                  {step.label}
                </div>
                <div
                  style={{
                    fontSize: 11,
                    color: "var(--muted-foreground)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {isPathASkipped ? "Skipped — Path A" : step.description}
                </div>
              </div>
            </button>

            {i < STEPS.length - 1 && (
              <div
                style={{
                  width: 14,
                  height: 1,
                  background:
                    i < maxUnlockedIdx ? "#c6ef38" : "var(--border)",
                  margin: "0 4px",
                  flexShrink: 0,
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/**
 * Pipeline phase → highest unlocked step index. Skipped steps are still
 * "passed" (they bump the unlocked cursor), so the visual "completed" state
 * stays consistent with what's actually been processed by the backend.
 */
function computeMaxUnlocked(phase: DemoPhase): number {
  switch (phase) {
    case "idle":
    case "uploading":
    case "error":
      return -1;
    case "phase1":
      return 0;
    case "region_consensus":
      return 1;
    case "region_hypotheses":
    case "region_evaluation":
      return 2;
    case "region_decision":
    case "country_assess":
      return 3;
    case "country_hypotheses":
    case "country_evaluation":
      return 4;
    case "final":
    case "done":
      return 5;
    default:
      return -1;
  }
}
