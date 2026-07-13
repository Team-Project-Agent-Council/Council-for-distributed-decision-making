"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check, X } from "lucide-react";
import type {
  DemoEvalConfidence,
  DemoHypothesis,
  DemoHypothesisEvaluation,
} from "@/services/api/types";
import { DEMO_AGENT_PROFILES } from "@/services/api/demoAgents";

interface HypothesisMatrixProps {
  /** "Region" or "Country" — used for the section heading. */
  title: string;
  /** Hypothesis objects in the order they should appear as table rows. */
  hypotheses: DemoHypothesis[];
  /** Map keyed by `${hypothesisId}|${agentId}` (matches store layout). */
  evaluations: Record<string, DemoHypothesisEvaluation>;
  /**
   * Optional: support/contradict counts already rolled up by the backend.
   * Shown in the section header ("Spain: 4/5 support · …").
   */
  summary?: { hypothesisId: string; supportCount: number; totalAgents: number }[] | null;
}

const POSITIVE_BG: Record<DemoEvalConfidence, string> = {
  strongly_support: "rgba(198,239,56,0.85)",
  support: "rgba(198,239,56,0.55)",
  neutral: "rgba(120,120,120,0.4)",
  contradicts: "rgba(120,120,120,0.4)",
  strongly_contradicts: "rgba(120,120,120,0.4)",
};

const NEGATIVE_BG: Record<DemoEvalConfidence, string> = {
  strongly_support: "rgba(120,120,120,0.4)",
  support: "rgba(120,120,120,0.4)",
  neutral: "rgba(120,120,120,0.4)",
  contradicts: "rgba(239,68,68,0.55)",
  strongly_contradicts: "rgba(239,68,68,0.85)",
};

const NEUTRAL_BG = "rgba(255,255,255,0.08)";

const AGENT_COLOR_BY_ID: Record<string, string> = Object.fromEntries(
  DEMO_AGENT_PROFILES.map((a) => [a.agentId, a.color])
);

export function HypothesisMatrix({
  title,
  hypotheses,
  evaluations,
  summary,
}: HypothesisMatrixProps) {
  // selected cell key: `${hypothesisId}|${agentId}` — drives the detail panel
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const selectedEval =
    selectedKey && evaluations[selectedKey] ? evaluations[selectedKey] : null;

  function toggleCell(key: string, isFilled: boolean) {
    if (!isFilled) return;
    setSelectedKey((prev) => (prev === key ? null : key));
  }

  if (hypotheses.length === 0) return null;

  return (
    <div
      style={{
        border: "1px solid var(--border)",
        borderRadius: 12,
        overflow: "hidden",
        background: "var(--card, var(--background))",
      }}
    >
      <div
        style={{
          padding: "8px 12px",
          fontSize: 11,
          fontWeight: 700,
          color: "var(--muted-foreground)",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span>{title} — Hypothesis evaluation</span>
        {summary && summary.length > 0 && (
          <span style={{ fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>
            {summary
              .map((s) => {
                // Look up the hypothesis value via id so the badge reads
                // "Spain: 4/5" instead of "country_spain: 4/5".
                const hyp = hypotheses.find((h) => h.id === s.hypothesisId);
                const label = hyp?.value ?? s.hypothesisId;
                return `${label}: ${s.supportCount}/${s.totalAgents}`;
              })
              .join(" · ")}
          </span>
        )}
      </div>

      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: 12,
        }}
      >
        <thead>
          <tr>
            <th style={cellStyle({ header: true, leftCol: true })}>Hypothesis</th>
            {DEMO_AGENT_PROFILES.map((agent) => (
              <th key={agent.agentId} style={cellStyle({ header: true })}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 4,
                    color: agent.color,
                  }}
                >
                  <span>{agent.avatarEmoji}</span>
                  <span style={{ fontSize: 10, fontWeight: 700 }}>
                    {agent.displayName}
                  </span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {hypotheses.map((hyp) => (
            <tr key={hyp.id}>
              <td style={cellStyle({ leftCol: true })}>
                <div style={{ fontWeight: 600 }}>{hyp.value}</div>
              </td>
              {DEMO_AGENT_PROFILES.map((agent) => {
                const key = `${hyp.id}|${agent.agentId}`;
                const eval_ = evaluations[key];
                const isFilled = !!eval_;
                const isSelected = selectedKey === key;
                return (
                  <td key={agent.agentId} style={cellStyle({})}>
                    <AnimatePresence>
                      {isFilled ? (
                        <motion.button
                          key="filled"
                          type="button"
                          onClick={() => toggleCell(key, isFilled)}
                          aria-pressed={isSelected}
                          aria-label={`${eval_.confidence} for ${hyp.value} — ${agent.displayName}. Click for reasoning.`}
                          initial={{ scale: 0, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ type: "spring", stiffness: 300, damping: 20 }}
                          style={{
                            appearance: "none",
                            border: isSelected
                              ? `2px solid ${agent.color}`
                              : "1px solid transparent",
                            cursor: "pointer",
                            display: "inline-flex",
                            alignItems: "center",
                            justifyContent: "center",
                            width: 26,
                            height: 26,
                            borderRadius: 6,
                            background: pickCellBg(eval_),
                            color: "#111",
                            padding: 0,
                            outline: isSelected
                              ? `2px solid ${agent.color}66`
                              : "none",
                            outlineOffset: 1,
                            transition: "border-color 0.15s, outline 0.15s",
                          }}
                        >
                          {cellGlyph(eval_)}
                        </motion.button>
                      ) : (
                        <motion.div
                          key="pending"
                          style={{
                            display: "inline-block",
                            width: 26,
                            height: 26,
                            borderRadius: 6,
                            background: "var(--border)",
                            opacity: 0.4,
                          }}
                        />
                      )}
                    </AnimatePresence>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      <AnimatePresence>
        {selectedEval && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            style={{ overflow: "hidden", borderTop: "1px solid var(--border)" }}
          >
            <EvaluationDetail
              evaluation={selectedEval}
              onClose={() => setSelectedKey(null)}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function pickCellBg(eval_: DemoHypothesisEvaluation): string {
  if (eval_.supports) {
    return POSITIVE_BG[eval_.confidence] ?? POSITIVE_BG.support;
  }
  if (eval_.confidence === "contradicts" || eval_.confidence === "strongly_contradicts") {
    return NEGATIVE_BG[eval_.confidence];
  }
  return NEUTRAL_BG;
}

function cellGlyph(eval_: DemoHypothesisEvaluation) {
  if (eval_.supports) return <Check size={14} />;
  if (
    eval_.confidence === "contradicts" ||
    eval_.confidence === "strongly_contradicts"
  ) {
    return <X size={14} />;
  }
  return <span style={{ fontSize: 10, fontWeight: 700, color: "#fff" }}>·</span>;
}

function EvaluationDetail({
  evaluation,
  onClose,
}: {
  evaluation: DemoHypothesisEvaluation;
  onClose: () => void;
}) {
  const color = AGENT_COLOR_BY_ID[evaluation.agentName] ?? "#888";
  const verdictColor = evaluation.supports
    ? "#c6ef38"
    : evaluation.confidence === "contradicts" ||
      evaluation.confidence === "strongly_contradicts"
    ? "#ef4444"
    : "#9ca3af";
  const verdictText =
    evaluation.confidence === "strongly_support"
      ? "STRONGLY SUPPORTS"
      : evaluation.confidence === "support"
      ? "SUPPORTS"
      : evaluation.confidence === "neutral"
      ? "NEUTRAL"
      : evaluation.confidence === "contradicts"
      ? "CONTRADICTS"
      : "STRONGLY CONTRADICTS";

  return (
    <div
      style={{
        padding: "12px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 8,
        background: "rgba(255,255,255,0.02)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
          <span
            style={{
              fontSize: 10,
              fontWeight: 700,
              color,
              letterSpacing: "0.04em",
              textTransform: "uppercase",
              padding: "2px 8px",
              borderRadius: 999,
              background: color + "22",
              border: `1px solid ${color}55`,
            }}
          >
            {evaluation.agentName}
          </span>
          <span style={{ fontSize: 12, fontWeight: 700, color: "var(--foreground)" }}>
            {evaluation.hypothesisValue}
          </span>
          <span
            style={{
              fontSize: 9,
              fontWeight: 800,
              color: verdictColor,
              letterSpacing: "0.06em",
            }}
          >
            {verdictText}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close reasoning"
          style={{
            appearance: "none",
            border: "1px solid var(--border)",
            background: "transparent",
            color: "var(--muted-foreground)",
            cursor: "pointer",
            fontSize: 10,
            fontWeight: 600,
            padding: "2px 8px",
            borderRadius: 6,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
          }}
        >
          Close
        </button>
      </div>

      <div
        style={{
          fontSize: 12,
          color: "var(--foreground)",
          lineHeight: 1.5,
        }}
      >
        {evaluation.reasoning || "(no reasoning provided)"}
      </div>

      {evaluation.evidence.length > 0 && (
        <ul
          style={{
            margin: 0,
            padding: 0,
            listStyle: "none",
            display: "flex",
            flexWrap: "wrap",
            gap: 4,
          }}
        >
          {evaluation.evidence.map((ev, i) => (
            <li
              key={i}
              style={{
                fontSize: 11,
                padding: "2px 8px",
                borderRadius: 6,
                background: "var(--background)",
                border: "1px solid var(--border)",
                color: "var(--foreground)",
              }}
            >
              {ev}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function cellStyle({
  header = false,
  leftCol = false,
}: {
  header?: boolean;
  leftCol?: boolean;
}): React.CSSProperties {
  return {
    padding: "8px 10px",
    textAlign: leftCol ? "left" : "center",
    background: header ? "rgba(255,255,255,0.03)" : "transparent",
    borderBottom: "1px solid var(--border)",
    fontSize: header ? 11 : 12,
    fontWeight: header ? 700 : 500,
    color: header ? "var(--muted-foreground)" : "var(--foreground)",
    minWidth: leftCol ? 110 : 60,
    width: leftCol ? "auto" : 60,
  };
}
