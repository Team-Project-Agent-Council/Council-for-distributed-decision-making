"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle, ChevronDown, ChevronUp, Eye, EyeOff, Loader2 } from "lucide-react";
import type {
  DemoAgentAssessment,
  DemoAgentId,
  DemoAgentProfile,
  DemoCandidate,
} from "@/services/api/types";

interface DemoAgentPanelProps {
  agents: DemoAgentProfile[];
  assessments: Partial<Record<DemoAgentId, DemoAgentAssessment>>;
  ready: Record<DemoAgentId, boolean>;
  active: boolean;
  /** Which agents are currently shown on the map. */
  selected: Set<DemoAgentId>;
  /** Click handler — toggles a single agent. */
  onToggle: (agentId: DemoAgentId) => void;
}

export function DemoAgentPanel({
  agents,
  assessments,
  ready,
  active,
  selected,
  onToggle,
}: DemoAgentPanelProps) {
  // Track which agent cards are expanded to show full per-candidate reasoning.
  const [expanded, setExpanded] = useState<Set<DemoAgentId>>(new Set());

  function toggleExpanded(id: DemoAgentId) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {agents.map((agent) => {
        const assessment = assessments[agent.agentId];
        const isReady = ready[agent.agentId] === true;
        const isThinking = active && !isReady;
        const status: "idle" | "thinking" | "ready" = !active
          ? "idle"
          : isReady
          ? "ready"
          : "thinking";
        const isSelected = selected.has(agent.agentId);
        const isExpanded = expanded.has(agent.agentId);
        const cardOpacity = isSelected ? 1 : 0.42;
        const hasReasoning = !!assessment && assessment.candidates.length > 0;

        return (
          <div
            key={agent.agentId}
            role="button"
            tabIndex={0}
            aria-pressed={isSelected}
            onClick={() => onToggle(agent.agentId)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onToggle(agent.agentId);
              }
            }}
            title={
              isSelected
                ? `Hide ${agent.displayName} on the map`
                : `Show ${agent.displayName} on the map`
            }
            style={{
              cursor: "pointer",
              padding: "10px 12px",
              borderRadius: 12,
              background: "var(--background)",
              border: `1px solid ${
                isSelected
                  ? status === "ready"
                    ? agent.color + "80"
                    : agent.color + "40"
                  : "var(--border)"
              }`,
              boxShadow: isSelected ? `inset 0 0 0 1px ${agent.color}30` : "none",
              transition: "border-color 0.2s, opacity 0.2s, box-shadow 0.2s",
              display: "flex",
              flexDirection: "column",
              gap: 8,
              opacity: cardOpacity,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: "50%",
                  flexShrink: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 16,
                  background:
                    status === "idle" ? "var(--border)" : agent.color + "22",
                  border: `1.5px solid ${
                    status === "idle" ? "var(--border)" : agent.color
                  }`,
                  opacity: status === "idle" ? 0.5 : 1,
                  transition: "all 0.3s",
                }}
              >
                {agent.avatarEmoji}
              </div>
              <div style={{ flex: 1, minWidth: 0, textAlign: "left" }}>
                <div
                  style={{
                    fontWeight: 700,
                    fontSize: 13,
                    color:
                      status === "idle"
                        ? "var(--muted-foreground)"
                        : "var(--foreground)",
                  }}
                >
                  {agent.displayName}
                </div>
                <div
                  style={{
                    fontSize: 11,
                    color:
                      status === "idle" ? "var(--muted-foreground)" : agent.color,
                    fontWeight: 500,
                  }}
                >
                  {status === "thinking"
                    ? "Analysing…"
                    : status === "ready"
                    ? agent.tagline
                    : agent.specialization}
                </div>
              </div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  flexShrink: 0,
                }}
              >
                {hasReasoning && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleExpanded(agent.agentId);
                    }}
                    aria-expanded={isExpanded}
                    aria-label={
                      isExpanded
                        ? `Hide ${agent.displayName} reasoning`
                        : `Show ${agent.displayName} reasoning`
                    }
                    style={{
                      appearance: "none",
                      background: isExpanded
                        ? agent.color + "22"
                        : "transparent",
                      border: `1px solid ${
                        isExpanded ? agent.color + "60" : "var(--border)"
                      }`,
                      borderRadius: 6,
                      padding: "2px 6px",
                      cursor: "pointer",
                      color: isExpanded ? agent.color : "var(--muted-foreground)",
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 2,
                      fontSize: 10,
                      fontWeight: 700,
                      textTransform: "uppercase",
                      letterSpacing: "0.04em",
                    }}
                  >
                    {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    Why
                  </button>
                )}
                <div
                  style={{
                    width: 20,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {isThinking && (
                    <Loader2
                      size={16}
                      style={{
                        color: agent.color,
                        animation: "spin 1s linear infinite",
                      }}
                    />
                  )}
                  {!isThinking && isReady && isSelected && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 300 }}
                    >
                      <CheckCircle size={16} style={{ color: agent.color }} />
                    </motion.div>
                  )}
                  {!isThinking && !isSelected && (
                    <EyeOff size={15} style={{ color: "var(--muted-foreground)" }} />
                  )}
                  {!isThinking && !isReady && isSelected && (
                    <Eye size={15} style={{ color: "var(--muted-foreground)" }} />
                  )}
                </div>
              </div>
            </div>

            <AnimatePresence>
              {hasReasoning && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  style={{ display: "flex", flexWrap: "wrap", gap: 4 }}
                >
                  {assessment!.candidates.slice(0, 3).map((cand, i) => (
                    <motion.div
                      key={`${cand.country}-${i}`}
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.08 }}
                      title={cand.reasoning}
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 4,
                        padding: "3px 8px",
                        borderRadius: 999,
                        background: agent.color + "1f",
                        border: `1px solid ${agent.color}66`,
                        fontSize: 11,
                        fontWeight: 600,
                        color: "var(--foreground)",
                      }}
                    >
                      {cand.country}
                      <span
                        style={{
                          fontSize: 9,
                          fontWeight: 700,
                          color: agent.color,
                          letterSpacing: "0.04em",
                          textTransform: "uppercase",
                        }}
                      >
                        {cand.confidence}
                      </span>
                    </motion.div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {hasReasoning && isExpanded && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  style={{ overflow: "hidden" }}
                >
                  <ReasoningDetail
                    color={agent.color}
                    candidates={assessment!.candidates}
                    evidence={assessment!.evidence}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}
    </div>
  );
}

function ReasoningDetail({
  color,
  candidates,
  evidence,
}: {
  color: string;
  candidates: DemoCandidate[];
  evidence: string[];
}) {
  return (
    <div
      style={{
        marginTop: 4,
        padding: "10px 12px",
        borderRadius: 10,
        background: color + "0d",
        border: `1px solid ${color}33`,
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {candidates.map((cand, i) => (
          <div
            key={`${cand.country}-${i}`}
            style={{ display: "flex", flexDirection: "column", gap: 3 }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontSize: 12,
                fontWeight: 700,
                color: "var(--foreground)",
              }}
            >
              <span>{cand.country}</span>
              <span
                style={{
                  fontSize: 9,
                  fontWeight: 700,
                  color,
                  letterSpacing: "0.04em",
                  textTransform: "uppercase",
                  padding: "1px 6px",
                  borderRadius: 999,
                  background: color + "20",
                  border: `1px solid ${color}55`,
                }}
              >
                {cand.confidence}
              </span>
            </div>
            <div
              style={{
                fontSize: 11,
                color: "var(--muted-foreground)",
                lineHeight: 1.5,
              }}
            >
              {cand.reasoning || "(no reasoning provided)"}
            </div>
          </div>
        ))}
      </div>

      {evidence.length > 0 && (
        <div
          style={{
            paddingTop: 8,
            borderTop: `1px dashed ${color}33`,
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          <div
            style={{
              fontSize: 10,
              fontWeight: 700,
              color: "var(--muted-foreground)",
              letterSpacing: "0.06em",
              textTransform: "uppercase",
            }}
          >
            Evidence
          </div>
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
            {evidence.map((ev, i) => (
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
        </div>
      )}
    </div>
  );
}
