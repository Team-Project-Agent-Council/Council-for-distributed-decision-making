"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { ToolBadge } from "./ToolBadge";
import { AgentAnalysis } from "./AgentAnalysis";
import type { AgentProfile } from "@/services/api/types";

interface AgentCardProps {
  agent: AgentProfile;
}

export function AgentCard({ agent }: AgentCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="rounded-2xl p-5 cursor-pointer transition-all duration-200"
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
      }}
      onClick={() => setExpanded((v) => !v)}
    >
      <div className="flex items-start gap-4">
        {/* Avatar */}
        <div
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-2xl"
          style={{ background: `${agent.color}15`, border: `1.5px solid ${agent.color}30` }}
        >
          {agent.avatarEmoji}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="font-bold text-sm text-foreground">{agent.displayName}</h3>
            <span
              className="text-xs font-semibold rounded-full px-2 py-0.5"
              style={{ background: `${agent.color}12`, color: agent.color }}
            >
              {agent.specialization}
            </span>
          </div>
          <p className="text-sm text-muted-foreground mt-0.5 leading-snug">{agent.tagline}</p>
        </div>

        <div className="text-muted-foreground/50 shrink-0 mt-0.5">
          {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5 mt-3">
        {agent.tools.map((t) => (
          <ToolBadge key={t} tool={t} />
        ))}
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1, transition: { duration: 0.2 } }}
            exit={{ height: 0, opacity: 0, transition: { duration: 0.15 } }}
            className="overflow-hidden"
          >
            <p className="mt-3 text-sm text-muted-foreground leading-relaxed pt-3 border-t border-border">
              {agent.description}
            </p>

            {(agent.winRate != null || agent.avgAccuracyKm != null) && (
              <div className="mt-2 text-xs text-muted-foreground">
                {agent.winRate != null && (
                  <span>
                    Win Rate: <span className="font-[family-name:var(--font-geist-mono)] font-medium text-foreground">{agent.winRate}%</span>
                  </span>
                )}
                {agent.winRate != null && agent.avgAccuracyKm != null && (
                  <span className="mx-2 text-border">|</span>
                )}
                {agent.avgAccuracyKm != null && (
                  <span>
                    Avg. Accuracy: <span className="font-[family-name:var(--font-geist-mono)] font-medium text-foreground">{agent.avgAccuracyKm} km</span>
                  </span>
                )}
              </div>
            )}

            {agent.exampleAnalysis && (
              <AgentAnalysis text={agent.exampleAnalysis} />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
