"use client";

import { motion } from "framer-motion";
import { AgentCard } from "./AgentCard";
import type { AgentProfile } from "@/services/api/types";

interface AgentGridProps {
  agents: AgentProfile[];
}

export function AgentGrid({ agents }: AgentGridProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {agents.map((agent, i) => (
        <motion.div
          key={agent.agentId}
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{
            duration: 0.5,
            ease: [0.16, 1, 0.3, 1],
            delay: i * 0.08,
          }}
        >
          <AgentCard agent={agent} />
        </motion.div>
      ))}
    </div>
  );
}
