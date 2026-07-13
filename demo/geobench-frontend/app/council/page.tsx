"use client";

import { useEffect, useState } from "react";
import { CouncilHero } from "@/components/council/CouncilHero";
import { AgentGrid } from "@/components/council/AgentGrid";
import { CollaborationFlow } from "@/components/council/CollaborationFlow";
import { councilService } from "@/services/api/councilService";
import type { CouncilInfo } from "@/services/api/types";
import { CouncilSkeleton } from "@/components/ui/page-skeleton";
import { motion } from "framer-motion";
import { fadeIn } from "@/lib/animations";

export default function CouncilPage() {
  const [info, setInfo] = useState<CouncilInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    councilService.getCouncilInfo().then((d) => {
      setInfo(d);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return <CouncilSkeleton />;
  }

  if (!info) return null;

  return (
    <motion.div
      variants={fadeIn}
      initial="hidden"
      animate="visible"
      className="mx-auto max-w-7xl px-4 sm:px-6 py-10 space-y-10"
    >
      <CouncilHero />

      <div>
        <h2 className="text-2xl font-black mb-6">The Agents</h2>
        <AgentGrid agents={info.agents} />
      </div>

      <CollaborationFlow steps={info.collaborationSteps} agents={info.agents} />
    </motion.div>
  );
}
