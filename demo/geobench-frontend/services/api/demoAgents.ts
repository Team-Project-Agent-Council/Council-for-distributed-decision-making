import type { DemoAgentId, DemoAgentProfile } from "./types";

/**
 * Static metadata for the 5 specialist agents used by the Progressive
 * Narrowing demo. Mirrors the agent names in the vendored vlm_council
 * (linguistic, landscape, botanics, regulatory, meta).
 *
 * Colours stay synced with the per-country fills + dot clusters drawn by
 * `DemoMap.tsx`, so the same agent always renders in the same hue across
 * the map, the agent panel, and the hypothesis matrix.
 */
export const DEMO_AGENT_PROFILES: DemoAgentProfile[] = [
  {
    agentId: "linguistic",
    displayName: "Linguistic",
    tagline: "Reads scripts, languages, and signs",
    specialization: "Language & script evidence",
    avatarEmoji: "🔤",
    color: "#8b5cf6",
  },
  {
    agentId: "landscape",
    displayName: "Landscape",
    tagline: "Terrain, climate, settlement patterns",
    specialization: "Physical geography",
    avatarEmoji: "🏔️",
    color: "#3b82f6",
  },
  {
    agentId: "botanics",
    displayName: "Botanics",
    tagline: "Vegetation, crops, endemic species",
    specialization: "Flora identification",
    avatarEmoji: "🌿",
    color: "#22c55e",
  },
  {
    agentId: "regulatory",
    displayName: "Regulatory",
    tagline: "Roads, signs, license plates",
    specialization: "Infrastructure & traffic",
    avatarEmoji: "🛑",
    color: "#ef4444",
  },
  {
    agentId: "meta",
    displayName: "Meta",
    tagline: "Country-specific visual details",
    specialization: "GeoGuessr meta",
    avatarEmoji: "📍",
    color: "#f59e0b",
  },
];

export const DEMO_AGENT_IDS: DemoAgentId[] = [
  "linguistic",
  "landscape",
  "botanics",
  "regulatory",
  "meta",
];
