// =============================================================================
// GeoBench frontend type definitions
// =============================================================================
//
// Two independent domains live here:
//
//   1. Council metadata (`AgentProfile`, `CollaborationStep`, `CouncilInfo`)
//      — powers the `/council` page. Fetched once from
//      `GET /api/council/agents` and rendered statically.
//
//   2. Progressive Narrowing demo (`Demo*` types) — the live SSE stream
//      from `/api/demo/*`. Schema mirrors the events emitted by
//      `geobench-backend/council_adapters/progressive_narrowing_adapter.py`
//      (vendored vlm_council pipeline).

export interface LatLng {
  lat: number;
  lng: number;
}

// =============================================================================
// Council metadata
// =============================================================================

/**
 * String id for a council agent. On the demo path the concrete values are
 * `linguistic | landscape | botanics | regulatory | meta` (`DemoAgentId`),
 * but the council metadata page keeps the type open so future extensions
 * can add more specialists without a type break.
 */
export type AgentId = string;

export interface AgentProfile {
  agentId: AgentId;
  displayName: string;
  tagline: string;
  description: string;
  avatarEmoji: string;
  tools: string[];
  specialization: string;
  color: string;
  /** Optional aggregate stats — reserved for future dashboards. */
  winRate?: number;
  avgAccuracyKm?: number;
  exampleAnalysis?: string;
}

export interface CollaborationStep {
  stepNumber: number;
  title: string;
  description: string;
  agentsInvolved: AgentId[];
}

export interface CouncilInfo {
  agents: AgentProfile[];
  collaborationSteps: CollaborationStep[];
}

// =============================================================================
// Progressive Narrowing demo (VLM-Council)
// =============================================================================

export type DemoAgentId =
  | "linguistic"
  | "landscape"
  | "botanics"
  | "regulatory"
  | "meta";

/** Phase-1 candidate confidence — legacy "high|medium|low|speculative" scale. */
export type DemoCandidateConfidence = "high" | "medium" | "low" | "speculative";

/** Hypothesis-evaluation confidence — 5-level scale from vlm_council. */
export type DemoEvalConfidence =
  | "strongly_support"
  | "support"
  | "neutral"
  | "contradicts"
  | "strongly_contradicts";

/** Hypothesis level — region narrowing happens before country narrowing. */
export type DemoHypothesisLevel = "region" | "country";

export interface DemoCandidate {
  country: string;
  lat: number;
  lng: number;
  confidence: DemoCandidateConfidence;
  reasoning: string;
}

export interface DemoAgentAssessment {
  agentName: DemoAgentId;
  candidates: DemoCandidate[];
  evidence: string[];
}

export interface DemoHypothesis {
  id: string; // e.g. "region_western_europe" / "country_spain"
  level: DemoHypothesisLevel;
  value: string; // "Western Europe" / "Spain"
  statement: string; // "This image is from Western Europe"
}

export interface DemoHypothesisEvaluation {
  agentName: DemoAgentId;
  hypothesisId: string;
  hypothesisValue: string;
  level: DemoHypothesisLevel | "";
  confidence: DemoEvalConfidence;
  /** Convenience flag — confidence in {strongly_support, support}. */
  supports: boolean;
  reasoning: string;
  evidence: string[];
}

export interface DemoEvaluationSummaryEntry {
  hypothesisId: string;
  hypothesisValue: string;
  supportCount: number;
  totalAgents: number;
  byConfidence: Partial<Record<DemoEvalConfidence, number>>;
}

export interface DemoFinalResult {
  country: string;
  lat: number;
  lng: number;
  reasoning: string;
}

export interface DemoGroundTruth {
  lat: number;
  lng: number;
  label: string;
}

export interface DemoRunStartPayload {
  runId: string;
  imageUrl: string;
  groundTruth?: DemoGroundTruth;
}

export interface DemoRegionConsensusResult {
  consensus: boolean;
  confirmedRegion: string | null;
  proposedRegions: string[];
  /** `{region: {country: agent_count}}` — built by `judge.check_region_consensus`. */
  regionCandidates: Record<string, Record<string, number>>;
}

export interface DemoRegionDecision {
  confirmedRegion: string;
  reasoning: string;
}

/**
 * Discriminated union of every SSE event the backend emits during a
 * Progressive Narrowing run. Order (Path B): `run_started → phase1_started →
 * agent_assessment × 5 → region_consensus_result → region_hypotheses_generated
 * → region_evaluation × N×5 → region_evaluation_complete → region_decision →
 * country_assessment × 5 → country_hypotheses_generated → country_evaluation ×
 * N×5 → country_evaluation_complete → final_started → final_result → done`.
 * Path A skips everything between `region_consensus_result` and
 * `country_hypotheses_generated`.
 */
export type DemoSseEvent =
  | { type: "run_started"; ts: number; data: DemoRunStartPayload }
  | { type: "phase1_started"; ts: number; data: { agents: DemoAgentId[] } }
  | { type: "agent_assessment"; ts: number; data: DemoAgentAssessment }
  | { type: "region_consensus_result"; ts: number; data: DemoRegionConsensusResult }
  | { type: "region_hypotheses_generated"; ts: number; data: { hypotheses: DemoHypothesis[] } }
  | { type: "region_evaluation"; ts: number; data: DemoHypothesisEvaluation }
  | { type: "region_evaluation_complete"; ts: number; data: { summary: DemoEvaluationSummaryEntry[] } }
  | { type: "region_decision"; ts: number; data: DemoRegionDecision }
  | { type: "country_assessment"; ts: number; data: DemoAgentAssessment }
  | { type: "country_hypotheses_generated"; ts: number; data: { hypotheses: DemoHypothesis[] } }
  | { type: "country_evaluation"; ts: number; data: DemoHypothesisEvaluation }
  | { type: "country_evaluation_complete"; ts: number; data: { summary: DemoEvaluationSummaryEntry[] } }
  | { type: "final_started"; ts: number; data: Record<string, never> }
  | { type: "final_result"; ts: number; data: DemoFinalResult }
  | { type: "error"; ts: number; data: { message: string } }
  | { type: "done"; ts: number; data: Record<string, never> };

export type DemoSseEventType = DemoSseEvent["type"];

export interface DemoStartRunInput {
  file?: File;
  datasetId?: string;
}

export interface DemoStartRunResponse {
  runId: string;
  imageUrl: string;
  groundTruth?: DemoGroundTruth;
}

export interface DemoRandomLocation {
  datasetId: string;
  imageUrl: string;
  lat: number;
  lng: number;
  label: string;
}

export interface DemoAgentProfile {
  agentId: DemoAgentId;
  displayName: string;
  tagline: string;
  specialization: string;
  avatarEmoji: string;
  color: string;
}
