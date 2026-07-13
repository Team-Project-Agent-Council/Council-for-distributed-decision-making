import { create } from "zustand";
import type {
  DemoAgentAssessment,
  DemoAgentId,
  DemoEvaluationSummaryEntry,
  DemoFinalResult,
  DemoGroundTruth,
  DemoHypothesis,
  DemoHypothesisEvaluation,
  DemoRegionConsensusResult,
  DemoRegionDecision,
  DemoSseEvent,
  DemoStartRunInput,
} from "@/services/api/types";
import { demoService } from "@/services/api/activeDemoService";
import { DEMO_AGENT_IDS } from "@/services/api/demoAgents";

/**
 * Coarse pipeline phase. Maps roughly to the LangGraph node currently
 * running. The UI uses this for stepper / map mode dispatch — finer-grained
 * progress state lives in the per-section substates below.
 *
 * Path A (region consensus) skips the region_* phases and jumps from
 * `region_consensus` straight to `country_hypotheses`.
 */
export type DemoPhase =
  | "idle"
  | "uploading"
  | "phase1"
  | "region_consensus"
  | "region_hypotheses"
  | "region_evaluation"
  | "region_decision"
  | "country_assess"
  | "country_hypotheses"
  | "country_evaluation"
  | "final"
  | "done"
  | "error";

/** All evaluations for one phase, keyed by `${hypothesisId}|${agentId}`. */
export type EvaluationsMap = Record<string, DemoHypothesisEvaluation>;

interface DemoState {
  runId: string | null;
  imageUrl: string | null;
  groundTruth: DemoGroundTruth | null;

  phase: DemoPhase;

  // Phase 1 — initial assessments.
  agentAssessments: Partial<Record<DemoAgentId, DemoAgentAssessment>>;
  agentReady: Record<DemoAgentId, boolean>;

  // Region phase (Path B only).
  regionConsensus: DemoRegionConsensusResult | null;
  regionHypotheses: DemoHypothesis[];
  regionEvaluations: EvaluationsMap;
  regionEvaluationSummary: DemoEvaluationSummaryEntry[] | null;
  regionDecision: DemoRegionDecision | null;

  // Country-constrained re-assessments (Path B only).
  countryAssessments: Partial<Record<DemoAgentId, DemoAgentAssessment>>;

  // Country phase (always runs).
  countryHypotheses: DemoHypothesis[];
  countryEvaluations: EvaluationsMap;
  countryEvaluationSummary: DemoEvaluationSummaryEntry[] | null;

  // Final.
  finalResult: DemoFinalResult | null;

  error: string | null;
  unsubscribe: (() => void) | null;

  startRun: (input: DemoStartRunInput) => Promise<void>;
  reset: () => void;
}

const emptyAgentReady: Record<DemoAgentId, boolean> = {
  linguistic: false,
  landscape: false,
  botanics: false,
  regulatory: false,
  meta: false,
};

const initialState: Omit<DemoState, "startRun" | "reset"> = {
  runId: null,
  imageUrl: null,
  groundTruth: null,
  phase: "idle",
  agentAssessments: {},
  agentReady: { ...emptyAgentReady },
  regionConsensus: null,
  regionHypotheses: [],
  regionEvaluations: {},
  regionEvaluationSummary: null,
  regionDecision: null,
  countryAssessments: {},
  countryHypotheses: [],
  countryEvaluations: {},
  countryEvaluationSummary: null,
  finalResult: null,
  error: null,
  unsubscribe: null,
};

export const useDemoStore = create<DemoState>((set, get) => ({
  ...initialState,

  startRun: async (input) => {
    // Cancel any prior subscription.
    const prev = get().unsubscribe;
    if (prev) prev();

    set({
      ...initialState,
      phase: "uploading",
      agentReady: { ...emptyAgentReady },
    });

    let runResp;
    try {
      runResp = await demoService.startRun(input);
    } catch (err) {
      set({
        phase: "error",
        error: err instanceof Error ? err.message : String(err),
      });
      return;
    }

    set({
      runId: runResp.runId,
      imageUrl: runResp.imageUrl,
      groundTruth: runResp.groundTruth ?? null,
      phase: "phase1",
    });

    const unsub = demoService.subscribeToRun(runResp.runId, {
      onEvent: (e) => handleEvent(set, get, e),
      onError: (err) => {
        set({ phase: "error", error: err.message });
      },
      onClose: () => {
        // Final state already set by `done`/`final_result` handlers.
      },
    });

    set({ unsubscribe: unsub });
  },

  reset: () => {
    const unsub = get().unsubscribe;
    if (unsub) unsub();
    set({ ...initialState });
  },
}));

function handleEvent(
  set: (
    partial: Partial<DemoState> | ((s: DemoState) => Partial<DemoState>)
  ) => void,
  get: () => DemoState,
  e: DemoSseEvent
) {
  switch (e.type) {
    case "run_started":
      set({
        runId: e.data.runId,
        imageUrl: e.data.imageUrl,
        groundTruth: e.data.groundTruth ?? null,
      });
      return;

    case "phase1_started":
      set({ phase: "phase1" });
      return;

    case "agent_assessment": {
      const a = e.data;
      set((s) => ({
        agentAssessments: { ...s.agentAssessments, [a.agentName]: a },
        agentReady: { ...s.agentReady, [a.agentName]: true },
      }));
      return;
    }

    case "region_consensus_result":
      set({
        phase: "region_consensus",
        regionConsensus: e.data,
      });
      return;

    case "region_hypotheses_generated":
      set({
        phase: "region_hypotheses",
        regionHypotheses: e.data.hypotheses,
      });
      return;

    case "region_evaluation": {
      const ev = e.data;
      const key = `${ev.hypothesisId}|${ev.agentName}`;
      set((s) => ({
        phase: s.phase === "region_evaluation" ? s.phase : "region_evaluation",
        regionEvaluations: { ...s.regionEvaluations, [key]: ev },
      }));
      return;
    }

    case "region_evaluation_complete":
      set({ regionEvaluationSummary: e.data.summary });
      return;

    case "region_decision":
      set({
        phase: "region_decision",
        regionDecision: e.data,
      });
      return;

    case "country_assessment": {
      const a = e.data;
      set((s) => ({
        phase: s.phase === "country_assess" ? s.phase : "country_assess",
        countryAssessments: { ...s.countryAssessments, [a.agentName]: a },
      }));
      return;
    }

    case "country_hypotheses_generated":
      set({
        phase: "country_hypotheses",
        countryHypotheses: e.data.hypotheses,
      });
      return;

    case "country_evaluation": {
      const ev = e.data;
      const key = `${ev.hypothesisId}|${ev.agentName}`;
      set((s) => ({
        phase: s.phase === "country_evaluation" ? s.phase : "country_evaluation",
        countryEvaluations: { ...s.countryEvaluations, [key]: ev },
      }));
      return;
    }

    case "country_evaluation_complete":
      set({ countryEvaluationSummary: e.data.summary });
      return;

    case "final_started":
      set({ phase: "final" });
      return;

    case "final_result":
      set({ finalResult: e.data, phase: "final" });
      return;

    case "error":
      set({ phase: "error", error: e.data.message });
      return;

    case "done":
      set({ phase: "done" });
      return;
  }
}

/**
 * Did Path A (region consensus) get taken? Derives from regionConsensus state.
 *
 * - null            → not yet known
 * - true            → Path A; region phase events were skipped
 * - false           → Path B; full region phase ran
 */
export function isRegionConsensusPath(
  consensus: DemoRegionConsensusResult | null
): boolean | null {
  if (consensus === null) return null;
  return consensus.consensus;
}

/** Convenience: are all phase-1 agents finished? */
export function allPhase1AgentsReady(
  ready: Record<DemoAgentId, boolean>
): boolean {
  return DEMO_AGENT_IDS.every((id) => ready[id]);
}
