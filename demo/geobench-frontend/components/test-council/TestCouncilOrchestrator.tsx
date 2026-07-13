"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Sparkles, MapPin, Shuffle, RotateCcw } from "lucide-react";
import { ImageUploadZone, ImagePreview } from "./ImageUploadZone";
import { DemoAgentPanel } from "./DemoAgentPanel";
import { DemoMap } from "./DemoMap";
import { ProgressTimeline, type DemoViewStep } from "./ProgressTimeline";
import { HypothesisMatrix } from "./HypothesisMatrix";
import {
  DEMO_AGENT_PROFILES,
  DEMO_AGENT_IDS,
} from "@/services/api/demoAgents";
import type {
  DemoAgentAssessment,
  DemoAgentId,
  DemoHypothesisEvaluation,
} from "@/services/api/types";
import {
  useDemoStore,
  type DemoPhase,
  isRegionConsensusPath,
} from "@/stores/demoStore";
import { demoService } from "@/services/api/activeDemoService";
import { BASE_PATH, STATIC_DEMO_MODE } from "@/lib/constants";

export function TestCouncilOrchestrator() {
  const phase = useDemoStore((s) => s.phase);
  const imageUrl = useDemoStore((s) => s.imageUrl);
  const groundTruth = useDemoStore((s) => s.groundTruth);
  const agentAssessments = useDemoStore((s) => s.agentAssessments);
  const agentReady = useDemoStore((s) => s.agentReady);
  const regionConsensus = useDemoStore((s) => s.regionConsensus);
  const regionHypotheses = useDemoStore((s) => s.regionHypotheses);
  const regionEvaluations = useDemoStore((s) => s.regionEvaluations);
  const regionEvaluationSummary = useDemoStore((s) => s.regionEvaluationSummary);
  const regionDecision = useDemoStore((s) => s.regionDecision);
  const countryAssessments = useDemoStore((s) => s.countryAssessments);
  const countryHypotheses = useDemoStore((s) => s.countryHypotheses);
  const countryEvaluations = useDemoStore((s) => s.countryEvaluations);
  const countryEvaluationSummary = useDemoStore(
    (s) => s.countryEvaluationSummary,
  );
  const finalResult = useDemoStore((s) => s.finalResult);
  const error = useDemoStore((s) => s.error);
  const startRun = useDemoStore((s) => s.startRun);
  const reset = useDemoStore((s) => s.reset);

  // Local state for the staging step (image picked but not yet sent).
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingPreviewUrl, setPendingPreviewUrl] = useState<string | null>(null);
  const [pendingDatasetId, setPendingDatasetId] = useState<string | null>(null);

  // Which agents the user wants to see on the map. Defaults to all-on.
  const [selectedAgents, setSelectedAgents] = useState<Set<DemoAgentId>>(
    () => new Set(DEMO_AGENT_IDS),
  );

  function toggleAgent(id: DemoAgentId) {
    setSelectedAgents((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectAllAgents() {
    setSelectedAgents(new Set(DEMO_AGENT_IDS));
  }

  // Step scrubbing: while the run is live, the viewed step auto-tracks the
  // pipeline phase. Once the user clicks a step, we lock to that manual value
  // until they reset / start a new run.
  const [manualStep, setManualStep] = useState<DemoViewStep | null>(null);

  const liveStep: DemoViewStep = useMemo(
    () => phaseToViewStep(phase),
    [phase],
  );
  const viewedStep: DemoViewStep = manualStep ?? liveStep;

  function handleStepClick(step: DemoViewStep) {
    setManualStep(step);
  }

  const isRunning =
    phase === "uploading" ||
    phase === "phase1" ||
    phase === "region_consensus" ||
    phase === "region_hypotheses" ||
    phase === "region_evaluation" ||
    phase === "region_decision" ||
    phase === "country_assess" ||
    phase === "country_hypotheses" ||
    phase === "country_evaluation" ||
    phase === "final";

  const isDone = phase === "done" || finalResult !== null;
  const canStart =
    !isRunning &&
    (pendingFile !== null || pendingDatasetId !== null || STATIC_DEMO_MODE);

  // In static-replay mode there is only one image and the user cannot
  // upload / pick a different one. Pre-populate the staging slot on first
  // mount so the demo image is visible immediately (no "Random" click
  // required) and the primary button reads as "Analyse with Council".
  useEffect(() => {
    if (!STATIC_DEMO_MODE) return;
    if (pendingDatasetId || pendingFile || imageUrl) return;
    setPendingDatasetId("static-demo");
    setPendingPreviewUrl(`${BASE_PATH}/demo-fixture/image.png`);
    // Only meant to run on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleImage(file: File, url: string) {
    if (isRunning) return;
    reset();
    setPendingFile(file);
    setPendingPreviewUrl(url);
    setPendingDatasetId(null);
    setManualStep(null);
  }

  function handleResetStaging() {
    reset();
    setPendingFile(null);
    setPendingPreviewUrl(null);
    setPendingDatasetId(null);
    setManualStep(null);
  }

  async function handleRandom() {
    if (isRunning) return;
    reset();
    setManualStep(null);
    try {
      const loc = await demoService.getRandomLocation();
      setPendingFile(null);
      setPendingDatasetId(loc.datasetId);
      setPendingPreviewUrl(loc.imageUrl);
    } catch (err) {
      // Surface as inline error so failures aren't silent.
      useDemoStore.setState({
        phase: "error",
        error:
          err instanceof Error
            ? `Random location failed: ${err.message}`
            : "Random location failed",
      });
    }
  }

  function handleStart() {
    if (!canStart) return;
    setManualStep(null);
    if (pendingFile) {
      void startRun({ file: pendingFile });
    } else if (pendingDatasetId) {
      void startRun({ datasetId: pendingDatasetId });
    } else if (STATIC_DEMO_MODE) {
      // Belt-and-braces: in static mode the datasetId is always the same
      // fixture. If for any reason the staging slot got cleared, fall
      // back to it explicitly so "Run Again" always works.
      void startRun({ datasetId: "static-demo" });
    }
  }

  // ---------------------------------------------------------------------
  // Map vote computation
  // ---------------------------------------------------------------------
  //
  // The map shows per-country agent votes. The vote map is rebuilt depending
  // on which view-step the user is currently looking at:
  //
  //  - phase1 / region_consensus
  //      → seed votes from Phase-1 candidates; no narrowing applied.
  //
  //  - region_eval
  //      → same Phase-1 seed; cross out the AGENT × REGION pair when an
  //        agent contradicted that region. (Country fills don't change yet,
  //        because evaluations are at region-level — the visual cue is the
  //        agent dot row, not the polygon.)
  //
  //  - country_assess (Path B only)
  //      → re-seed votes from countryAssessments instead of Phase-1.
  //        (Phase-1 candidates outside the chosen region disappear.)
  //
  //  - country_eval / final
  //      → seed from the appropriate assessment source (Path A: phase-1,
  //        Path B: country_assessments) and remove agents whose
  //        country-level evaluation contradicts the country.
  //
  // Selected agents are filtered out at the very end before the votes
  // reach the map.

  const isPathA = isRegionConsensusPath(regionConsensus);

  const { agentVotesByCountry, countryCentroids } = useMemo(() => {
    const votes: Record<string, Set<DemoAgentId>> = {};
    const centroids: Record<string, { lat: number; lng: number }> = {};

    function seedFrom(
      assessments: Partial<Record<DemoAgentId, DemoAgentAssessment>>,
    ) {
      for (const id of DEMO_AGENT_IDS) {
        if (!selectedAgents.has(id)) continue;
        const a = assessments[id];
        if (!a) continue;
        for (const cand of a.candidates) {
          if (!votes[cand.country]) votes[cand.country] = new Set();
          votes[cand.country].add(id);
          if (!centroids[cand.country]) {
            centroids[cand.country] = { lat: cand.lat, lng: cand.lng };
          }
        }
      }
    }

    function applyCountryContradictions(
      evals: Record<string, DemoHypothesisEvaluation>,
    ) {
      for (const ev of Object.values(evals)) {
        if (!selectedAgents.has(ev.agentName)) continue;
        if (ev.supports) continue;
        // hypothesisValue is the country name in human form; we already keyed
        // votes by country name during seeding.
        const country = ev.hypothesisValue;
        if (votes[country]) votes[country].delete(ev.agentName);
      }
    }

    if (
      viewedStep === "phase1" ||
      viewedStep === "region_consensus" ||
      viewedStep === "region_eval"
    ) {
      // Seed from Phase-1 candidates. Region-level contradictions live in
      // a separate row of the dot cluster (see HypothesisMatrix), not on
      // the country fills, so we don't subtract here.
      seedFrom(agentAssessments);
    } else if (viewedStep === "country_assess") {
      // Path B: replace the Phase-1 seed with the constrained assessments.
      // Path A: country_assess is skipped — fall back to Phase-1.
      if (isPathA === false) {
        seedFrom(countryAssessments);
      } else {
        seedFrom(agentAssessments);
      }
    } else {
      // country_eval / final
      if (isPathA === false) {
        seedFrom(countryAssessments);
      } else {
        seedFrom(agentAssessments);
      }
      applyCountryContradictions(countryEvaluations);
    }

    // Drop empty entries so the map doesn't render dead countries.
    const out: Record<string, DemoAgentId[]> = {};
    for (const [country, set] of Object.entries(votes)) {
      if (set.size === 0) continue;
      // Keep stable agent ordering (driven by DEMO_AGENT_IDS).
      out[country] = DEMO_AGENT_IDS.filter((id) => set.has(id));
    }
    return { agentVotesByCountry: out, countryCentroids: centroids };
  }, [
    viewedStep,
    selectedAgents,
    agentAssessments,
    countryAssessments,
    countryEvaluations,
    isPathA,
  ]);

  // Which assessments to surface in the DemoAgentPanel on the left rail.
  // Mirrors the map's vote-source logic so the left/right panes stay in
  // sync as the user scrubs across steps:
  //  - phase1 / region_consensus / region_eval → always Phase-1 assessments
  //  - country_assess / country_eval / final on Path B → constrained
  //    (region-narrowed) assessments per agent
  //  - country_* / final on Path A → Phase-1 (there's no re-assessment)
  //  - fall back to Phase-1 if constrained assessments haven't arrived yet
  //    for a given agent (SSE events trickle in one agent at a time)
  const viewedAssessments = useMemo(() => {
    const useCountry =
      isPathA === false &&
      (viewedStep === "country_assess" ||
        viewedStep === "country_eval" ||
        viewedStep === "final");
    if (!useCountry) return agentAssessments;
    const merged: Partial<Record<DemoAgentId, DemoAgentAssessment>> = {};
    for (const id of DEMO_AGENT_IDS) {
      merged[id] = countryAssessments[id] ?? agentAssessments[id];
    }
    return merged;
  }, [viewedStep, isPathA, agentAssessments, countryAssessments]);

  // Map only knows about a coarse phase model. Map the view step into it.
  const mapPhase = viewStepToMapPhase(viewedStep);
  // Hide the consensus pin until the user has scrubbed to the final step.
  const consensusForMap =
    mapPhase === "final" || mapPhase === "done" ? finalResult : null;

  // Judge reasoning to surface in the side rail. We collapse multiple
  // sources (consensus reasoning, region decision, final reasoning) into a
  // single block keyed by the current viewed step so the user always sees
  // text relevant to the step they're inspecting.
  //
  // Design principle: reasoning stays "sticky" as long as it's still
  // informative for the current step. E.g. the region_decision reasoning
  // remains relevant during country_assess / country_eval because the user
  // is looking at how the picked region narrows the country search —
  // hiding it right when country_assess starts loses context.
  const judgeReasoningPanel = useMemo(() => {
    if (
      viewedStep === "region_consensus" &&
      regionConsensus &&
      regionConsensus.consensus &&
      regionConsensus.confirmedRegion
    ) {
      return {
        title: "Region consensus",
        body: `All specialists agreed on ${regionConsensus.confirmedRegion}. Region-level hypothesis testing skipped.`,
      };
    }
    if (
      viewedStep === "region_consensus" &&
      regionConsensus &&
      !regionConsensus.consensus
    ) {
      return {
        title: "Region split",
        body: `No consensus among the specialists. Proposed regions: ${regionConsensus.proposedRegions.join(", ") || "—"}.`,
      };
    }
    // Region-decision reasoning is relevant from region_eval all the way
    // through country_assess / country_eval — the user is watching how the
    // judge's region pick narrows the country search. We only hand off to
    // finalResult's reasoning once the country decision itself is in.
    if (
      (viewedStep === "region_eval" ||
        viewedStep === "country_assess" ||
        (viewedStep === "country_eval" && !finalResult)) &&
      regionDecision
    ) {
      return {
        title: `Region: ${regionDecision.confirmedRegion}`,
        body: regionDecision.reasoning || "(no reasoning provided)",
      };
    }
    if ((viewedStep === "final" || viewedStep === "country_eval") && finalResult) {
      return {
        title: `Country: ${finalResult.country}`,
        body: finalResult.reasoning || "(no reasoning provided)",
      };
    }
    return null;
  }, [viewedStep, regionConsensus, regionDecision, finalResult]);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "360px 1fr",
        gap: 20,
        height: "calc(100vh - 56px)",
        padding: 20,
        maxWidth: 1500,
        margin: "0 auto",
        boxSizing: "border-box",
      }}
    >
      {/* Left rail — controls + agents */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 14,
          overflowY: "auto",
          paddingRight: 4,
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <Sparkles size={18} style={{ color: "#c6ef38" }} />
            <h1 style={{ fontWeight: 800, fontSize: 20, margin: 0 }}>
              Progressive Narrowing Demo
            </h1>
          </div>
          <p style={{ fontSize: 12, color: "var(--muted-foreground)", margin: 0 }}>
            5 specialist VLM agents → judge picks region → re-assess inside region → judge picks country.
          </p>
        </div>

        {pendingPreviewUrl || imageUrl ? (
          <ImagePreview
            url={pendingPreviewUrl ?? imageUrl ?? ""}
            onReset={STATIC_DEMO_MODE ? undefined : handleResetStaging}
          />
        ) : (
          !STATIC_DEMO_MODE && <ImageUploadZone onImage={handleImage} />
        )}

        <div style={{ display: "flex", gap: 8 }}>
          {!STATIC_DEMO_MODE && (
            <button
              onClick={handleRandom}
              disabled={isRunning}
              style={btnSecondaryStyle(isRunning)}
            >
              <Shuffle size={14} />
              Random
            </button>
          )}
          <button
            onClick={handleStart}
            disabled={!canStart}
            style={btnPrimaryStyle(!canStart)}
          >
            <Sparkles size={14} />
            {isRunning
              ? "Analysing…"
              : isDone
              ? "Run Again"
              : "Analyse with Council"}
          </button>
        </div>

        {error && (
          <div
            style={{
              padding: "10px 12px",
              borderRadius: 10,
              background: "rgba(239,68,68,0.12)",
              border: "1px solid rgba(239,68,68,0.4)",
              fontSize: 12,
              color: "#f87171",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <span>{error}</span>
            <button
              onClick={handleResetStaging}
              style={{
                marginLeft: "auto",
                background: "transparent",
                border: "1px solid rgba(239,68,68,0.4)",
                color: "#f87171",
                fontSize: 11,
                padding: "2px 8px",
                borderRadius: 6,
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: 4,
              }}
            >
              <RotateCcw size={11} /> Reset
            </button>
          </div>
        )}

        <div>
          <div
            style={{
              ...sectionLabelStyle,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 8,
            }}
          >
            <span>
              Council Agents{" "}
              <span
                style={{
                  fontWeight: 500,
                  color: "var(--muted-foreground)",
                  letterSpacing: 0,
                  textTransform: "none",
                }}
              >
                · {selectedAgents.size}/{DEMO_AGENT_IDS.length} on map
              </span>
            </span>
            <button
              type="button"
              onClick={selectAllAgents}
              disabled={selectedAgents.size === DEMO_AGENT_IDS.length}
              style={{
                background: "transparent",
                border: "1px solid var(--border)",
                color:
                  selectedAgents.size === DEMO_AGENT_IDS.length
                    ? "var(--muted-foreground)"
                    : "var(--foreground)",
                fontSize: 10,
                fontWeight: 600,
                letterSpacing: "0.04em",
                textTransform: "uppercase",
                padding: "3px 8px",
                borderRadius: 999,
                cursor:
                  selectedAgents.size === DEMO_AGENT_IDS.length
                    ? "not-allowed"
                    : "pointer",
              }}
            >
              Show all
            </button>
          </div>
          <DemoAgentPanel
            agents={DEMO_AGENT_PROFILES}
            assessments={viewedAssessments}
            ready={agentReady}
            active={isRunning || isDone}
            selected={selectedAgents}
            onToggle={toggleAgent}
          />
        </div>

        <AnimatePresence>
          {judgeReasoningPanel && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              style={{
                padding: "12px 14px",
                borderRadius: 12,
                background: "rgba(198,239,56,0.06)",
                border: "1px solid rgba(198,239,56,0.25)",
              }}
            >
              <div style={sectionLabelStyle}>{judgeReasoningPanel.title}</div>
              <div
                style={{
                  fontSize: 12,
                  color: "var(--foreground)",
                  lineHeight: 1.5,
                  marginTop: 4,
                }}
              >
                {judgeReasoningPanel.body}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {finalResult && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              style={{
                padding: "14px 16px",
                borderRadius: 14,
                background: "rgba(198,239,56,0.1)",
                border: "1px solid rgba(198,239,56,0.45)",
              }}
            >
              <div
                style={{
                  fontSize: 11,
                  color: "#c6ef38",
                  fontWeight: 700,
                  letterSpacing: "0.08em",
                  marginBottom: 6,
                }}
              >
                COUNCIL CONSENSUS
              </div>
              <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                <MapPin size={16} style={{ color: "#c6ef38", flexShrink: 0, marginTop: 2 }} />
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 700, fontSize: 15 }}>{finalResult.country}</div>
                  <div style={{ fontSize: 11, color: "var(--muted-foreground)" }}>
                    {finalResult.lat.toFixed(3)}, {finalResult.lng.toFixed(3)}
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "var(--foreground)",
                      marginTop: 8,
                      lineHeight: 1.5,
                    }}
                  >
                    {finalResult.reasoning}
                  </div>
                  {groundTruth && (
                    <div
                      style={{
                        marginTop: 10,
                        paddingTop: 10,
                        borderTop: "1px solid rgba(198,239,56,0.25)",
                        fontSize: 11,
                        color: "var(--muted-foreground)",
                      }}
                    >
                      Actual:{" "}
                      <span style={{ color: "#4ade80", fontWeight: 600 }}>
                        {groundTruth.label}
                      </span>{" "}
                      · {haversineKm(finalResult, groundTruth).toFixed(0)} km off ·{" "}
                      <span style={{ color: "#c6ef38", fontWeight: 700 }}>
                        {Math.round(
                          5000 * Math.exp(-haversineKm(finalResult, groundTruth) / 2000),
                        )}{" "}
                        pts
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Right pane — timeline + map + matrix */}
      <div
        style={{
          display: "grid",
          gridTemplateRows: "auto 1fr auto",
          gap: 12,
          minHeight: 0,
        }}
      >
        <ProgressTimeline
          phase={phase}
          pathAConsensus={isPathA}
          viewStep={viewedStep}
          scrubbable={
            // Scrubbing is available as soon as there's *any* seedable data
            // for a previous step. Once at least one Phase-1 assessment has
            // arrived, the user can flip back to review earlier steps
            // without waiting for the whole run to finish. This lets them
            // re-read region-consensus reasoning while country_assess is
            // still streaming, etc.
            Object.keys(agentAssessments).length > 0 ||
            phase === "region_consensus" ||
            phase === "region_hypotheses" ||
            phase === "region_evaluation" ||
            phase === "region_decision" ||
            phase === "country_assess" ||
            phase === "country_hypotheses" ||
            phase === "country_evaluation" ||
            phase === "final" ||
            isDone
          }
          onStepClick={handleStepClick}
        />

        <div style={{ minHeight: 0, position: "relative" }}>
          <DemoMap
            agentVotesByCountry={agentVotesByCountry}
            countryCentroids={countryCentroids}
            consensusPin={consensusForMap}
            groundTruth={groundTruth}
            phase={mapPhase}
          />
        </div>

        <AnimatePresence>
          {(viewedStep === "region_eval" && regionHypotheses.length > 0) ||
          (viewedStep === "country_eval" && countryHypotheses.length > 0) ||
          (viewedStep === "final" && countryHypotheses.length > 0) ? (
            <motion.div
              key={viewedStep}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              style={{ maxHeight: 240, overflowY: "auto" }}
            >
              {viewedStep === "region_eval" ? (
                <HypothesisMatrix
                  title="Region"
                  hypotheses={regionHypotheses}
                  evaluations={regionEvaluations}
                  summary={regionEvaluationSummary}
                />
              ) : (
                <HypothesisMatrix
                  title="Country"
                  hypotheses={countryHypotheses}
                  evaluations={countryEvaluations}
                  summary={countryEvaluationSummary}
                />
              )}
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </div>
  );
}

const sectionLabelStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 700,
  color: "var(--muted-foreground)",
  letterSpacing: "0.08em",
  marginBottom: 8,
  textTransform: "uppercase",
};

function btnPrimaryStyle(disabled: boolean): React.CSSProperties {
  return {
    flex: 1,
    padding: "12px",
    borderRadius: 12,
    border: "none",
    background: disabled ? "var(--border)" : "#c6ef38",
    color: disabled ? "var(--muted-foreground)" : "#111",
    fontWeight: 700,
    fontSize: 14,
    cursor: disabled ? "not-allowed" : "pointer",
    transition: "background 0.2s",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
  };
}

function btnSecondaryStyle(disabled: boolean): React.CSSProperties {
  return {
    padding: "12px 14px",
    borderRadius: 12,
    border: "1px solid var(--border)",
    background: "transparent",
    color: disabled ? "var(--muted-foreground)" : "var(--foreground)",
    fontWeight: 600,
    fontSize: 13,
    cursor: disabled ? "not-allowed" : "pointer",
    transition: "background 0.2s",
    display: "flex",
    alignItems: "center",
    gap: 6,
  };
}

function haversineKm(
  a: { lat: number; lng: number },
  b: { lat: number; lng: number },
): number {
  const R = 6371;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.sin(dLng / 2) ** 2 * Math.cos(lat1) * Math.cos(lat2);
  return 2 * R * Math.asin(Math.sqrt(h));
}

/**
 * Live pipeline phase → default view step. The user-facing stepper has 6
 * steps, but the live pipeline phase is more granular. Multiple substates
 * collapse onto the same view step.
 */
function phaseToViewStep(phase: DemoPhase): DemoViewStep {
  switch (phase) {
    case "idle":
    case "uploading":
    case "phase1":
      return "phase1";
    case "region_consensus":
      return "region_consensus";
    case "region_hypotheses":
    case "region_evaluation":
      return "region_eval";
    case "region_decision":
    case "country_assess":
      return "country_assess";
    case "country_hypotheses":
    case "country_evaluation":
      return "country_eval";
    case "final":
    case "done":
      return "final";
    case "error":
      return "phase1";
    default:
      return "phase1";
  }
}

/**
 * View step → coarse map mode. The DemoMap only differentiates "live narrowing"
 * vs "final reveal" — region/country distinction is invisible at the map level.
 */
function viewStepToMapPhase(
  step: DemoViewStep,
): "idle" | "phase1" | "judging" | "hypotheses" | "final" | "done" {
  switch (step) {
    case "phase1":
      return "phase1";
    case "region_consensus":
      return "judging";
    case "region_eval":
    case "country_assess":
    case "country_eval":
      return "hypotheses";
    case "final":
      return "final";
  }
}
