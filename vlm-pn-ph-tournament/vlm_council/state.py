"""State schema for the VLM Council, Progressive Narrowing architecture.

v12 extensions:
- ``rag_findings``, deterministic eliminations from prefilters (kind="elim_driving"
  or kind="elim_road_marking") and tournament context.
- ``rag_refs_seen``, set of (country, image_path) tuples already shown, so refs
  aren't repeated across rounds.
- ``road_filter_warnings``, surfaced when a prefilter would have eliminated all
  candidates and recovery kept them; the warning text is fed into the tournament
  judge as a hint.
- ``tournament_log``, list of bracket matches with winner/reasoning, so the
  final country_result can include "Tournament:" provenance like v10's
  "Road Check:" block.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict


# Confidence levels for hypothesis evaluation
ConfidenceLevel = Literal[
    "strongly_support", "support", "neutral", "contradicts", "strongly_contradicts"
]


class CandidateEntry(TypedDict):
    """A single country candidate with its own confidence and reasoning."""

    country: str
    confidence: str  # "high" | "medium" | "low" | "speculative"
    reasoning: str


class AgentAssessment(TypedDict):
    """Structured output from a specialist VLM agent."""

    agent_name: str
    candidates: list[CandidateEntry]
    evidence: list[str]


class Hypothesis(TypedDict):
    """A hypothesis to be evaluated by specialists."""

    hypothesis_id: str    # e.g. "region_europe" or "country_germany"
    level: str            # "region" | "country"
    value: str            # e.g. "Europe" or "Germany"
    statement: str        # "This image is from Europe"


class HypothesisEvaluation(TypedDict):
    """One agent's evaluation of one hypothesis."""

    agent_name: str
    hypothesis_id: str
    confidence: str       # ConfidenceLevel
    reasoning: str
    key_evidence: list[str]


class RAGFinding(TypedDict, total=False):
    """A deterministic finding produced by prefilters or by the tournament loop.

    kind:
      - "elim_driving"      → driving_side filter eliminated this country
      - "elim_road_marking" → road_marking filter eliminated this country
      - "recovery"          → all candidates would have been eliminated; reverted
      - "tournament_match"  → outcome of one bracket match
    """

    kind: str
    country: str
    detail: str
    # tournament_match-only:
    opponent: str
    winner: str


class TournamentMatch(TypedDict, total=False):
    """One pairwise comparison from the bracket."""

    round_label: str           # "semi-1" | "semi-2" | "final"
    country_a: str
    country_b: str
    pool_rank_a: int           # 0-indexed seed in candidate_pool (lower = higher seed)
    pool_rank_b: int
    winner: str
    reasoning: str
    agreement: str             # "agree" | "disagree" | "forward_only" | "reverse_only" | "both_empty" | "judge_error"


class RoadEvidence(TypedDict, total=False):
    """Structured observations from the dedicated road_evidence_extractor agent.

    Replaces the regex-on-evidence parsing the prefilters used to do.
    """

    outside_color: str             # white | yellow | red | blue | none | unclear
    inside_color: str              # same set
    driving_side: str              # LEFT | RIGHT | UNCLEAR
    driving_side_basis: str        # oncoming_traffic | front_car_lane |
                                   # asymmetric_marking | plate_or_sign | none


class VLMCouncilState(TypedDict, total=False):
    """State that flows through the VLM Council graph."""

    # Input
    image_path: str
    image_b64: str
    image_mime: str

    # Phase 1: Initial VLM assessments
    linguistic_assessment: AgentAssessment
    landscape_assessment: AgentAssessment
    botanics_assessment: AgentAssessment
    regulatory_assessment: AgentAssessment
    meta_assessment: AgentAssessment

    # Progressive Narrowing
    current_phase: str                    # "initial" | "region" | "country" | "prefilter" | "tournament"
    region_consensus: bool                # True when all agents agree on region
    confirmed_region: str                 # Set region after consensus/decision
    proposed_regions: list[str]           # Regions identified by judge
    region_candidates: dict               # {region: {country: agent_count}} mapping from consensus check

    # Hypotheses & Evaluations
    active_hypotheses: list[Hypothesis]
    hypothesis_evaluations: Annotated[list[HypothesisEvaluation], operator.add]

    # Country-Level Assessment (with Region-Constraint, only Path B)
    linguistic_country_assessment: AgentAssessment
    landscape_country_assessment: AgentAssessment
    botanics_country_assessment: AgentAssessment
    regulatory_country_assessment: AgentAssessment
    meta_country_assessment: AgentAssessment

    # v12: RAG prefilters + Tournament bracket
    candidate_pool: list[str]             # countries surviving prefilters → enter tournament
    rag_findings: Annotated[list[RAGFinding], operator.add]
    rag_refs_seen: list[list[str]]        # [[country, ref_path], ...]
    road_filter_warnings: list[str]       # human-readable warnings for the tournament judge
    tournament_log: Annotated[list[TournamentMatch], operator.add]

    # v12.1: dedicated road-evidence extractor (replaces regex-on-regulatory)
    road_evidence: RoadEvidence

    # Top-2 region selection from decide_region
    runner_up_region: str | None

    # Region survivors after the region-level road filter (top-N pruned)
    surviving_regions: list[str]

    # Results
    region_decision_reasoning: str
    country_result: str          # Final Output
    coordinates: Optional[dict[str, float]]  # {"lat", "lng"} or None
    error: Optional[str]                     # Set if the judge failed hard
    final_reasoning: str         # Final Output
