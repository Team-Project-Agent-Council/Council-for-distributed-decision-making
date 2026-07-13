"""State schema for the VLM Council (Tournament Only architecture).

- ``rag_findings``, deterministic findings from the tournament loop
  (kind="tournament_match").
- ``rag_refs_seen``, set of (country, image_path) tuples already shown, so refs
  aren't repeated across rounds.
- ``tournament_log``, list of bracket matches with winner/reasoning, so the
  final country_result can include "Tournament:" provenance.
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
    """A deterministic finding produced by the tournament loop.

    kind:
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

    # Legacy region-schema fields (kept for back-compat with stored result.json)
    current_phase: str                    # "initial" | "country" | "tournament"
    region_consensus: bool                # True when all agents agree on region
    confirmed_region: str                 # Set region after consensus/decision
    proposed_regions: list[str]           # Regions identified by judge
    region_candidates: dict               # {region: {country: agent_count}} mapping from consensus check

    # Hypotheses & Evaluations
    active_hypotheses: list[Hypothesis]
    hypothesis_evaluations: Annotated[list[HypothesisEvaluation], operator.add]

    # Country-Level Assessment (legacy, kept for back-compat with stored result.json)
    linguistic_country_assessment: AgentAssessment
    landscape_country_assessment: AgentAssessment
    botanics_country_assessment: AgentAssessment
    regulatory_country_assessment: AgentAssessment
    meta_country_assessment: AgentAssessment

    # RAG references + Tournament bracket
    candidate_pool: list[str]             # top-K countries that enter the tournament
    rag_findings: Annotated[list[RAGFinding], operator.add]
    rag_refs_seen: list[list[str]]        # [[country, ref_path], ...]
    road_filter_warnings: list[str]       # human-readable warnings for the tournament judge
    tournament_log: Annotated[list[TournamentMatch], operator.add]

    # Top-2 region selection from decide_region
    runner_up_region: str | None

    # Region survivors after the region-level road filter (top-N pruned)
    surviving_regions: list[str]

    # Results
    region_decision_reasoning: str
    country_result: str          # Final Output
    coordinates: Optional[dict[str, float]]  # {"lat", "lng"} or None
    final_reasoning: str
    error: Optional[str]                     # Set on hard failure         # Final Output
