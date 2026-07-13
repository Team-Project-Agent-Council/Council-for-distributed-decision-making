"""State schema for the VLM Council — Progressive Narrowing architecture."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


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


class VLMCouncilState(TypedDict):
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
    current_phase: str                    # "initial" | "region" | "country"
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

    # Results
    region_decision_reasoning: str
    country_result: str          # Final Output
    coordinates: str             # Final Output
    final_reasoning: str         # Final Output
