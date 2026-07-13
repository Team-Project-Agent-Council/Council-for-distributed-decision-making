"""State schema for the VLM Council, Global Context Re-guess approach."""

from __future__ import annotations

from typing import Optional, TypedDict


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


class VLMCouncilState(TypedDict):
    """State that flows through the VLM Council graph.

    Global Context Re-guess topology:
        Round 1: Each agent analyses the image independently.
        Round 2: Each agent re-guesses with full context from all Round 1
                 assessments (every agent sees every other agent's Round 1).
        Judge:   Receives all Round 1 + Round 2 traces and produces the
                 final country + coordinates.
    """

    # Input
    image_path: str
    image_b64: str
    image_mime: str

    # Round 1: Initial independent assessments
    round_1_linguistic: AgentAssessment
    round_1_landscape: AgentAssessment
    round_1_botanics: AgentAssessment
    round_1_regulatory: AgentAssessment
    round_1_meta: AgentAssessment

    # Round 2: Re-guess with global context
    round_2_linguistic: AgentAssessment
    round_2_landscape: AgentAssessment
    round_2_botanics: AgentAssessment
    round_2_regulatory: AgentAssessment
    round_2_meta: AgentAssessment

    # Final output
    country_result: str                              # Raw judge free-text
    coordinates: Optional[dict[str, float]]          # {"lat": float, "lng": float} | None
    final_reasoning: str
    error: Optional[str]                             # Set if the judge failed hard
