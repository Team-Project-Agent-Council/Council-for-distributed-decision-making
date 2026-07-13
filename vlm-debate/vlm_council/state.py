"""State schema for the VLM Council, Debate approach."""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


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


class DebateMessage(TypedDict):
    """A single message in a debate exchange."""

    agent_name: str
    position: str
    revised: bool
    confidence: str
    argument: str
    key_evidence: list[str]


class DebatePairing(TypedDict):
    """A single debate pairing between two agents in one debate round."""

    debate_round: int
    agent_a: str
    agent_b: str
    agent_a_initial_position: str
    agent_b_initial_position: str
    exchanges: list[DebateMessage]


class ModeratorDecision(TypedDict):
    """The moderator's decision after examining agent positions."""

    debate_round: int
    contradictions_found: list[dict]
    pairings_opened: list[dict]
    reasoning: str
    terminate: bool
    termination_reason: str


class VLMCouncilState(TypedDict):
    """State that flows through the VLM Council graph.

    Debate topology:
        Round 1: Each agent analyses the image independently
        Moderator: Identifies contradictions, pairs agents for debate
        Debate: Paired agents argue/revise in rounds (recursive loop)
        Judge: Receives Round 1 + all debate transcripts, makes final determination
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

    # Debate system
    debate_pairings: Annotated[list[DebatePairing], operator.add]
    moderator_decisions: Annotated[list[ModeratorDecision], operator.add]
    current_debate_round: int
    debate_terminated: bool

    # Final output
    country_result: str
    coordinates: Optional[dict[str, float]]  # {'lat', 'lng'} or None
    final_reasoning: str
    error: Optional[str]                     # Set if the judge failed hard
