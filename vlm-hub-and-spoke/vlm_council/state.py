"""State schema for the VLM Council."""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


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


class DiscussionEntry(TypedDict):
    """One round of hub-and-spoke discussion."""

    round_number: int
    judge_question: str
    target_agent: str
    agent_response: str


class VLMCouncilState(TypedDict):
    """State that flows through the VLM Council graph."""

    # Input
    image_path: str
    image_b64: str
    image_mime: str

    # Phase 1: VLM assessments
    linguistic_assessment: AgentAssessment
    landscape_assessment: AgentAssessment
    botanics_assessment: AgentAssessment
    regulatory_assessment: AgentAssessment
    meta_assessment: AgentAssessment

    # Phase 2: Hub-and-spoke discussion
    discussion_log: Annotated[list[DiscussionEntry], operator.add]
    discussion_round: int
    judge_messages: Annotated[list[BaseMessage], add_messages]

    # Final output
    country_result: str                              # Raw judge free-text
    coordinates: Optional[dict[str, float]]          # {"lat": float, "lng": float} | None
    final_reasoning: str
    error: Optional[str]                             # Set if the judge failed hard
