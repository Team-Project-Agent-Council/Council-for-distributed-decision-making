"""API response models for the Progressive Narrowing demo + Council info.

Most of the old Game/Stats/User models were removed when the /game flow was
dropped. The only remaining endpoint that uses Pydantic response models is
the council info one (`routers/council.py`); the demo router builds JSON
responses inline because the SSE event payloads are dynamic.
"""

from typing import Optional

from pydantic import BaseModel


class AgentProfile(BaseModel):
    agentId: str
    displayName: str
    tagline: str
    description: str
    avatarEmoji: str
    tools: list[str]
    specialization: str
    color: str
    exampleAnalysis: Optional[str] = None


class CollaborationStep(BaseModel):
    stepNumber: int
    title: str
    description: str
    agentsInvolved: list[str]


class CouncilInfo(BaseModel):
    agents: list[AgentProfile]
    collaborationSteps: list[CollaborationStep]
