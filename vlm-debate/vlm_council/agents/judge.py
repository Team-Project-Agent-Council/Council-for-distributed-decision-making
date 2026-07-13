"""Judge Agent: Moderator + Final Synthesizer for the Debate approach.

The judge has two roles:
1. Moderator: After Round 1 (and after each debate round), identifies contradictions
   between agents and decides which pairs should debate.
2. Final Synthesizer: Once debate terminates, synthesizes Round 1 + all debate
   transcripts into a final country determination.
"""

from __future__ import annotations

import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

from vlm_council.llm import get_vlm

_THINKING_ENABLED = os.environ.get("VLM_JUDGE_THINKING", "true").lower() in ("true", "1", "yes")
_THINK_PREFIX = "<|think|>\n" if _THINKING_ENABLED else ""


# === Moderator Role ===

MODERATOR_SYSTEM_PROMPT = """\
You are the Moderator of a GeoGuessr council debate. You have 5 specialist agents who independently analyzed a Street View image in Round 1. Your job is to identify disagreements and orchestrate targeted debates.

Your tasks:
1. IDENTIFY CONTRADICTIONS: Find agents whose top-1 country prediction DIFFERS from other agents. Only consider agents with confidence >= "{min_confidence}" as contradicting.
2. DECIDE PAIRINGS: Select which disagreeing agents should debate each other.
3. DECIDE TERMINATION: Should the debate end?

PAIRING PRIORITY (highest value → lowest):
1. Linguistic or Regulatory vs any disagreeing agent, these have HARD CONSTRAINTS (text, driving side, plates) that can definitively resolve disagreements.
2. Botanics vs any disagreeing agent, endemic species provide strong regional evidence.
3. Landscape vs Meta, AVOID this pairing unless no other option exists. Both rely on ambiguous visual features and debates between them rarely produce revisions or useful information.

TERMINATE the debate when ANY of these conditions is true:
- All agents with confidence >= "{min_confidence}" agree on the same top country (consensus)
- The evidence is genuinely ambiguous and further debate is unlikely to resolve it (stalemate)
- Only one agent disagrees and has "low" or "speculative" confidence (weak dissent)

DO NOT pair agents who already agree on the same top country.
DO NOT pair more than 3 pairs per round (focus on the strongest contradictions).
DO NOT pair agents that have "insufficient evidence" or no candidates.

Respond with JSON only:
{{"contradictions": [{{"agent_a": "<name>", "agent_b": "<name>", "country_a": "<country>", "country_b": "<country>"}}], "pairings": [{{"agent_a": "<name>", "agent_b": "<name>"}}], "reasoning": "<1-2 sentences>", "terminate": <true|false>, "termination_reason": "<reason if terminate=true, else empty string>"}}\
"""


async def moderate(state: dict, min_confidence: str = "medium", llm=None) -> str:
    """Examine current agent positions and decide debate pairings or termination."""
    if llm is None:
        llm = get_vlm("judge")

    positions_text = _format_current_positions(state)
    system = MODERATOR_SYSTEM_PROMPT.format(min_confidence=min_confidence)

    debate_history = ""
    debate_pairings = state.get("debate_pairings", [])
    if debate_pairings:
        debate_history = "\n\nPrevious debate exchanges:\n" + _format_debate_history(debate_pairings)

    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=(
            f"Current agent positions after Round 1:\n\n"
            f"{positions_text}"
            f"{debate_history}\n\n"
            "Analyze contradictions and decide pairings or termination."
        )),
    ])
    return response.content


def _format_current_positions(state: dict) -> str:
    """Format each agent's current top-1 position for the moderator."""
    agents = ["linguistic", "landscape", "botanics", "regulatory", "meta"]
    parts = []

    for name in agents:
        position = _get_agent_current_position(state, name)
        parts.append(f"  {name}: {position}")

    return "\n".join(parts)


def _get_agent_current_position(state: dict, agent_name: str) -> str:
    """Get an agent's current position, from latest debate message or Round 1."""
    debate_pairings = state.get("debate_pairings", [])
    for pairing in reversed(debate_pairings):
        for exchange in reversed(pairing.get("exchanges", [])):
            if exchange.get("agent_name") == agent_name:
                return f"{exchange['position']} ({exchange['confidence']})"

    assessment = state.get(f"round_1_{agent_name}", {})
    candidates = assessment.get("candidates", [])
    if candidates:
        top = candidates[0]
        return f"{top['country']} ({top['confidence']})"
    return "(no assessment)"


def _format_debate_history(debate_pairings: list) -> str:
    """Format debate history for context."""
    parts = []
    for pairing in debate_pairings:
        round_num = pairing.get("debate_round", "?")
        agent_a = pairing.get("agent_a", "?")
        agent_b = pairing.get("agent_b", "?")
        parts.append(f"  [Debate Round {round_num}: {agent_a} vs {agent_b}]")
        for ex in pairing.get("exchanges", []):
            revised_str = " (REVISED)" if ex.get("revised") else ""
            parts.append(
                f"    {ex['agent_name']}{revised_str}: "
                f"{ex.get('position', '?')} ({ex.get('confidence', '?')}), "
                f"{ex.get('argument', '')}"
            )
    return "\n".join(parts)


# === Final Synthesizer Role ===

FINAL_SYSTEM_PROMPT = """\
You are the Judge of a GeoGuessr council. You receive the COMPLETE analysis:

- Round 1: Each of 5 specialist agents independently analyzed the image
- Debate transcripts: Agents that disagreed were paired for adversarial debate where they had to defend or revise their positions

Your task is to synthesize ALL evidence to make the FINAL country determination.

DECISION RULES (follow in order):
1. MAJORITY SIGNAL: If 3+ agents agreed in Round 1 and the debate ended in stalemate (no concessions), the pre-debate majority is almost certainly correct. A stalemate does NOT mean both sides are equally valid, it means the minority could not produce evidence strong enough to convince the majority. Trust the majority unless a hard constraint eliminates their country.
2. EVIDENCE HIERARCHY (strongest → weakest):
   - Transcribed text, specific script, language identification (STRONGEST)
   - Driving side, license plate format, road sign conventions
   - Endemic plant species, region-specific crops
   - General terrain, climate, soil color (WEAK, shared across many countries)
   - Street furniture, camera rig, bollard style (WEAKEST, highly ambiguous)
3. CONCESSIONS are the strongest signal of all, an agent changed their mind because the evidence was compelling. Always follow the direction of concessions.
4. Only override the Round 1 majority if: (a) a hard constraint (driving side, script, plates) eliminates the majority country, OR (b) an agent conceded with specific, verifiable evidence.
5. Agents that were NOT involved in any debate still provide valid Round 1 evidence, do not ignore them.

Estimate coordinates based on your chosen country and any regional clues from the agents.

Respond with EXACTLY this format (no period after country name, no qualifiers, no parentheses):
Country: <country name>
Coordinates: <lat>, <lon>
Reasoning: <2-3 sentences explaining your choice and what evidence supported it>\
"""


async def finalize(state: dict, llm=None) -> str:
    """Make the final country determination based on Round 1 + all debate transcripts."""
    if llm is None:
        llm = get_vlm("judge", thinking=_THINKING_ENABLED)

    all_context = _format_all_for_judge(state)

    response = await llm.ainvoke([
        SystemMessage(content=_THINK_PREFIX + FINAL_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Complete Council Analysis (Round 1 + Debate):\n\n"
            f"{all_context}\n\n"
            "Based on ALL evidence from Round 1 and the debate transcripts, "
            "make your final determination. Provide Country, Coordinates, and Reasoning."
        )),
    ])
    return response.content


def _format_all_for_judge(state: dict) -> str:
    """Format Round 1 assessments + debate transcripts for the final judge."""
    agents = ["linguistic", "landscape", "botanics", "regulatory", "meta"]
    parts = []

    parts.append("=" * 60)
    parts.append("ROUND 1, Initial Independent Assessments")
    parts.append("=" * 60)

    for name in agents:
        assessment = state.get(f"round_1_{name}", {})
        parts.append(_format_single_assessment(name, assessment))

    debate_pairings = state.get("debate_pairings", [])
    if debate_pairings:
        parts.append("")
        parts.append("=" * 60)
        parts.append("DEBATE TRANSCRIPTS")
        parts.append("=" * 60)
        parts.append(_format_debate_history(debate_pairings))
    else:
        parts.append("")
        parts.append("(No debate was necessary, all agents agreed after Round 1)")

    return "\n\n".join(parts)


def _format_single_assessment(name: str, assessment: dict) -> str:
    """Format a single agent's Round 1 assessment."""
    candidates = assessment.get("candidates", [])
    evidence = assessment.get("evidence", [])
    evidence_str = ", ".join(str(e) for e in evidence) if evidence else "(none)"

    if not candidates:
        return f"[{name.upper()} AGENT]\n  (insufficient evidence)"

    cand_lines = []
    for c in candidates:
        country = c.get("country", "?")
        conf = c.get("confidence", "?")
        reasoning = c.get("reasoning", "")
        cand_lines.append(f"  - {country} ({conf}): {reasoning}")

    return (
        f"[{name.upper()} AGENT]\n"
        + "\n".join(cand_lines) + "\n"
        f"  Evidence: {evidence_str}"
    )
