"""Judge Agent: Makes final determination based on ALL Round 1 + Round 2 traces.

Global Context Re-guess approach: The judge receives the complete conversation history
from both rounds and synthesizes a final answer. No back-and-forth questions.
Thinking mode is enabled for the judge to allow deep reasoning.
"""

from __future__ import annotations

import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

from vlm_council.llm import get_vlm

_THINKING_ENABLED = os.environ.get("VLM_JUDGE_THINKING", "true").lower() in ("true", "1", "yes")
_THINK_PREFIX = "<|think|>\n" if _THINKING_ENABLED else ""

FINAL_SYSTEM_PROMPT = """\
You are the Judge of a GeoGuessr council. You receive the COMPLETE analysis from 5 specialist agents across TWO rounds:

- Round 1: Each agent independently analyzed the image
- Round 2: Each agent re-evaluated after seeing ALL other agents' Round 1 assessments

Your task is to synthesize ALL evidence from both rounds and make the FINAL country determination.

Decision process:
1. ELIMINATE first: check if any agent's evidence rules out candidates. Driving side, script, license plate format, or other hard constraints can immediately discard countries, regardless of how many agents suggested them.
2. Look at how agents CHANGED between rounds. If an agent shifted their top pick after seeing other evidence, that shift is informative, they found the collective evidence compelling enough to revise.
3. Evaluate ALL remaining candidates across all agents and both rounds. Specific evidence (identified text, unique road sign, endemic species) outweighs generic regional evidence (temperate climate, flat terrain).
4. For your chosen country, verify that no agent provided evidence that contradicts it in either round.

Estimate coordinates based on your chosen country and any regional clues from the agents.

Respond with EXACTLY this format:
Country: <name>
Coordinates: <lat>, <lon>
Reasoning: <2-3 sentences explaining your choice and what evidence supported it>\
"""


def _format_all_assessments(state: dict) -> str:
    """Format ALL Round 1 and Round 2 assessments for the judge."""
    agents = ["linguistic", "landscape", "botanics", "regulatory", "meta"]
    parts = []

    parts.append("=" * 60)
    parts.append("ROUND 1, Initial Independent Assessments")
    parts.append("=" * 60)

    for name in agents:
        assessment = state.get(f"round_1_{name}", {})
        parts.append(_format_single_assessment(name, assessment, round_num=1))

    parts.append("")
    parts.append("=" * 60)
    parts.append("ROUND 2, Re-guess After Seeing All Round 1 Evidence")
    parts.append("=" * 60)

    for name in agents:
        assessment = state.get(f"round_2_{name}", {})
        parts.append(_format_single_assessment(name, assessment, round_num=2))

    return "\n\n".join(parts)


def _format_single_assessment(name: str, assessment: dict, round_num: int) -> str:
    """Format a single agent's assessment."""
    candidates = assessment.get("candidates", [])
    evidence = assessment.get("evidence", [])
    evidence_str = ", ".join(str(e) for e in evidence) if evidence else "(none)"

    if not candidates:
        return f"[{name.upper()} AGENT, Round {round_num}]\n  (insufficient evidence)"

    cand_lines = []
    for c in candidates:
        country = c.get("country", "?")
        conf = c.get("confidence", "?")
        reasoning = c.get("reasoning", "")
        cand_lines.append(f"  - {country} ({conf}): {reasoning}")

    return (
        f"[{name.upper()} AGENT, Round {round_num}]\n"
        + "\n".join(cand_lines) + "\n"
        f"  Evidence: {evidence_str}"
    )


async def finalize(state: dict, llm=None) -> str:
    """Make the final country determination based on ALL rounds."""
    if llm is None:
        llm = get_vlm("judge", thinking=_THINKING_ENABLED)

    all_assessments_text = _format_all_assessments(state)

    response = await llm.ainvoke([
        SystemMessage(content=_THINK_PREFIX + FINAL_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Complete Council Analysis (Round 1 + Round 2):\n\n"
            f"{all_assessments_text}\n\n"
            "Based on ALL evidence from both rounds, make your final determination. "
            "Provide Country, Coordinates, and Reasoning."
        )),
    ])
    return response.content
