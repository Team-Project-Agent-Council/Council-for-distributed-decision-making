"""Judge Agent: Progressive Narrowing"""

from __future__ import annotations

import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

from vlm_council.llm import get_vlm

# Gemma 4 thinking token
_THINKING_ENABLED = os.environ.get("VLM_JUDGE_THINKING", "false").lower() in ("true", "1", "yes")
_THINK_PREFIX = "<|think|>\n" if _THINKING_ENABLED else ""

def _strip_think_tags(text: str) -> tuple[str, str]:
    """Separate thinking chain from the actual response."""
    # Explicit <think>...</think> wrapper
    think_match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), think_match.group(2).strip()
    # Gemma 4: <|channel>thought...<channel|>
    channel_match = re.search(r"<\|channel\>thought(.*?)<channel\|>(.*)", text, re.DOTALL)
    if channel_match:
        return channel_match.group(1).strip(), channel_match.group(2).strip()
    # </think> without opening tag
    think_end = re.search(r"</think>(.*)", text, re.DOTALL)
    if think_end:
        return text[:think_end.start()].strip(), think_end.group(1).strip()
    return "", text


def _parse_json_from_response(text: str) -> dict | list | None:
    """Extract JSON object or array from LLM response."""
    _, response = _strip_think_tags(text)
    text = response.strip() if response else text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    # Try to find JSON object
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _format_assessments(state: dict, prefix: str = "") -> str:
    """Format agent assessments for the judge.

    prefix="" uses initial assessments (linguistic_assessment, etc.)
    prefix="country_" uses constrained assessments (linguistic_country_assessment, etc.)
    """
    agents = ["linguistic", "landscape", "botanics", "regulatory", "meta"]
    parts = []
    for name in agents:
        key = f"{name}_{prefix}assessment" if prefix else f"{name}_assessment"
        assessment = state.get(key, {})
        candidates = assessment.get("candidates", [])
        evidence = assessment.get("evidence", [])
        evidence_str = ", ".join(str(e) for e in evidence) if evidence else "(none)"

        if not candidates:
            parts.append(f"[{name.upper()} AGENT]\n  (insufficient evidence)")
            continue

        cand_lines = []
        for c in candidates:
            country = c.get("country", "?")
            conf = c.get("confidence", "?")
            reasoning = c.get("reasoning", "")
            cand_lines.append(f"  - {country} ({conf}): {reasoning}")
        parts.append(
            f"[{name.upper()} AGENT]\n"
            + "\n".join(cand_lines) + "\n"
            f"  Evidence: {evidence_str}"
        )
    return "\n\n".join(parts)


#  1. Region Consensus Check

CONSENSUS_SYSTEM_PROMPT = """\
You are the Judge of a GeoGuessr council analyzing whether all 5 specialist agents agree on the world REGION.

Given the assessments below:
1. Map EVERY candidate from EVERY agent to its global region (Europe, East Asia, Southeast Asia, South Asia, Central Asia, Middle East, North Africa, Sub-Saharan Africa, North America, Central America & Caribbean, South America, Oceania).
2. Build a region_candidates object: for each region, list every country candidate and how many agents proposed it.
3. Check if ALL candidates from ALL agents fall within the SAME region.
4. STRICT RULE: If even ONE candidate from ANY agent belongs to a DIFFERENT region, consensus is FALSE.
5. If consensus exists, state which region.
6. If no consensus, list ALL unique regions that appear.

Respond with JSON only:
{"consensus": true/false, "consensus_region": "<region or null>", "proposed_regions": ["<region1>", ...], "region_candidates": {"<region>": {"<country>": <agent_count>, ...}, ...}}\
"""


async def check_region_consensus(state: dict, llm=None) -> dict:
    """Analyze all 5 assessments and check if there's region consensus.

    Returns region_candidates mapping: {region: {country: count}}
    """
    if llm is None:
        llm = get_vlm("judge")

    assessments_text = _format_assessments(state)

    response = await llm.ainvoke([
        SystemMessage(content=_THINK_PREFIX + CONSENSUS_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Agent Assessments:\n\n{assessments_text}\n\n"
            "Determine region consensus. Respond with JSON only."
        )),
    ])

    parsed = _parse_json_from_response(response.content)
    if parsed and isinstance(parsed, dict):
        consensus = parsed.get("consensus", False)
        consensus_region = parsed.get("consensus_region")
        proposed_regions = parsed.get("proposed_regions", [])
        region_candidates = parsed.get("region_candidates", {})

        if not proposed_regions and consensus_region:
            proposed_regions = [consensus_region]

        return {
            "consensus": bool(consensus),
            "consensus_region": consensus_region if consensus else None,
            "proposed_regions": proposed_regions,
            "region_candidates": region_candidates,
        }

    # Fallback: if LLM response is unparseable, assume no consensus
    return {"consensus": False, "consensus_region": None, "proposed_regions": [], "region_candidates": {}}


# 2. Region Decision

REGION_DECISION_SYSTEM_PROMPT = """\
You are the Judge deciding which REGION this image is from based on hypothesis evaluations from 5 specialist agents.

Each agent evaluated region hypotheses with confidence levels:
- strongly_support (weight: +2)
- support (weight: +1)
- neutral (weight: 0)
- contradicts (weight: -1)
- strongly_contradicts (weight: -2)

Decision rules:
1. Calculate weighted score for each region by summing all agents' confidence weights.
2. If any agent gives "strongly_contradicts" with hard physical evidence, that region should be heavily penalized.
3. Choose the region with the highest weighted score.
4. If tied, prefer the region with more "strongly_support" votes.

Respond with JSON only:
{"decided_region": "<region name>", "reasoning": "<2-3 sentences explaining why>"}\
"""


async def decide_region(state: dict, llm=None) -> dict:
    """Decide on the region based on hypothesis evaluations."""
    if llm is None:
        llm = get_vlm("judge")

    evaluations = state.get("hypothesis_evaluations", [])
    region_evals = [e for e in evaluations if "region_" in e.get("hypothesis_id", "")]

    # Format evaluations for the judge
    eval_text_parts = []
    for e in region_evals:
        eval_text_parts.append(
            f"  [{e['agent_name']}] {e['hypothesis_id']}: {e['confidence']}, {e['reasoning']}"
        )
    eval_text = "\n".join(eval_text_parts) if eval_text_parts else "(no evaluations)"

    response = await llm.ainvoke([
        SystemMessage(content=_THINK_PREFIX + REGION_DECISION_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Hypothesis Evaluations:\n{eval_text}\n\n"
            "Decide the region. Respond with JSON only."
        )),
    ])

    parsed = _parse_json_from_response(response.content)
    if parsed and isinstance(parsed, dict) and parsed.get("decided_region"):
        return {
            "decided_region": parsed["decided_region"],
            "reasoning": parsed.get("reasoning", ""),
        }

    # If LLM response unparseable, use first proposed region
    proposed = state.get("proposed_regions", [])
    return {
        "decided_region": proposed[0] if proposed else "Unknown",
        "reasoning": "LLM response unparseable.",
    }



#  3. Country Decision

COUNTRY_DECISION_SYSTEM_PROMPT = """\
You are the Judge making the final COUNTRY determination for a GeoGuessr image.

The image has been confirmed to be from: {region}

You have hypothesis evaluations from 5 specialist agents for country-level hypotheses.
Each agent evaluated with confidence levels:
- strongly_support (+2), support (+1), neutral (0), contradicts (-1), strongly_contradicts (-2)

Decision process:
1. Calculate weighted score for each country.
2. If any agent gives "strongly_contradicts" with hard evidence, that country should be heavily penalized.
3. Among top-scoring countries, prefer the one with the most specific evidence.
4. Estimate coordinates based on your chosen country and evidence clues.

Respond with EXACTLY this format:
Country: <name>
Coordinates: <lat>, <lon>
Reasoning: <2-3 sentences explaining your choice and what evidence supported it>\
"""


async def decide_country(state: dict, llm=None) -> str:
    """Make the final country determination based on hypothesis evaluations."""
    if llm is None:
        llm = get_vlm("judge")

    confirmed_region = state.get("confirmed_region", "Unknown")
    evaluations = state.get("hypothesis_evaluations", [])
    country_evals = [e for e in evaluations if "country_" in e.get("hypothesis_id", "")]

    # Format evaluations
    eval_text_parts = []
    for e in country_evals:
        eval_text_parts.append(
            f"  [{e['agent_name']}] {e['hypothesis_id']}: {e['confidence']}, {e['reasoning']}"
        )
    eval_text = "\n".join(eval_text_parts) if eval_text_parts else "(no evaluations)"

    # Also include the assessments for context
    if state.get("region_consensus", False):
        assessments_text = _format_assessments(state, prefix="")
    else:
        assessments_text = _format_assessments(state, prefix="country_")

    system = COUNTRY_DECISION_SYSTEM_PROMPT.format(region=confirmed_region)

    response = await llm.ainvoke([
        SystemMessage(content=_THINK_PREFIX + system),
        HumanMessage(content=(
            f"Confirmed Region: {confirmed_region}\n\n"
            f"Agent Assessments:\n{assessments_text}\n\n"
            f"Country Hypothesis Evaluations:\n{eval_text}\n\n"
            "Make your final determination. Provide Country, Coordinates, and Reasoning."
        )),
    ])

    return response.content
