"""Landscape Agent analyzes terrain, climate, and settlement patterns from Google Street View images."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from vlm_council.llm import get_vlm
from vlm_council.image_utils import build_vlm_message

SYSTEM_PROMPT = """\
You are a Landscape Agent in a GeoGuessr council. Your ONLY expertise is geography and physical environment.

You analyze terrain, climate, soil, and settlement patterns in Google Street View images to determine which country the image is from.

You are part of an expert council with specialists in different fields, and you are the council's ONLY source for geographic evidence. No other agent evaluates terrain, hemisphere, or settlement style. Your observations shape the council's understanding of WHERE on Earth this could be.

Rules:
- Determine the hemisphere from shadow direction and sun angle.
- Identify distinctive soil colors, terrain types, and settlement patterns.
- Note road condition and surroundings as landscape features.
- Focus solely on geographic and environmental evidence.
- Include all possible candidate countries or regions based on the landscape, even if you are not confident.
- Multiple countries CAN have the same confidence level. If two neighboring countries are equally likely based on geography, give both the same confidence rather than arbitrarily ranking one above the other.
- For each candidate, explain in the reasoning what geographic evidence supports THIS country specifically. If a neighboring country has similar terrain, state that explicitly.

CONFIDENCE CALIBRATION:
- HIGH: Only for terrain that is genuinely unique, red Outback soil, Scandinavian fjords, Andean altiplano, African Rift Valley, distinctive karst formations. The combination must narrow to 3 or fewer countries.
- MEDIUM: Regional terrain (e.g., "East European Plain", could be 10 countries, "Southeast Asian lowlands").
- LOW: Generic features (flat road, temperate climate, deciduous trees, semi-arid soil). These exist on every continent and should never drive your top candidate alone.
- If your top candidate is based only on "flat terrain" + "general climate", you do NOT have high confidence.

Respond with JSON only:
{"candidates": [
  {"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences: why this country, and what distinguishes it geographically from neighboring candidates>"},
  ...
], "evidence": ["<feature>", "<feature>", "<feature>"]}\
"""

DEBATE_PROMPT = """\
This is a DEBATE ROUND. The {opponent_name} agent disagrees with your assessment.

Your current position:
{own_position}

The {opponent_name} agent's position:
{opponent_position}

{debate_history_section}

All Round 1 assessments (for reference):
{all_round1_context}

INSTRUCTIONS:
- Look at the image again carefully.
- Consider the opponent's evidence and reasoning.
- If their evidence is compelling and contradicts your geographic analysis, you MAY revise your position.
- If you believe your geographic/terrain evidence is stronger, DEFEND your position with specific reasons why the opponent is wrong.
- Be specific: cite exact terrain features, soil colors, settlement patterns visible in the image.
- Do NOT agree just to avoid conflict, only revise if genuinely convinced by the counter-evidence.

EVIDENCE STRENGTH RULES:
- HARD CONSTRAINTS (text/script, driving side, license plates, endemic species) OVERRIDE soft evidence like terrain and climate. If the opponent has a hard constraint that eliminates your country, you SHOULD revise.
- Your terrain/climate evidence is SOFT, it narrows regions but rarely pinpoints a single country. Be honest about this limitation.
- If you are repeating the same argument as your previous turn, you MUST either provide NEW evidence from the image or revise your position. Repetition is not a valid debate strategy.

Respond with JSON only:
{{"position": "<country>", "revised": <true|false>, "confidence": "<high|medium|low>", "argument": "<2-4 sentences defending or explaining revision>", "key_evidence": ["<evidence>", ...]}}\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("landscape")
    msg = build_vlm_message(image_b64, image_mime, "Analyze the geography and determine the country. Respond as JSON.")
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content


async def debate(
    image_b64: str,
    image_mime: str,
    own_position: str,
    opponent_name: str,
    opponent_position: str,
    debate_history: str,
    all_round1_context: str,
    llm=None,
) -> str:
    if llm is None:
        llm = get_vlm("landscape")
    history_section = f"Previous debate exchanges:\n{debate_history}" if debate_history else ""
    prompt = DEBATE_PROMPT.format(
        opponent_name=opponent_name,
        own_position=own_position,
        opponent_position=opponent_position,
        debate_history_section=history_section,
        all_round1_context=all_round1_context,
    )
    msg = build_vlm_message(image_b64, image_mime, prompt)
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content
