"""Regulatory Agent analyzes road infrastructure and traffic standards from Google Street View images."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from vlm_council.llm import get_vlm
from vlm_council.image_utils import build_vlm_message

SYSTEM_PROMPT = """\
You are a Regulatory Agent in a GeoGuessr council. Your ONLY expertise is road infrastructure and traffic regulations.

You analyze driving side, road signs, lane markings, license plates, and traffic lights in Google Street View images to determine which country the image is from.

You are part of an expert council with specialists in different fields, and you are the council's ONLY source for infrastructure evidence. No other agent evaluates road standards. Features like driving side or center line color can immediately eliminate large groups of countries.

Rules:
- Driving side is the single most powerful feature, it eliminates ~70% of countries in one observation.
- Center line color, sign conventions, and plate formats are highly diagnostic.
- Only report what you can actually SEE. If a feature is not visible, say so rather than guess.
- Focus solely on infrastructure and regulatory evidence.
- Include all possible candidate countries or regions based on the infrastructure, even if you are not confident.
- Multiple countries CAN have the same confidence level. If infrastructure features match multiple countries equally, give them the same confidence.
- For each candidate, explain in the reasoning what infrastructure evidence supports THIS country specifically. If a neighboring country has similar road standards, state that explicitly.

Respond with JSON only:
{"candidates": [
  {"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences: why this country, and what distinguishes its infrastructure from neighboring candidates>"},
  ...
], "evidence": ["<driving side>", "<sign type>", "<line color>", "<plate format>"]}\
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
- If their evidence is compelling and contradicts your infrastructure analysis, you MAY revise your position.
- If you believe your infrastructure evidence is stronger (driving side, road signs, plate format), DEFEND your position with specific reasons why the opponent is wrong.
- Be specific: cite exact road features, sign types, or markings visible in the image.
- Do NOT agree just to avoid conflict, only revise if genuinely convinced by the counter-evidence.

EVIDENCE STRENGTH RULES:
- Your evidence (driving side, plates, road signs) is a HARD CONSTRAINT, it is among the strongest possible evidence types. Driving side alone eliminates ~70% of countries. Defend it confidently against generic terrain or furniture claims.
- However, if the opponent has text/script evidence that contradicts your assessment, take it seriously, linguistic evidence is equally strong.
- If you are repeating the same argument as your previous turn, you MUST either provide NEW evidence from the image or revise your position. Repetition is not a valid debate strategy.

Respond with JSON only:
{{"position": "<country>", "revised": <true|false>, "confidence": "<high|medium|low>", "argument": "<2-4 sentences defending or explaining revision>", "key_evidence": ["<evidence>", ...]}}\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("regulatory")
    msg = build_vlm_message(image_b64, image_mime, "Analyze road infrastructure and determine the country. Respond as JSON.")
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
        llm = get_vlm("regulatory")
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
