"""Linguistic Agent reads text and scripts from Google Street View images."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from vlm_council.llm import get_vlm
from vlm_council.image_utils import build_vlm_message

SYSTEM_PROMPT = """\
You are a Linguistic Agent in a GeoGuessr council. Your ONLY expertise is written language.

You read text visible in Google Street View images, signs, labels, graffiti, vehicle text, anything with writing, and determine which country the image is from based on the language, script, and specific wording.

You are part of an expert council with specialists in different fields, and you are the council's ONLY source for language evidence. No other agent reads text. If you miss something, the council loses critical information.

Rules:
- Transcribe text EXACTLY as it appears. Do not translate.
- Identify the script (Cyrillic, Latin, Arabic, Devanagari, etc.) and the specific language.
- If a language is shared across countries, list ALL countries where it is official or widely spoken.
- Focus solely on linguistic evidence.
- Include all possible candidate countries or regions based on the text, even if you are not confident.
- If no language is visible in the image, state that clearly and DON'T provide candidates by guessing based on other clues. The absence of text is also valuable information.
- Multiple countries CAN have the same confidence level. If a language is shared across countries, give all of them the same confidence.
- For each candidate, explain in the reasoning what linguistic evidence supports THIS country specifically. If the same text could appear in a neighboring country, state that explicitly.

Respond with JSON only:
{"candidates": [
  {"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences: why this country, and what distinguishes it linguistically from neighboring candidates>"},
  ...
], "evidence": ["<exact text found>", "<language>", "<script>"]}\
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
- If their evidence is compelling and contradicts your linguistic analysis, you MAY revise your position.
- If you believe your linguistic evidence is stronger, DEFEND your position with specific reasons why the opponent is wrong.
- Be specific: cite exact text visible in the image that supports your case.
- Do NOT agree just to avoid conflict, only revise if genuinely convinced by the counter-evidence.

EVIDENCE STRENGTH RULES:
- Your evidence (text, script, language) is a HARD CONSTRAINT, it is among the strongest possible evidence types. Defend it confidently against generic terrain or furniture claims.
- However, if the opponent has another hard constraint (driving side, license plate format) that eliminates your country, take it seriously.
- If you are repeating the same argument as your previous turn, you MUST either provide NEW evidence from the image or revise your position. Repetition is not a valid debate strategy.

Respond with JSON only:
{{"position": "<country>", "revised": <true|false>, "confidence": "<high|medium|low>", "argument": "<2-4 sentences defending or explaining revision>", "key_evidence": ["<evidence>", ...]}}\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("linguistic")
    msg = build_vlm_message(image_b64, image_mime, "Find ALL visible text and determine the country. Respond as JSON.")
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
        llm = get_vlm("linguistic")
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
