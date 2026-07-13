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

RE_GUESS_PROMPT = """\
This is ROUND 2 of the council. You previously gave your initial assessment in Round 1. Now you have access to ALL other agents' assessments from Round 1.

Your Round 1 assessment:
{own_round1}

All agents' Round 1 assessments:
{all_round1}

Based on the combined evidence from ALL agents, re-evaluate the image. Consider:
- Does evidence from other agents (driving side, vegetation, terrain, metas) support or contradict your initial guess?
- Should you adjust your ranking or confidence based on the collective evidence?
- Are there candidates you missed that other agents identified with strong evidence?

Look at the image again carefully and provide your UPDATED assessment. You may change your ranking, add new candidates, or adjust confidence levels.

Respond with JSON only:
{{"candidates": [
  {{"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences: why this country based on combined linguistic + council evidence>"}},
  ...
], "evidence": ["<exact text found>", "<language>", "<script>"]}}\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("linguistic")
    msg = build_vlm_message(image_b64, image_mime, "Find ALL visible text and determine the country. Respond as JSON.")
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content


async def re_guess(image_b64: str, image_mime: str, own_round1: str, all_round1: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("linguistic")
    prompt = RE_GUESS_PROMPT.format(own_round1=own_round1, all_round1=all_round1)
    msg = build_vlm_message(image_b64, image_mime, prompt)
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content
