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

Respond with JSON only:
{"candidates": [
  {"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences: why this country, and what distinguishes it geographically from neighboring candidates>"},
  ...
], "evidence": ["<feature>", "<feature>", "<feature>"]}\
"""

RE_GUESS_PROMPT = """\
This is ROUND 2 of the council. You previously gave your initial assessment in Round 1. Now you have access to ALL other agents' assessments from Round 1.

Your Round 1 assessment:
{own_round1}

All agents' Round 1 assessments:
{all_round1}

Based on the combined evidence from ALL agents, re-evaluate the image. Consider:
- Does evidence from other agents (language/text, vegetation, road infrastructure, metas) support or contradict your initial guess?
- Should you adjust your ranking or confidence based on the collective evidence?
- Are there candidates you missed that other agents identified with strong evidence?

Look at the image again carefully and provide your UPDATED assessment. You may change your ranking, add new candidates, or adjust confidence levels.

Respond with JSON only:
{{"candidates": [
  {{"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences: why this country based on combined geographic + council evidence>"}},
  ...
], "evidence": ["<feature>", "<feature>", "<feature>"]}}\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("landscape")
    msg = build_vlm_message(image_b64, image_mime, "Analyze the geography and determine the country. Respond as JSON.")
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content


async def re_guess(image_b64: str, image_mime: str, own_round1: str, all_round1: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("landscape")
    prompt = RE_GUESS_PROMPT.format(own_round1=own_round1, all_round1=all_round1)
    msg = build_vlm_message(image_b64, image_mime, prompt)
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content
