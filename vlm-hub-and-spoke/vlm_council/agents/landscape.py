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

DISCUSSION_PROMPT = """\
The Judge is challenging your assessment based on evidence from other agents in the council.

Your previous assessment:
{previous_assessment}

Judge's question:
{question}

Reevaluate the terrain and geography in the image with the Judge's question in mind. If the Judge mentions a competing candidate country, specifically address what geographic evidence distinguishes your top pick from that alternative. If you cannot name a concrete geographic differentiator, reconsider your ranking.

Respond with an updated JSON assessment.\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("landscape")
    msg = build_vlm_message(image_b64, image_mime, "Analyze the geography and determine the country. Respond as JSON.")
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content


async def discuss(image_b64: str, image_mime: str, previous: str, question: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("landscape")
    prompt = DISCUSSION_PROMPT.format(previous_assessment=previous, question=question)
    msg = build_vlm_message(image_b64, image_mime, prompt)
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content
