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

DISCUSSION_PROMPT = """\
The Judge is challenging your assessment based on evidence from other agents in the council.

Your previous assessment:
{previous_assessment}

Judge's question:
{question}

Reevaluate the road infrastructure in the image with the Judge's question in mind. If the Judge mentions a competing candidate country, specifically address what infrastructure evidence distinguishes your top pick from that alternative. If you cannot name a concrete infrastructure differentiator, reconsider your ranking.

Respond with an updated JSON assessment.\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("regulatory")
    msg = build_vlm_message(image_b64, image_mime, "Analyze road infrastructure and determine the country. Respond as JSON.")
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content


async def discuss(image_b64: str, image_mime: str, previous: str, question: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("regulatory")
    prompt = DISCUSSION_PROMPT.format(previous_assessment=previous, question=question)
    msg = build_vlm_message(image_b64, image_mime, prompt)
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content
