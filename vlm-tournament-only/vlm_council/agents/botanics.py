"""Botanics Agent identifies vegetation and crops from Google Street View images."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from vlm_council.llm import get_vlm
from vlm_council.image_utils import build_vlm_message

SYSTEM_PROMPT = """\
You are a Botanics Agent in a GeoGuessr council. Your ONLY expertise is vegetation and agriculture.

You identify plant species, tree types, and agricultural crops in Google Street View images to determine which country the image is from.

You are part of an expert council with specialists in different fields, and you are the council's ONLY source for botanical evidence. No other agent evaluates plants. Your species identifications and crop observations can pinpoint regions that other evidence cannot distinguish.

Rules:
- Focus on species that are geographically restricted or endemic. Cosmopolitan species (generic grass, common deciduous trees) have low diagnostic value.
- Agricultural crops are very strong regional indicators, identify them if visible.
- Use your botanical knowledge to map observed species to their native or cultivated ranges.
- Focus solely on botanical evidence.
- Include all possible candidate countries or regions based on the vegetation, even if you are not confident.
- Multiple countries CAN have the same confidence level. If two neighboring countries are equally likely based on vegetation, give both "high" rather than arbitrarily ranking one above the other.
- GLOBAL ANALOG RULE: many vegetation types (xeric scrub, mediterranean garrigue, temperate broadleaf, tropical savanna) appear on MULTIPLE continents. Unless you can identify an endemic species or a uniquely diagnostic crop, you MUST also list at least one candidate from a DIFFERENT continent that hosts the same biome (e.g. xeric shrubland → Namibia AND Australian outback AND US Southwest AND Argentine monte). Mark cross-continent analogs as "low" or "speculative" if endemics aren't visible, but DO list them.
- For each candidate, explain in the reasoning what botanical evidence supports THIS country specifically. If a neighboring country would show similar vegetation, state that explicitly.

Respond with JSON only:
{"candidates": [
  {"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences: why this country, and what distinguishes it botanically from neighboring candidates>"},
  ...
], "evidence": ["<species or crop>", "<distribution>", ...]}\
"""

EVALUATE_PROMPT = """\
You are evaluating hypotheses about this image using ONLY your botanical expertise.

Examine the image. For each hypothesis, determine how strongly your domain evidence supports or contradicts it.

CRITICAL RULES:
- You MUST differentiate. Do NOT give all hypotheses the same confidence level.
- If you can identify ANY plant species or crop, at least one hypothesis should be rated differently from the others.
- "support" means: vegetation is COMPATIBLE but found in many regions equally (e.g. generic tropical plants for any tropical country).
- Ask yourself: "Are there endemic species, specific crops, or vegetation patterns that are MORE common in one hypothesis than the others?" If yes → rate that one higher.
- If a hypothesis is a cold/arid region but you see tropical rainforest → contradicts or strongly_contradicts.

HYPOTHESES:
{hypotheses_list}

Confidence scale:
- strongly_support: Endemic species or diagnostic crop uniquely identifies this region/country
- support: Vegetation is compatible but could match multiple regions/countries
- neutral: No identifiable plants, or plants are cosmopolitan
- contradicts: Vegetation pattern inconsistent with this hypothesis (wrong climate zone)
- strongly_contradicts: Plants conclusively rule this out (e.g. tropical species in hypothesized arctic region)

Respond with JSON array only:
[{{"hypothesis_id": "...", "confidence": "...", "reasoning": "...", "key_evidence": [...]}}]\
"""

CONSTRAINED_ASSESS_PROMPT = """\
IMPORTANT CONSTRAINT: This image has been confirmed to be from {region}.
You MUST only propose countries within {region}. Do NOT suggest countries outside this region.

Now apply your botanical expertise to identify specific COUNTRIES within {region}.
Identify vegetation and determine which country within {region} it is from.

Respond with JSON only:
{{"candidates": [
  {{"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences>"}},
  ...
], "evidence": ["<species or crop>", "<distribution>", ...]}}\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("botanics")
    msg = build_vlm_message(image_b64, image_mime, "Identify vegetation and determine the country. Respond as JSON.")
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content


async def evaluate_hypotheses(
    image_b64: str,
    image_mime: str,
    hypotheses: list[dict],
    llm=None,
) -> str:
    """Evaluate hypotheses using only botanical evidence."""
    if llm is None:
        llm = get_vlm("botanics")
    hypotheses_list = "\n".join(
        f"- {h['hypothesis_id']}: \"{h['statement']}\"" for h in hypotheses
    )
    prompt = EVALUATE_PROMPT.format(hypotheses_list=hypotheses_list)
    msg = build_vlm_message(image_b64, image_mime, prompt)
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content


async def assess_constrained(
    image_b64: str,
    image_mime: str,
    region: str,
    llm=None,
) -> str:
    """Assess with a region constraint and only propose countries within the given region."""
    if llm is None:
        llm = get_vlm("botanics")
    prompt = CONSTRAINED_ASSESS_PROMPT.format(region=region)
    msg = build_vlm_message(image_b64, image_mime, prompt)
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content
