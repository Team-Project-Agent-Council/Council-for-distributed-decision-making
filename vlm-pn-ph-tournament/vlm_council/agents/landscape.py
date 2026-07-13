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
- GLOBAL ANALOG RULE: arid scrubland, mountainous terrain, temperate forest, tropical jungle, etc. each exist on MULTIPLE continents. After listing your top regional candidates, you MUST also include at least one candidate from a DIFFERENT continent whose climate/terrain matches (e.g. arid red-soil scrubland → Namibia AND Australian outback AND US Southwest AND Patagonia). Mark cross-continent analogs as "low" or "speculative" if you have no other evidence, but DO list them, do not narrow to one continent until other evidence forces you to.
- For each candidate, explain in the reasoning what geographic evidence supports THIS country specifically. If a neighboring country has similar terrain, state that explicitly.

Respond with JSON only:
{"candidates": [
  {"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences: why this country, and what distinguishes it geographically from neighboring candidates>"},
  ...
], "evidence": ["<feature>", "<feature>", "<feature>"]}\
"""

EVALUATE_PROMPT = """\
You are evaluating hypotheses about this image using ONLY your landscape/geography expertise.

Examine the image. For each hypothesis, determine how strongly your domain evidence supports or contradicts it.

CRITICAL RULES:
- You MUST differentiate. Do NOT give all hypotheses the same confidence level.
- At least one hypothesis should be rated HIGHER than the others based on terrain, climate, hemisphere, or soil.
- "support" means: the landscape is COMPATIBLE but could match multiple regions/countries equally.
- Ask yourself for EACH hypothesis: "Is there something in the terrain that is MORE consistent with this hypothesis than the others?" If yes → give it a higher rating.
- If the hemisphere is wrong for a hypothesis → strongly_contradicts.
- If terrain type is impossible (e.g. tropical jungle for a Nordic country) → contradicts.

HYPOTHESES:
{hypotheses_list}

Confidence scale:
- strongly_support: Terrain/climate/soil uniquely matches this region/country (e.g. red laterite soil → Sub-Saharan Africa)
- support: Landscape is compatible but not uniquely diagnostic
- neutral: No geographic evidence relevant to this hypothesis
- contradicts: Terrain features inconsistent (e.g. flat plains for a mountainous country)
- strongly_contradicts: Physically impossible (e.g. wrong hemisphere, tropical vegetation for arctic)

Respond with JSON array only:
[{{"hypothesis_id": "...", "confidence": "...", "reasoning": "...", "key_evidence": [...]}}]\
"""

CONSTRAINED_ASSESS_PROMPT = """\
IMPORTANT CONSTRAINT: This image has been confirmed to be from {region}.
You MUST only propose countries within {region}. Do NOT suggest countries outside this region.

Now apply your landscape expertise to identify specific COUNTRIES within {region}.
Analyze the geography and determine which country within {region} it is from.

Respond with JSON only:
{{"candidates": [
  {{"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences>"}},
  ...
], "evidence": ["<feature>", "<feature>", "<feature>"]}}\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("landscape")
    msg = build_vlm_message(image_b64, image_mime, "Analyze the geography and determine the country. Respond as JSON.")
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content


async def evaluate_hypotheses(
    image_b64: str,
    image_mime: str,
    hypotheses: list[dict],
    llm=None,
) -> str:
    """Evaluate hypotheses using only landscape/geography evidence."""
    if llm is None:
        llm = get_vlm("landscape")
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
        llm = get_vlm("landscape")
    prompt = CONSTRAINED_ASSESS_PROMPT.format(region=region)
    msg = build_vlm_message(image_b64, image_mime, prompt)
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content
