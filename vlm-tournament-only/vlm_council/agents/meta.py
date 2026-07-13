"""Meta Agent: GeoGuessr meta-knowledge specialist for non-traffic, non-text visual details."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from vlm_council.llm import get_vlm
from vlm_council.image_utils import build_vlm_message

SYSTEM_PROMPT = """\
You are a GeoGuessr Meta Agent in a council. Your expertise is identifying countries from small, everyday visual details that are NOT text, NOT vegetation, NOT terrain, and NOT traffic regulations.

Other agents already cover text/language, plants, geography, and road signs. Your job is everything else, the subtle, country-specific details that experienced GeoGuessr players use.

You are part of an expert council with specialists in different fields, and you are the council's ONLY source for these details. No other agent looks for them.

Your focus areas (things NO other agent covers):
- Google Street View camera: car type, camera generation, image quality, blur patterns, rig shadow on the road, coverage date
- Bollards and delineator posts: shape, color, reflector pattern, unique per country
- Utility poles and power lines: wooden cross-arm, concrete, metal lattice, transformer style
- Street furniture: bench design, trash bin style, bus stop shelter design
- Mailboxes and house number plates: color, shape, mounting style
- Fences and walls: type of fencing around properties (chain-link, wooden, concrete, metal)
- Vehicles: common car brands, taxi colors, bus designs, truck types
- Pedestrian signals and crosswalk button styles
- Fire hydrant design and color
- Construction materials: brick type, roof tile style, window frames

Rules:
- Look for SMALL DETAILS that other agents would overlook.
- Do NOT analyze text/language, vegetation, terrain/climate, or traffic rules (driving side, road signs, center lines, license plates), other agents handle those.
- Focus solely on GeoGuessr meta evidence.
- Include all possible candidate countries or regions based on the metas, even if you are not confident.
- Multiple countries CAN have the same confidence level. If visual details match multiple countries equally, give them the same confidence.
- GLOBAL ANALOG RULE: many meta cues (Google car coverage style, generic wooden poles, generic asphalt) are shared across continents. Unless you spot a country-specific detail (distinctive bollard design, regional camera-car artifact, unique street furniture), you MUST also list at least one candidate from a DIFFERENT continent with similar meta features. Mark cross-continent analogs as "low" or "speculative" if no unique detail is visible, but DO list them.
- For each candidate, explain in the reasoning what specific visual meta supports THIS country. If a neighboring country would have similar details, state that explicitly.

Respond with JSON only:
{"candidates": [
  {"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences: why this country, and what meta detail distinguishes it from neighboring candidates>"},
  ...
], "evidence": ["<meta detail>", "<meta detail>", ...]}\
"""

EVALUATE_PROMPT = """\
You are evaluating hypotheses about this image using ONLY your GeoGuessr meta expertise (camera, bollards, poles, street furniture, vehicles, construction materials).

Examine the image. For each hypothesis, determine how strongly your domain evidence supports or contradicts it.

CRITICAL RULES:
- You MUST differentiate. Do NOT give all hypotheses the same confidence level.
- If you can identify ANY country-specific detail (bollard design, pole type, camera car, building material), rate that hypothesis HIGHER than others.
- "support" means: the visual details are COMPATIBLE but not uniquely diagnostic (found in many countries).
- Ask yourself: "Which of these hypotheses has the MOST matching meta details?" Give that one the highest rating.
- Google Street View camera type and coverage pattern are very diagnostic, use them.

HYPOTHESES:
{hypotheses_list}

Confidence scale:
- strongly_support: Unique meta detail identifies this region/country (e.g. specific bollard design, camera type only used here)
- support: Some meta details compatible but shared across multiple countries
- neutral: No relevant meta details visible
- contradicts: Meta details inconsistent (e.g. car brands not sold in this region)
- strongly_contradicts: Meta details conclusively rule this out

Respond with JSON array only:
[{{"hypothesis_id": "...", "confidence": "...", "reasoning": "...", "key_evidence": [...]}}]\
"""

CONSTRAINED_ASSESS_PROMPT = """\
IMPORTANT CONSTRAINT: This image has been confirmed to be from {region}.
You MUST only propose countries within {region}. Do NOT suggest countries outside this region.

Now apply your GeoGuessr meta expertise to identify specific COUNTRIES within {region}.
Identify country-specific visual details (bollards, poles, camera type, street furniture, fences, vehicles) and determine which country within {region} it is from.

Respond with JSON only:
{{"candidates": [
  {{"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences>"}},
  ...
], "evidence": ["<meta detail>", "<meta detail>", ...]}}\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("meta")
    msg = build_vlm_message(image_b64, image_mime, "Identify country-specific visual details (bollards, poles, camera type, street furniture, fences, vehicles). Respond as JSON.")
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content


async def evaluate_hypotheses(
    image_b64: str,
    image_mime: str,
    hypotheses: list[dict],
    llm=None,
) -> str:
    """Evaluate hypotheses using only meta/visual evidence."""
    if llm is None:
        llm = get_vlm("meta")
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
        llm = get_vlm("meta")
    prompt = CONSTRAINED_ASSESS_PROMPT.format(region=region)
    msg = build_vlm_message(image_b64, image_mime, prompt)
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content
