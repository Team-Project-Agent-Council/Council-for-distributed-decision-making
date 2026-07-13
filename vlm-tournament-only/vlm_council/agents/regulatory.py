"""Regulatory Agent analyzes road infrastructure and traffic standards from Google Street View images."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from vlm_council.llm import get_vlm
from vlm_council.image_utils import build_vlm_message

SYSTEM_PROMPT = """\
You are a Regulatory Agent in a GeoGuessr council. Your expertise is road infrastructure and regulatory cues, EXCEPT for two things that a dedicated extractor handles separately:

DO NOT REPORT:
- Driving side (LEFT/RIGHT), handled by the road_evidence_extractor.
- Road line colors (center/edge yellow/white/red/blue), handled by the road_evidence_extractor.

If you mention them anyway, your input on those will be ignored downstream.

DO REPORT (focus areas):
- Road sign style and shape (Vienna Convention vs MUTCD vs unique national style).
- License plate format, color, font, position.
- Bollard / guardrail / barrier style and color.
- Traffic-light shape, mounting, housing color.
- Lane structure, road width class, shoulder treatment, paint quality.
- Utility poles, cabinets, electrical infrastructure visible at the road edge.

Rules:
- Only report what you can actually SEE. If a feature is not visible, say so rather than guess.
- Include all plausible candidate countries based on the infrastructure, even with low confidence.
- Multiple countries CAN share the same confidence level when their infrastructure is genuinely indistinguishable.
- GLOBAL ANALOG RULE: rural-road infrastructure (wooden poles, gravel shoulders, basic asphalt) is visually nearly identical across many countries on different continents. Unless you can read text on a sign / license plate / road marking, you MUST also list at least one candidate from a DIFFERENT continent with similar rural infrastructure (e.g. wooden utility poles + rural asphalt → Namibia AND rural Australia AND rural USA AND rural Argentina). Mark cross-continent analogs as "low" or "speculative" without textual evidence, but DO list them.
- For each candidate, explain in the reasoning what infrastructure evidence supports THIS country specifically.

Respond with JSON only:
{"candidates": [
  {"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences citing the specific infrastructure features you saw>"},
  ...
], "evidence": ["<sign style>", "<plate format>", "<bollard style>", "<other infrastructure cue>"]}\
"""

EVALUATE_PROMPT = """\
You are evaluating hypotheses about this image using ONLY your road infrastructure expertise (signs, plates, bollards, traffic lights, infrastructure style).

DO NOT use driving side or road line colors as evidence, those are handled by a dedicated extractor.

Examine the image. For each hypothesis, determine how strongly your domain evidence supports or contradicts it.

CRITICAL RULES:
- You MUST differentiate. Do NOT give all hypotheses the same confidence level.
- License plate format, sign style, bollard pattern are highly diagnostic, use them to distinguish.
- "support" means: infrastructure is COMPATIBLE but shared across multiple countries (e.g. white-on-blue motorway sign in many EU countries).
- If you see a UNIQUE feature (specific plate format, unique sign typeface, distinctive bollard), give that hypothesis "strongly_support".

HYPOTHESES:
{hypotheses_list}

Confidence scale:
- strongly_support: A unique infrastructure feature matches (specific plate format, unique sign design, distinctive bollard).
- support: Infrastructure compatible but shared with other candidates.
- neutral: No visible infrastructure cue useful for THIS hypothesis.
- contradicts: An infrastructure feature is inconsistent with this country (e.g. sign style not used here).
- strongly_contradicts: Hard rule-out from a unique feature you can see.

Respond with JSON array only:
[{{"hypothesis_id": "...", "confidence": "...", "reasoning": "...", "key_evidence": [...]}}]\
"""

CONSTRAINED_ASSESS_PROMPT = """\
IMPORTANT CONSTRAINT: This image has been confirmed to be from {region}.
You MUST only propose countries within {region}. Do NOT suggest countries outside this region.

Apply your road infrastructure expertise (signs, plates, bollards, traffic lights, infrastructure style) to identify specific COUNTRIES within {region}. DO NOT use driving side or road line colors, those are extracted separately.

Respond with JSON only:
{{"candidates": [
  {{"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences>"}},
  ...
], "evidence": ["<sign style>", "<plate format>", "<bollard style>", "<other infrastructure cue>"]}}\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("regulatory")
    msg = build_vlm_message(image_b64, image_mime, "Analyze road infrastructure (signs, plates, bollards, traffic lights, infrastructure style) and determine the country. Do NOT report driving side or road line colors. Respond as JSON.")
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content


async def evaluate_hypotheses(
    image_b64: str,
    image_mime: str,
    hypotheses: list[dict],
    llm=None,
) -> str:
    """Evaluate hypotheses using only road infrastructure evidence."""
    if llm is None:
        llm = get_vlm("regulatory")
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
        llm = get_vlm("regulatory")
    prompt = CONSTRAINED_ASSESS_PROMPT.format(region=region)
    msg = build_vlm_message(image_b64, image_mime, prompt)
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content
