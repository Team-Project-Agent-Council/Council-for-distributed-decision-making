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
- Driving side is the single most powerful feature — it eliminates ~70% of countries in one observation.
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

EVALUATE_PROMPT = """\
You are evaluating hypotheses about this image using ONLY your road infrastructure expertise.

Examine the image. For each hypothesis, determine how strongly your domain evidence supports or contradicts it.

CRITICAL RULES:
- You MUST differentiate. Do NOT give all hypotheses the same confidence level.
- Driving side is ELIMINATING evidence: if you can determine it, all wrong-side countries get "strongly_contradicts".
- License plate format, road sign style, center line color are highly diagnostic — use them to distinguish between hypotheses.
- "support" means: infrastructure is COMPATIBLE but shared across multiple countries (e.g. white center line in many countries).
- If you see a UNIQUE feature (specific plate format, unique sign style), give that hypothesis "strongly_support" and others lower.

HYPOTHESES:
{hypotheses_list}

Confidence scale:
- strongly_support: Unique infrastructure feature matches (specific plate format, unique sign design)
- support: Infrastructure compatible but shared with other candidates
- neutral: No visible infrastructure to evaluate
- contradicts: Infrastructure feature inconsistent (e.g. sign style not used in this country)
- strongly_contradicts: Hard rule-out (wrong driving side, impossible plate format)

Respond with JSON array only:
[{{"hypothesis_id": "...", "confidence": "...", "reasoning": "...", "key_evidence": [...]}}]\
"""

CONSTRAINED_ASSESS_PROMPT = """\
IMPORTANT CONSTRAINT: This image has been confirmed to be from {region}.
You MUST only propose countries within {region}. Do NOT suggest countries outside this region.

Now apply your road infrastructure expertise to identify specific COUNTRIES within {region}.
Analyze road infrastructure and determine which country within {region} it is from.

Respond with JSON only:
{{"candidates": [
  {{"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences>"}},
  ...
], "evidence": ["<driving side>", "<sign type>", "<line color>", "<plate format>"]}}\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("regulatory")
    msg = build_vlm_message(image_b64, image_mime, "Analyze road infrastructure and determine the country. Respond as JSON.")
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
