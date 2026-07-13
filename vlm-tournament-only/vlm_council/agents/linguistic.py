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

EVALUATE_PROMPT = """\
You are evaluating hypotheses about this image using ONLY your linguistic expertise.

Examine the image. For each hypothesis, determine how strongly your domain evidence supports or contradicts it.

CRITICAL RULES:
- You MUST differentiate. Do NOT give all hypotheses the same confidence level.
- If you have no text evidence at all, give ALL hypotheses "neutral", but this should be rare.
- If you CAN see text: at least one hypothesis should get "strongly_support" or "strongly_contradicts" based on what the text says.
- "support" means: the language/script is COMPATIBLE but not uniquely diagnostic (shared across multiple countries).
- "contradicts" means: you can see text in a DIFFERENT language/script than expected for this hypothesis.
- Ask yourself: "Does this text UNIQUELY point to one hypothesis over the others?" If yes → strongly_support that one and contradicts/strongly_contradicts the others.

HYPOTHESES:
{hypotheses_list}

Confidence scale:
- strongly_support: Text/language uniquely identifies this region/country (e.g. Thai script → Thailand)
- support: Language is compatible but shared with other candidates (e.g. Spanish → could be many countries)
- neutral: No visible text, or text gives no information about this hypothesis
- contradicts: Visible text is in a different language/script than expected
- strongly_contradicts: Text conclusively rules this out (e.g. Cyrillic text → cannot be Southeast Asia)

Respond with JSON array only:
[{{"hypothesis_id": "...", "confidence": "...", "reasoning": "...", "key_evidence": [...]}}]\
"""

CONSTRAINED_ASSESS_PROMPT = """\
IMPORTANT CONSTRAINT: This image has been confirmed to be from {region}.
You MUST only propose countries within {region}. Do NOT suggest countries outside this region.

Now apply your linguistic expertise to identify specific COUNTRIES within {region}.
Find ALL visible text and determine which country within {region} it is from.

Respond with JSON only:
{{"candidates": [
  {{"country": "<name>", "confidence": "<high|medium|low|speculative>", "reasoning": "<2-3 sentences>"}},
  ...
], "evidence": ["<exact text found>", "<language>", "<script>"]}}\
"""


async def assess(image_b64: str, image_mime: str, llm=None) -> str:
    if llm is None:
        llm = get_vlm("linguistic")
    msg = build_vlm_message(image_b64, image_mime, "Find ALL visible text and determine the country. Respond as JSON.")
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content


async def evaluate_hypotheses(
    image_b64: str,
    image_mime: str,
    hypotheses: list[dict],
    llm=None,
) -> str:
    """Evaluate hypotheses using only linguistic evidence."""
    if llm is None:
        llm = get_vlm("linguistic")
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
        llm = get_vlm("linguistic")
    prompt = CONSTRAINED_ASSESS_PROMPT.format(region=region)
    msg = build_vlm_message(image_b64, image_mime, prompt)
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    return response.content
