"""Regulatory Agent - identifies countries from road markings, signs, and driving conventions.

Analyzes road marking standards, sign conventions, driving side, and utility
infrastructure to identify the country based on regulatory patterns.
"""

from __future__ import annotations

from council.llm import get_llm, get_thinking_prefix

REGULATORY_SYSTEM_PROMPT = """\
You are a Regulatory Agent specialising in identifying countries from road
markings, sign standards, driving conventions, and utility infrastructure visible in
street-level images.

Use your training knowledge of national road marking standards.

ROAD MARKINGS - follow this exact process:
Step 1: List EVERY road line/marking mentioned in the description with its stated color
and stated position. Only include what is explicitly written.
Step 2: If a color or position is NOT mentioned for a line, it does NOT exist.
Do NOT add yellow edge lines if the description only mentions white lines.
Step 3: Match the exact described combination against your knowledge of national standards.

Also consider: sign shapes/colors, driving side, license plate formats, utility pole
designs, traffic light configurations.

Analyze the description and provide your assessment directly without using any tools.\
"""

REGULATORY_REASON_PROMPT = """\
Based on the description above, provide a ranked list of candidate countries
based solely on regulatory and road infrastructure evidence.
Format:
1. <Country> - <what regulatory evidence supports this>
2. <Country> - <reason>
List 2-5 candidates, most likely first.

FIRST: quote exactly which road markings are described (colors and positions as stated
in the text - do not add any that aren't there). THEN match that combination.\
"""


async def run(prompt: str) -> str:
    """Analyze regulatory details and return a ranked country list."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm("regulatory")
    think = get_thinking_prefix("regulatory", "reason")

    response = await llm.ainvoke([
        SystemMessage(content=f"{think} {REGULATORY_SYSTEM_PROMPT}"),
        HumanMessage(content=prompt),
        HumanMessage(content=f"{think} {REGULATORY_REASON_PROMPT}"),
    ])
    return response.content
