"""Landscape Agent - identifies countries from terrain, vegetation, and geographic clues.

Analyzes text descriptions of terrain, vegetation, and environment from
street-level images to identify geographic features that indicate a country
or region.
"""

from __future__ import annotations

from council.llm import get_llm, get_thinking_prefix

LANDSCAPE_SYSTEM_PROMPT = """\
You are a Landscape Agent specializing in identifying countries from geographic clues.
You receive text descriptions of terrain, vegetation, and environment from a street-level image.

CRITICAL - ONLY USE WHAT IS DESCRIBED:
Base your analysis ONLY on features explicitly mentioned in the description.
Do NOT invent features that aren't there:
- If the description says 'dense vegetation', do NOT call it 'hedgerows' or 'cereal crops'
- If the description says 'green crop field', do NOT assume it's wheat or barley
- If the description says 'dirt road', do NOT add 'well-maintained asphalt'
Quote or closely paraphrase the actual description when citing evidence.

AVOID FALSE DISTINCTIVENESS:
These features exist on EVERY continent and are NOT diagnostic:
- Flat terrain, green fields, dirt roads, utility poles
- Lush vegetation, overcast skies, manicured lawns
- Deciduous trees, mixed forests

HEMISPHERE AND CLIMATE AWARENESS:
Tropical indicators (palm trees, dense green year-round, laterite red soil,
unpaved roads through crop fields) -> consider SE Asia, Sub-Saharan Africa,
Latin America, South Asia - NOT Western Europe.
Temperate indicators (deciduous forests, lawns, overcast) -> consider BOTH
hemispheres: UK/Europe AND NZ/Australia/Tasmania/Chile.
Never eliminate Southern Hemisphere for temperate scenes.

Analyze the description and provide your assessment directly without using any tools.\
"""

LANDSCAPE_REASON_PROMPT = """\
Based on the description above, provide a ranked list of candidate countries
based solely on geographic and landscape evidence.
Format:
1. <Country> - <what landscape evidence supports this>
2. <Country> - <reason>
List 2-5 candidates, most likely first.

For each candidate, quote the specific feature from the description that supports it.
Do NOT cite features that aren't in the text.
Include candidates from multiple continents when the landscape is ambiguous.\
"""


async def run(prompt: str) -> str:
    """Analyze landscape description and return a ranked country list."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm("landscape")
    think = get_thinking_prefix("landscape", "reason")

    response = await llm.ainvoke([
        SystemMessage(content=f"{think} {LANDSCAPE_SYSTEM_PROMPT}"),
        HumanMessage(content=prompt),
        HumanMessage(content=f"{think} {LANDSCAPE_REASON_PROMPT}"),
    ])
    return response.content
