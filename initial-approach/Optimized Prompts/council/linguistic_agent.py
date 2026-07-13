"""Linguistic Agent - identifies countries from language and script clues.

Analyzes text snippets extracted from signs, posters, and labels visible
in street-level images to identify language-specific features that indicate
a country or region.
"""

from __future__ import annotations

from council.llm import get_llm, get_thinking_prefix

LINGUISTIC_SYSTEM_PROMPT = """\
You are a Linguistic Agent specializing in identifying countries from language clues.
You receive text snippets extracted from signs, posters, and labels visible in a
street-level image.

Analyze the text and provide your assessment directly without using any tools.\
"""

LINGUISTIC_REASON_PROMPT = """\
Based on the text above, provide a ranked list of candidate countries
based solely on linguistic evidence.
Format:
1. <Country> - <what linguistic evidence supports this>
2. <Country> - <reason>
List 2-5 candidates, most likely first.

State clearly what evidence you have and what you don't.
If no country-specific linguistic feature was identified, say so.\
"""


async def run(prompt: str) -> str:
    """Analyze text snippets and return a ranked country list."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm("linguistic")
    think = get_thinking_prefix("linguistic", "reason")

    response = await llm.ainvoke([
        SystemMessage(content=f"{think} {LINGUISTIC_SYSTEM_PROMPT}"),
        HumanMessage(content=prompt),
        HumanMessage(content=f"{think} {LINGUISTIC_REASON_PROMPT}"),
    ])
    return response.content
