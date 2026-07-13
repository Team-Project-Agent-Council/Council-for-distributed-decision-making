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


async def respond_to_followup(
    original_result: str,
    question: str,
    original_prompt: str = "",
    prior_exchanges: list[dict] | None = None,
) -> str:
    """Re-evaluate position given a confrontational follow-up from the Judge."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    llm = get_llm("linguistic")
    think = get_thinking_prefix("linguistic", "followup")

    messages = [SystemMessage(content=f"{think} {LINGUISTIC_SYSTEM_PROMPT}")]

    if original_prompt:
        messages.append(HumanMessage(content=original_prompt))
    messages.append(AIMessage(content=original_result))

    for exchange in (prior_exchanges or []):
        messages.append(HumanMessage(content=f"JUDGE'S QUESTION: {exchange['question']}"))
        messages.append(AIMessage(content=exchange["answer"]))

    messages.append(HumanMessage(content=(
        f"{think} The Judge Agent challenges your assessment with the following question. "
        "You MUST take a clear position. If the new evidence changes your assessment, "
        "provide an UPDATED ranked country list. If it does not, explain precisely why "
        "your original assessment stands. Do not hedge or equivocate.\n\n"
        f"JUDGE'S QUESTION: {question}\n\n"
        "Provide your updated ranked list of candidate countries in the same format as before."
    )))

    response = await llm.ainvoke(messages)
    return response.content
