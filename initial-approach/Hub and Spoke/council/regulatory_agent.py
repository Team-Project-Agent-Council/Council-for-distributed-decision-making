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


async def respond_to_followup(
    original_result: str,
    question: str,
    original_prompt: str = "",
    prior_exchanges: list[dict] | None = None,
) -> str:
    """Re-evaluate position given a confrontational follow-up from the Judge."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    llm = get_llm("regulatory")
    think = get_thinking_prefix("regulatory", "followup")

    messages = [SystemMessage(content=f"{think} {REGULATORY_SYSTEM_PROMPT}")]

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
