"""Cultural Agent - identifies countries from cultural and human context clues.

Focuses on the human layer of a scene: clothing, murals, shop types, religious
buildings, colonial architectural heritage, and other cultural indicators that
other agents (infrastructure, landscape, regulatory) miss.
"""

from __future__ import annotations

from council.llm import get_llm, get_thinking_prefix

CULTURAL_SYSTEM_PROMPT = """\
You are a Cultural Agent specialising in identifying countries from cultural
and human context clues visible in street-level images.

You focus on elements that other agents miss - the human and cultural layer of a scene
that reveals regional identity. Your domain includes:

PEOPLE & CLOTHING:
- Traditional garments: ponchos, saris, sarongs, kimonos, dashikis, ao dai, etc.
- Work clothing and uniforms: hi-vis styles, military patterns, school uniforms
- Headwear: sombreros, turbans, conical hats, baseball caps, headscarves
- These are strong regional indicators when present

ART, MURALS & DECORATION:
- Building murals: subject matter (political, religious, commercial, folk art)
- Facade painting styles: bright pastels (Caribbean/Central America), whitewash (Mediterranean),
  ornamental woodwork (colonial Latin America, South Asia, SE Asia)
- Street art and graffiti styles

COMMERCIAL & RELIGIOUS CONTEXT:
- Shop types and naming conventions (farmacia, apotek, chemist, etc.)
- Market/vendor styles: how goods are displayed, stall construction
- Religious buildings: mosque minarets, church bell towers, Buddhist stupas, Hindu gopurams
- Shrines, spirit houses, roadside crosses, prayer flags

BUILDING CULTURE:
- Colonial architectural heritage: Spanish colonial (Latin America), Portuguese (Brazil,
  Mozambique, Goa), British colonial (India, SE Asia, Africa), Dutch colonial (Indonesia,
  South Africa, Caribbean)
- Balcony styles, window treatments, door designs vary by colonial heritage
- Construction informality level: exposed rebar, water tanks on roofs, corrugated metal

Analyze the description and provide your assessment directly without using any tools.\
"""

CULTURAL_REASON_PROMPT = """\
Based on the cultural and human context clues above, provide a ranked list
of candidate countries.
Format:
1. <Country> - <what cultural evidence supports this>
2. <Country> - <reason>
List 2-5 candidates, most likely first.

Focus on the clues that are most regionally distinctive. Use high-specificity
cultural clues (specific garments, colonial architecture styles, religious
building types) over generic ones.\
"""


async def run(cultural_prompt: str) -> str:
    """Analyze cultural clues and return a ranked country list."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm("infrastructure")
    think = get_thinking_prefix("infrastructure", "reason")

    response = await llm.ainvoke([
        SystemMessage(content=f"{think} {CULTURAL_SYSTEM_PROMPT}"),
        HumanMessage(content=cultural_prompt),
        HumanMessage(content=f"{think} {CULTURAL_REASON_PROMPT}"),
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

    llm = get_llm("infrastructure")
    think = get_thinking_prefix("infrastructure", "followup")

    messages = [SystemMessage(content=f"{think} {CULTURAL_SYSTEM_PROMPT}")]

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
