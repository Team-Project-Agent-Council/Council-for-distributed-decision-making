from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from council.llm import get_llm, get_thinking_prefix
from council.tools import wikidata_search, wikidata_sparql, geocode

_llm = get_llm("judge")


@tool
async def identify_country(linguistic: str, landscape: str, botanics: str, regulatory: str, rag_meta: str, infrastructure: str, climate: str) -> str:
    """Name the single most likely country given findings from all seven specialist agents."""
    think = get_thinking_prefix("judge", "tool")
    response = await _llm.ainvoke([
        SystemMessage(content=(
            f"{think} You are a geography expert making a final country determination. "
            "Given evidence from seven specialist agents, name the single most likely country. "
            "Respond with only the country name. Example: 'Austria'"
        )),
        HumanMessage(content=(
            f"Linguistic evidence: {linguistic}\n"
            f"Landscape evidence: {landscape}\n"
            f"Botanical evidence: {botanics}\n"
            f"Regulatory evidence: {regulatory}\n"
            f"Infrastructure evidence: {infrastructure}\n"
            f"Climate evidence: {climate}\n"
            f"RAG/meta candidates:\n{rag_meta}"
        )),
    ])
    return response.content.strip()


_llm_with_tools = _llm.bind_tools([identify_country, wikidata_search, wikidata_sparql, geocode])


async def run(
    linguistic_result: str,
    landscape_result: str,
    botanics_result: str,
    regulatory_result: str,
    rag_result: str,
    infrastructure_result: str,
    climate_result: str,
) -> AIMessage:
    """Call LLM - returns AIMessage with tool_calls for ToolNode to execute."""
    think = get_thinking_prefix("judge", "run")
    system = SystemMessage(content=(
        f"{think} You are a Judge Agent making the final country determination for a GeoGuessr image.\n\n"
        "You receive findings from seven specialist agents:\n"
        "- Linguistic Agent: language/script clues and their geographic implications\n"
        "- Landscape Agent: terrain, vegetation, and geographic clues\n"
        "- Botanics Agent: plant species distributions from GBIF/POWO\n"
        "- Regulatory Agent: road design, signs, markings, and infrastructure standards\n"
        "- Infrastructure Agent: vehicles, road surface, and building architecture\n"
        "- Climate Agent: climate zone, weather, and environmental conditions\n"
        "- Meta Agent: RAG knowledge base candidates\n\n"
        "Use the available tools to:\n"
        "1. wikidata_search - resolve any entity or property name to a Wikidata ID.\n"
        "2. wikidata_sparql - optionally verify or resolve conflicts between agents using structured facts.\n"
        "3. geocode - get GPS coordinates for the most specific location you can determine. "
        "Do NOT geocode just the country name. Use regional clues to form a specific query, "
        "e.g. 'Chiang Mai, Thailand', 'Bavaria, Germany', 'Cappadocia, Turkey'. "
        "The more specific, the better the score.\n"
        "4. identify_country - produce the final country determination.\n\n"
        "Always call geocode with a specific region or city (not just the country), "
        "then identify_country as your final action."
    ))
    human = HumanMessage(content=(
        f"Linguistic finding: {linguistic_result}\n"
        f"Landscape finding: {landscape_result}\n"
        f"Botanical finding: {botanics_result}\n"
        f"Regulatory finding: {regulatory_result}\n"
        f"Infrastructure finding: {infrastructure_result}\n"
        f"Climate finding: {climate_result}\n"
        f"RAG/meta candidates:\n{rag_result}"
    ))
    return await _llm_with_tools.ainvoke([system, human])


async def reason(messages: list[BaseMessage]) -> str:
    """Given messages including tool results, produce the final country answer."""
    think = get_thinking_prefix("judge", "reason")
    response = await _llm.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} Based on all evidence and tool results above, state the single most likely country, "
            "GPS coordinates, and a brief explanation. "
            "Format: 'Country: <name>\nCoordinates: <lat>, <lon>\nReasoning: <2-3 sentences>'"
        )),
    ])
    return response.content
