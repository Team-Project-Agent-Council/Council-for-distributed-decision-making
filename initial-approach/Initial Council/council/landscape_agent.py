from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from council.llm import get_llm, get_thinking_prefix
from council.tools import wikidata_search, wikidata_sparql

_llm = get_llm("landscape")


@tool
async def identify_landscape(description: str) -> str:
    """Identify the landscape type or geographic region from a textual description."""
    think = get_thinking_prefix("landscape", "tool")
    response = await _llm.ainvoke([
        SystemMessage(content=(
            f"{think} You are a landscape and geography expert. Given a description of terrain, "
            "vegetation, climate, and environment, identify the landscape type or geographic region "
            "in one concise sentence."
        )),
        HumanMessage(content=description),
    ])
    return response.content


_llm_with_tools = _llm.bind_tools([identify_landscape, wikidata_search, wikidata_sparql])


async def run(prompt: str) -> AIMessage:
    """Call LLM - returns AIMessage with tool_calls for ToolNode to execute."""
    think = get_thinking_prefix("landscape", "run")
    system = SystemMessage(content=(
        f"{think} You are a Landscape Agent specializing in identifying countries from geographic clues. "
        "You receive descriptions of terrain, vegetation, climate, and infrastructure visible in a street-level image.\n\n"
        "Use the available tools to:\n"
        "1. identify_landscape - classify the landscape type or geographic region from a description.\n"
        "2. wikidata_search - resolve a geographic term or property name to a Wikidata ID. "
        "Use kind='item' for biomes/continents/regions, kind='property' for predicates.\n"
        "3. wikidata_sparql - query Wikidata for countries using the IDs from wikidata_search.\n\n"
        "Workflow: identify_landscape -> wikidata_search (resolve IDs) -> wikidata_sparql (filter countries).\n"
        "Gather evidence, then state the most likely country/region based on the geographic clues."
    ))
    return await _llm_with_tools.ainvoke([system, HumanMessage(content=prompt)])


async def reason(messages: list[BaseMessage]) -> str:
    """Given messages including tool results, produce a geographic country assessment."""
    think = get_thinking_prefix("landscape", "reason")
    response = await _llm.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} Based on all tool results above, provide a ranked list of candidate countries "
            "based solely on geographic and landscape evidence. "
            "Format:\n"
            "1. <Country> (confidence: <high|medium|low>) - <reason this terrain/geography points here>\n"
            "2. <Country> (confidence: <high|medium|low>) - <reason>\n"
            "List 2-5 candidates, most likely first."
        )),
    ])
    return response.content
