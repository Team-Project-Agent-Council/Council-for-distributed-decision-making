from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from council.llm import get_llm, get_thinking_prefix
from council.tools.biodiversity import plant_search, gbif_distribution, powo_distribution

_llm = get_llm("botanics")

_llm_with_tools = _llm.bind_tools([plant_search, gbif_distribution, powo_distribution])


async def run(prompt: str) -> AIMessage:
    """Call LLM - returns AIMessage with tool_calls for ToolNode to execute."""
    think = get_thinking_prefix("botanics", "run")
    system = SystemMessage(content=(
        f"{think} You are a Botanics Agent specializing in identifying countries from "
        "plant species and vegetation visible in street-level images.\n\n"
        "You receive descriptions of vegetation. Your job is to identify species and look up "
        "their geographic distributions.\n\n"
        "IMPORTANT - how plant_search works:\n"
        "plant_search queries iNaturalist by NAME, not by description. It works with:\n"
        "  GOOD: 'Musa', 'Cyathea', 'fern', 'palm tree', 'Monstera', 'bamboo'\n"
        "  BAD: 'large tropical leaf plant', 'climbing vine with heart-shaped leaves'\n\n"
        "So your workflow is:\n"
        "1. Use your botanical knowledge to identify a likely genus or common name from "
        "the visual description (e.g. large paddle-shaped leaves -> banana -> Musa)\n"
        "2. Call plant_search with that NAME to get the exact species\n"
        "3. Call gbif_distribution with the species name for country distribution\n\n"
        "Call all tools in a single response. If you can already name the genus confidently, "
        "skip plant_search and go straight to gbif_distribution + powo_distribution."
    ))
    return await _llm_with_tools.ainvoke([system, HumanMessage(content=prompt)])


async def continue_with_tools(messages: list[BaseMessage]) -> AIMessage:
    """Second round: see plant_search results, call gbif/powo with real names."""
    think = get_thinking_prefix("botanics", "run")
    return await _llm_with_tools.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} You now have plant_search results with real scientific names. "
            "Pick the ONE most geographically distinctive species from the results and call:\n"
            "- gbif_distribution with the exact scientific name (e.g. 'Cyathea contaminans')\n"
            "- powo_distribution with the exact scientific name\n\n"
            "Use a SPECIES-level name (two words like 'Musa acuminata'), not just a genus. "
            "Call both tools in a single response."
        )),
    ])


async def reason(messages: list[BaseMessage]) -> str:
    """Given messages including tool results, produce a botanical country assessment."""
    think = get_thinking_prefix("botanics", "reason")
    response = await _llm.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} Based on all tool results above, provide a ranked list of candidate countries "
            "based solely on botanical and biodiversity evidence.\n"
            "Format:\n"
            "1. <Country> - <what species distribution data supports this>\n"
            "2. <Country> - <reason>\n"
            "List 2-5 candidates, most likely first.\n\n"
            "State clearly what evidence you have (e.g. 'GBIF shows 368 occurrences in MY') "
            "and what you don't (e.g. 'no species could be identified'). "
            "Note that GBIF/POWO databases don't have complete coverage for all countries - "
            "occurrence counts are indicative, not definitive. A species may be common in a country "
            "even if GBIF shows few records there. "
            "If you could not identify any species, say so - do not guess."
        )),
    ])
    return response.content
