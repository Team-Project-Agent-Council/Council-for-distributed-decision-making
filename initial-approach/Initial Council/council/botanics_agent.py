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
        "You receive descriptions of vegetation, trees, flowers, shrubs, and other flora. "
        "Some species may be named explicitly; others may only be described visually.\n\n"
        "Use the available tools in this order:\n"
        "1. plant_search - if a species is described but not named, search iNaturalist "
        "by description to get candidate scientific names.\n"
        "2. gbif_distribution - look up country-level occurrence counts for a species.\n"
        "3. powo_distribution - look up the native range of a vascular plant (more precise "
        "than GBIF for native vs. cultivated ranges).\n\n"
        "Prefer endemic or regionally restricted species over cosmopolitan ones. "
        "A species found in only 2-3 countries is far more useful than one found everywhere.\n\n"
        "Gather evidence, then state the most likely country/region based on the botanical clues."
    ))
    return await _llm_with_tools.ainvoke([system, HumanMessage(content=prompt)])


async def reason(messages: list[BaseMessage]) -> str:
    """Given messages including tool results, produce a botanical country assessment."""
    think = get_thinking_prefix("botanics", "reason")
    response = await _llm.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} Based on all tool results above, provide a ranked list of candidate countries "
            "based solely on botanical and biodiversity evidence. "
            "Format:\n"
            "1. <Country> (confidence: <high|medium|low>) - <reason this species distribution points here>\n"
            "2. <Country> (confidence: <high|medium|low>) - <reason>\n"
            "List 2-5 candidates, most likely first."
        )),
    ])
    return response.content
