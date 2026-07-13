from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from council.llm import get_llm, get_thinking_prefix
from council.tools.websearch import web_search, fetch_page

_llm = get_llm("regulatory")

_llm_with_tools = _llm.bind_tools([web_search, fetch_page])


async def run(prompt: str) -> AIMessage:
    """Call LLM - returns AIMessage with tool_calls for ToolNode to execute."""
    think = get_thinking_prefix("regulatory", "run")
    system = SystemMessage(content=(
        f"{think} You are a Regulatory Agent specializing in identifying countries from "
        "infrastructure, road design, and regulatory visual clues in street-level images.\n\n"
        "You receive descriptions of roads, signs, markings, utility infrastructure, "
        "vehicles, and other man-made elements visible in the image.\n\n"
        "Use the available tools in order:\n"
        "1. web_search - search for country-specific regulations and standards.\n"
        "2. fetch_page - if a search result looks highly relevant (e.g. a GeoGuessr "
        "guide, a Wikipedia article on road signs by country, a road markings reference), "
        "fetch its full content for detailed information.\n\n"
        "- Driving side (left vs right) - visible from road markings and vehicle positions\n"
        "- Road sign shapes and colors - red-bordered circles (Europe), rectangles (USA/Canada)\n"
        "- Center line color - yellow (North America) vs white (Europe/Asia)\n"
        "- License plate format - EU blue strip, country codes, shape\n"
        "- Utility poles - wooden (North America) vs concrete/steel (Europe/Asia)\n"
        "- Traffic light position - overhead gantry vs side post\n"
        "- Street furniture - bollard styles, guardrail types, road surface\n\n"
        "Search for the most specific and distinctive clues first. "
        "Gather evidence, then state the most likely country/region based on regulatory clues."
    ))
    return await _llm_with_tools.ainvoke([system, HumanMessage(content=prompt)])


async def reason(messages: list[BaseMessage]) -> str:
    """Given messages including web search results, produce a regulatory country assessment."""
    think = get_thinking_prefix("regulatory", "reason")
    response = await _llm.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} Based on all web search results above, provide a ranked list of candidate countries "
            "based solely on regulatory and infrastructure evidence. "
            "Format:\n"
            "1. <Country> (confidence: <high|medium|low>) - <reason this infrastructure clue points here>\n"
            "2. <Country> (confidence: <high|medium|low>) - <reason>\n"
            "List 2-5 candidates, most likely first."
        )),
    ])
    return response.content
