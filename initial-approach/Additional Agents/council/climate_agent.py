from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from council.llm import get_llm, get_thinking_prefix

_llm = get_llm("climate")


@tool
async def climate_analysis(description: str) -> str:
    """Identify the climate zone and season from the image description and infer the
    geographic region.
    Consider: vegetation density and type (tropical rainforest, savanna, desert, temperate
    forest, boreal/taiga, tundra), weather conditions (snow, rain, sunshine, fog, dust),
    temperature indicators (frost, dry cracked soil, lush green), light quality and sun angle,
    seasonal cues (bare trees -> winter in temperate zone; dry brown grass -> dry season in
    tropics/subtropics), and humidity indicators."""
    think = get_thinking_prefix("climate", "tool")
    response = await _llm.ainvoke([
        SystemMessage(content=(
            f"{think} You are a climatology and biogeography expert for geographic identification. "
            "Given a description of a scene, identify the climate zone using the Köppen climate "
            "classification or a simplified equivalent: "
            "Tropical (Af/Am/Aw) - hot and humid or with dry season; "
            "Arid (BWh/BWk/BSh/BSk) - desert or semi-arid; "
            "Mediterranean (Csa/Csb) - hot/warm dry summer, mild wet winter; "
            "Humid subtropical (Cfa) - hot summer, mild winter, rain year-round; "
            "Oceanic (Cfb) - mild temperatures, frequent rain; "
            "Continental (Dfa/Dfb/Dfc) - cold winters, warm/cool summers; "
            "Subarctic/Tundra (Dfc/ET) - very cold, short growing season. "
            "Also note seasonal cues, vegetation type, snow/ice presence, soil moisture, "
            "and light/sun angle as evidence. "
            "Based on this climate profile, list the most likely countries or broad regions. "
            "Respond with the identified climate zone, supporting evidence from the description, "
            "and a ranked list of candidate regions."
        )),
        HumanMessage(content=description),
    ])
    return response.content.strip()


_llm_with_tools = _llm.bind_tools([climate_analysis])


async def run(prompt: str) -> AIMessage:
    """Call LLM - returns AIMessage with tool_calls for ToolNode to execute."""
    think = get_thinking_prefix("climate", "run")
    system = SystemMessage(content=(
        f"{think} You are a Climate Agent specialising in identifying countries and regions "
        "from climate, weather, and environmental conditions visible in street-level images.\n\n"
        "You receive descriptions of the scene including vegetation, weather, terrain moisture, "
        "seasonal indicators, and light conditions.\n\n"
        "Use the available tool:\n"
        "1. climate_analysis - identify the climate zone from the description and infer the region.\n\n"
        "Use the climate zone, vegetation biome, and seasonal cues to narrow down the most likely "
        "country or region."
    ))
    return await _llm_with_tools.ainvoke([system, HumanMessage(content=prompt)])


async def reason(messages: list[BaseMessage]) -> str:
    """Given messages including tool results, produce a climate-based country assessment."""
    think = get_thinking_prefix("climate", "reason")
    response = await _llm.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} Based on all tool results above (climate and environmental analysis), "
            "provide a ranked list of candidate countries "
            "based solely on climate, weather, and environmental evidence. "
            "Format:\n"
            "1. <Country> (confidence: <high|medium|low>) - <reason this climate clue points here>\n"
            "2. <Country> (confidence: <high|medium|low>) - <reason>\n"
            "List 2-5 candidates, most likely first."
        )),
    ])
    return response.content
