from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from council.llm import get_llm, get_thinking_prefix

_llm = get_llm("infrastructure")


@tool
async def vehicle_analysis(description: str) -> str:
    """Analyse vehicles (cars, motorcycles, trucks, tuk-tuks, rickshaws, etc.) visible in the
    image description to infer the geographic region.
    Consider: vehicle types, their relative frequency (e.g. many motorbikes -> Southeast Asia),
    vehicle brands, license plate styles, and driving side."""
    think = get_thinking_prefix("infrastructure", "tool")
    response = await _llm.ainvoke([
        SystemMessage(content=(
            f"{think} You are a vehicle and transportation expert for geographic identification. "
            "Given a description of vehicles in an image, analyse what types and brands of vehicles "
            "are visible, their relative frequency (e.g. many motorcycles vs. few), driving side, "
            "and any visible license plate styles. "
            "Based on this, list the most likely countries or regions. "
            "Consider: Southeast Asia often has many motorcycles and tuk-tuks; Germany is known for "
            "German car brands; North America has large trucks and pickups; Japan has kei cars; "
            "India has auto-rickshaws. "
            "Respond with a ranked list of candidate regions and your reasoning."
        )),
        HumanMessage(content=description),
    ])
    return response.content.strip()


@tool
async def street_analysis(description: str) -> str:
    """Analyse road surface quality, road type, lane markings, and street infrastructure
    to infer the geographic region.
    Consider: road quality (paved/unpaved, potholes), lane width, center line color
    (yellow=North America, white=Europe/Asia), sidewalk presence, curb types, and
    special road types like the German Autobahn."""
    think = get_thinking_prefix("infrastructure", "tool")
    response = await _llm.ainvoke([
        SystemMessage(content=(
            f"{think} You are a road infrastructure expert for geographic identification. "
            "Given a description of roads and street infrastructure, analyse: road surface quality "
            "(paved, unpaved, potholed), road width and number of lanes, center line color "
            "(yellow lines = North America; white lines = Europe/Asia/Australia), sidewalk presence "
            "and quality, curb stones, barriers and guardrails, road markings, special road types "
            "(e.g. Autobahn in Germany, Route Nationale in France, unpaved tracks in rural Africa). "
            "Based on this, list the most likely countries or regions. "
            "Respond with a ranked list of candidate regions and your reasoning."
        )),
        HumanMessage(content=description),
    ])
    return response.content.strip()


@tool
async def architecture_analysis(description: str) -> str:
    """Analyse building styles, construction materials, and architectural features to infer
    the geographic region.
    Consider: roof styles (flat roofs -> Mediterranean/Middle East/arid regions; steep roofs ->
    Northern Europe/Canada), building materials (timber framing -> Germany/UK; concrete blocks ->
    developing world; wooden houses -> Scandinavia/Russia/North America), cultural architectural
    elements (pagodas -> East/Southeast Asia; minarets -> Muslim world; colonial style -> former
    colonies), and typical local colors or decoration."""
    think = get_thinking_prefix("infrastructure", "tool")
    response = await _llm.ainvoke([
        SystemMessage(content=(
            f"{think} You are an architectural expert for geographic identification. "
            "Given a description of buildings and structures, analyse: architectural style, "
            "construction materials (timber, brick, concrete, stone, wood), roof shapes "
            "(flat, sloped, tiled, thatched), building colors and decoration, window styles, "
            "fences and walls, and any culturally distinctive features "
            "(pagodas, minarets, colonial facades, Soviet-era apartment blocks, Nordic wooden houses, "
            "Mediterranean whitewashed walls, etc.). "
            "Based on this, list the most likely countries or regions. "
            "Respond with a ranked list of candidate regions and your reasoning."
        )),
        HumanMessage(content=description),
    ])
    return response.content.strip()


_llm_with_tools = _llm.bind_tools([vehicle_analysis, street_analysis, architecture_analysis])


async def run(prompt: str) -> AIMessage:
    """Call LLM - returns AIMessage with tool_calls for ToolNode to execute."""
    think = get_thinking_prefix("infrastructure", "run")
    system = SystemMessage(content=(
        f"{think} You are an Infrastructure Agent specialising in identifying countries from "
        "vehicles, road infrastructure, and building architecture visible in street-level images.\n\n"
        "You receive descriptions of the scene including vehicles on the road, road surface and "
        "markings, and building architecture.\n\n"
        "Use the available tools:\n"
        "1. vehicle_analysis - analyse vehicle types, brands, and frequency to narrow down the region.\n"
        "2. street_analysis - analyse road quality, lane markings, and street infrastructure.\n"
        "3. architecture_analysis - analyse building styles, materials, and architectural features.\n\n"
        "Call all three tools to gather comprehensive evidence, then reason about the most likely "
        "country based on the combined findings."
    ))
    return await _llm_with_tools.ainvoke([system, HumanMessage(content=prompt)])


async def reason(messages: list[BaseMessage]) -> str:
    """Given messages including tool results, produce an infrastructure country assessment."""
    think = get_thinking_prefix("infrastructure", "reason")
    response = await _llm.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} Based on all tool results above (vehicle analysis, street analysis, and "
            "architecture analysis), provide a ranked list of candidate countries "
            "based solely on infrastructure and architectural evidence. "
            "Format:\n"
            "1. <Country> (confidence: <high|medium|low>) - <reason this infrastructure clue points here>\n"
            "2. <Country> (confidence: <high|medium|low>) - <reason>\n"
            "List 2-5 candidates, most likely first."
        )),
    ])
    return response.content
