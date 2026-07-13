from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from council.orchestrator_agent import run as run_orchestrator
from council.linguistic_agent import run as run_linguistic, reason as reason_linguistic, detect_language
from council.landscape_agent import run as run_landscape, reason as reason_landscape, identify_landscape
from council.botanics_agent import run as run_botanics, reason as reason_botanics
from council.regulatory_agent import run as run_regulatory, reason as reason_regulatory
from council.meta_agent import run as run_meta, reason as reason_meta, rag_search
from council.judge_agent import run as run_judge, reason as reason_judge, identify_country
from council.vision_agent import run as run_vision
from council.tools import wikidata_search, wikidata_sparql, geocode
from council.tools.biodiversity import plant_search, gbif_distribution, powo_distribution
from council.tools.websearch import web_search, fetch_page


class CouncilState(TypedDict):
    image_path: str
    general_description: str
    crop_descriptions: list[str]
    linguistic_prompt: str
    landscape_prompt: str
    botanics_prompt: str
    regulatory_prompt: str
    linguistic_messages: Annotated[list[BaseMessage], add_messages]
    landscape_messages: Annotated[list[BaseMessage], add_messages]
    botanics_messages: Annotated[list[BaseMessage], add_messages]
    regulatory_messages: Annotated[list[BaseMessage], add_messages]
    meta_messages: Annotated[list[BaseMessage], add_messages]
    judge_messages: Annotated[list[BaseMessage], add_messages]
    linguistic_result: str
    landscape_result: str
    botanics_result: str
    regulatory_result: str
    rag_result: str
    country_result: str


# -- vision --------------------------------------------------------------------

async def vision_node(state: CouncilState) -> dict:
    output = await run_vision(state["image_path"])
    return {"general_description": output.general_description, "crop_descriptions": output.crop_descriptions}


# -- orchestrator --------------------------------------------------------------

async def orchestrator_node(state: CouncilState) -> dict:
    output = await run_orchestrator(state["general_description"], state["crop_descriptions"])
    return {
        "linguistic_prompt": output.linguistic_prompt,
        "landscape_prompt": output.landscape_prompt,
        "botanics_prompt": output.botanics_prompt,
        "regulatory_prompt": output.regulatory_prompt,
    }


# -- linguistic branch ---------------------------------------------------------

async def linguistic_node(state: CouncilState) -> dict:
    response = await run_linguistic(state["linguistic_prompt"])
    return {"linguistic_messages": [response]}


def route_linguistic(state: CouncilState) -> str:
    last = state["linguistic_messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "linguistic_tools"
    return "linguistic_reasoning"


async def linguistic_reasoning_node(state: CouncilState) -> dict:
    result = await reason_linguistic(state["linguistic_messages"])
    return {"linguistic_result": result}


# -- landscape branch ----------------------------------------------------------

async def landscape_node(state: CouncilState) -> dict:
    response = await run_landscape(state["landscape_prompt"])
    return {"landscape_messages": [response]}


def route_landscape(state: CouncilState) -> str:
    last = state["landscape_messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "landscape_tools"
    return "landscape_reasoning"


async def landscape_reasoning_node(state: CouncilState) -> dict:
    result = await reason_landscape(state["landscape_messages"])
    return {"landscape_result": result}


# -- botanics branch -----------------------------------------------------------

async def botanics_node(state: CouncilState) -> dict:
    response = await run_botanics(state["botanics_prompt"])
    return {"botanics_messages": [response]}


def route_botanics(state: CouncilState) -> str:
    last = state["botanics_messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "botanics_tools"
    return "botanics_reasoning"


async def botanics_reasoning_node(state: CouncilState) -> dict:
    result = await reason_botanics(state["botanics_messages"])
    return {"botanics_result": result}


# -- regulatory branch ---------------------------------------------------------

async def regulatory_node(state: CouncilState) -> dict:
    response = await run_regulatory(state["regulatory_prompt"])
    return {"regulatory_messages": [response]}


def route_regulatory(state: CouncilState) -> str:
    last = state["regulatory_messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "regulatory_tools"
    return "regulatory_reasoning"


async def regulatory_reasoning_node(state: CouncilState) -> dict:
    result = await reason_regulatory(state["regulatory_messages"])
    return {"regulatory_result": result}


# -- meta branch ---------------------------------------------------------------

async def meta_node(state: CouncilState) -> dict:
    response = await run_meta(state["general_description"], state["crop_descriptions"])
    return {"meta_messages": [response]}


def route_meta(state: CouncilState) -> str:
    last = state["meta_messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "meta_tools"
    return "meta_reasoning"


async def meta_reasoning_node(state: CouncilState) -> dict:
    result = await reason_meta(state["meta_messages"])
    return {"rag_result": result}


# -- judge branch --------------------------------------------------------------

async def judge_node(state: CouncilState) -> dict:
    response = await run_judge(
        state["linguistic_result"],
        state["landscape_result"],
        state["botanics_result"],
        state["regulatory_result"],
        state["rag_result"],
    )
    return {"judge_messages": [response]}


def route_judge(state: CouncilState) -> str:
    last = state["judge_messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "judge_tools"
    return "judge_reasoning"


async def judge_reasoning_node(state: CouncilState) -> dict:
    result = await reason_judge(state["judge_messages"])
    return {"country_result": result}


# -- graph ---------------------------------------------------------------------

def build_graph():
    builder = StateGraph(CouncilState)

    builder.add_node("vision_agent", vision_node)
    builder.add_node("orchestrator_agent", orchestrator_node)

    builder.add_node("linguistic_agent", linguistic_node)
    builder.add_node("linguistic_tools", ToolNode([detect_language, wikidata_search, wikidata_sparql], messages_key="linguistic_messages"))
    builder.add_node("linguistic_reasoning", linguistic_reasoning_node)

    builder.add_node("landscape_agent", landscape_node)
    builder.add_node("landscape_tools", ToolNode([identify_landscape, wikidata_search, wikidata_sparql], messages_key="landscape_messages"))
    builder.add_node("landscape_reasoning", landscape_reasoning_node)

    builder.add_node("botanics_agent", botanics_node)
    builder.add_node("botanics_tools", ToolNode([plant_search, gbif_distribution, powo_distribution], messages_key="botanics_messages"))
    builder.add_node("botanics_reasoning", botanics_reasoning_node)

    builder.add_node("regulatory_agent", regulatory_node)
    builder.add_node("regulatory_tools", ToolNode([web_search, fetch_page], messages_key="regulatory_messages"))
    builder.add_node("regulatory_reasoning", regulatory_reasoning_node)

    builder.add_node("meta_agent", meta_node)
    builder.add_node("meta_tools", ToolNode([rag_search, wikidata_search, wikidata_sparql], messages_key="meta_messages"))
    builder.add_node("meta_reasoning", meta_reasoning_node)

    builder.add_node("judge_agent", judge_node)
    builder.add_node("judge_tools", ToolNode([identify_country, wikidata_search, wikidata_sparql, geocode], messages_key="judge_messages"))
    builder.add_node("judge_reasoning", judge_reasoning_node)

    builder.add_edge(START, "vision_agent")
    builder.add_edge("vision_agent", "orchestrator_agent")

    # fan-out: five parallel branches
    builder.add_edge("orchestrator_agent", "linguistic_agent")
    builder.add_edge("orchestrator_agent", "landscape_agent")
    builder.add_edge("orchestrator_agent", "botanics_agent")
    builder.add_edge("orchestrator_agent", "regulatory_agent")
    builder.add_edge("orchestrator_agent", "meta_agent")

    builder.add_conditional_edges("linguistic_agent", route_linguistic,
                                  {"linguistic_tools": "linguistic_tools", "linguistic_reasoning": "linguistic_reasoning"})
    builder.add_edge("linguistic_tools", "linguistic_reasoning")

    builder.add_conditional_edges("landscape_agent", route_landscape,
                                  {"landscape_tools": "landscape_tools", "landscape_reasoning": "landscape_reasoning"})
    builder.add_edge("landscape_tools", "landscape_reasoning")

    builder.add_conditional_edges("botanics_agent", route_botanics,
                                  {"botanics_tools": "botanics_tools", "botanics_reasoning": "botanics_reasoning"})
    builder.add_edge("botanics_tools", "botanics_reasoning")

    builder.add_conditional_edges("regulatory_agent", route_regulatory,
                                  {"regulatory_tools": "regulatory_tools", "regulatory_reasoning": "regulatory_reasoning"})
    builder.add_edge("regulatory_tools", "regulatory_reasoning")

    builder.add_conditional_edges("meta_agent", route_meta,
                                  {"meta_tools": "meta_tools", "meta_reasoning": "meta_reasoning"})
    builder.add_edge("meta_tools", "meta_reasoning")

    # fan-in barrier: judge fires only after all five branches complete
    builder.add_edge(
        ["linguistic_reasoning", "landscape_reasoning", "botanics_reasoning",
         "regulatory_reasoning", "meta_reasoning"],
        "judge_agent",
    )

    builder.add_conditional_edges("judge_agent", route_judge,
                                  {"judge_tools": "judge_tools", "judge_reasoning": "judge_reasoning"})
    builder.add_edge("judge_tools", "judge_reasoning")
    builder.add_edge("judge_reasoning", END)

    return builder.compile()
