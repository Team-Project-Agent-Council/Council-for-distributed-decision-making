"""Evaluation-only graph: no vision node.

general_description and crop_descriptions must be pre-populated in the
initial state by the caller (from result.json files).

This graph mirrors council/graph.py exactly (same agents, same prompts)
but skips the vision pipeline - it starts at the orchestrator.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from council.orchestrator_agent import run as run_orchestrator
from council.linguistic_agent import run as run_linguistic
from council.landscape_agent import run as run_landscape
from council.infrastructure_agent import run as run_infrastructure
from council.regulatory_agent import run as run_regulatory
from council.botanics_agent import run as run_botanics, reason as reason_botanics
from council.cultural_agent import run as run_cultural
from council.judge_agent import run as run_judge, reason as reason_judge, identify_country
from council.tools import wikidata_search, wikidata_sparql, geocode
from council.tools.biodiversity import plant_search, gbif_distribution, powo_distribution


class EvalState(TypedDict):
    image_path: str
    general_description: str
    crop_descriptions: list[str]
    linguistic_prompt: str
    landscape_prompt: str
    botanics_prompt: str
    regulatory_prompt: str
    infrastructure_prompt: str
    cultural_prompt: str
    botanics_messages: Annotated[list[BaseMessage], add_messages]
    judge_messages: Annotated[list[BaseMessage], add_messages]
    linguistic_result: str
    landscape_result: str
    botanics_result: str
    regulatory_result: str
    infrastructure_result: str
    cultural_result: str
    rag_result: str
    country_result: str


# -- orchestrator --------------------------------------------------------------

async def orchestrator_node(state: EvalState) -> dict:
    output = await run_orchestrator(state["general_description"], state["crop_descriptions"])
    return {
        "linguistic_prompt": output.linguistic_prompt,
        "landscape_prompt": output.landscape_prompt,
        "botanics_prompt": output.botanics_prompt,
        "regulatory_prompt": output.regulatory_prompt,
        "infrastructure_prompt": output.infrastructure_prompt,
        "cultural_prompt": output.cultural_prompt,
    }


# -- agent nodes (import from council/*_agent.py) -----------------------------

async def linguistic_node(state: EvalState) -> dict:
    return {"linguistic_result": await run_linguistic(state["linguistic_prompt"])}

async def landscape_node(state: EvalState) -> dict:
    return {"landscape_result": await run_landscape(state["landscape_prompt"])}

async def infrastructure_node(state: EvalState) -> dict:
    return {"infrastructure_result": await run_infrastructure(state["infrastructure_prompt"])}

async def regulatory_node(state: EvalState) -> dict:
    return {"regulatory_result": await run_regulatory(state["regulatory_prompt"])}

async def cultural_node(state: EvalState) -> dict:
    prompt = state.get("cultural_prompt", "")
    if not prompt or "no cultural clues" in prompt.lower():
        return {"cultural_result": "No cultural clues visible in this scene."}
    return {"cultural_result": await run_cultural(prompt)}

async def meta_node(_state: EvalState) -> dict:
    return {"rag_result": "RAG/meta agent skipped (requires Ollama for embeddings)."}

async def botanics_node(state: EvalState) -> dict:
    response = await run_botanics(state["botanics_prompt"])
    return {"botanics_messages": [response]}

async def botanics_reasoning_node(state: EvalState) -> dict:
    return {"botanics_result": await reason_botanics(state["botanics_messages"])}

async def botanics_continue_node(state: EvalState) -> dict:
    from council.botanics_agent import continue_with_tools
    response = await continue_with_tools(state["botanics_messages"])
    return {"botanics_messages": [response]}

async def judge_node(state: EvalState) -> dict:
    response = await run_judge(
        state["linguistic_result"],
        state["landscape_result"],
        state["botanics_result"],
        state["regulatory_result"],
        state["rag_result"],
        state["infrastructure_result"],
        state["cultural_result"],
    )
    return {"judge_messages": [response]}

async def judge_reasoning_node(state: EvalState) -> dict:
    return {"country_result": await reason_judge(state["judge_messages"])}


# -- routing functions ---------------------------------------------------------

def _route_by_tool_calls(state: dict, messages_key: str, tools_target: str, reasoning_target: str) -> str:
    last = state[messages_key][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return tools_target
    return reasoning_target

def route_botanics(state: EvalState) -> str:
    return _route_by_tool_calls(state, "botanics_messages", "botanics_tools", "botanics_reasoning")

def route_botanics_continue(state: EvalState) -> str:
    return _route_by_tool_calls(state, "botanics_messages", "botanics_tools_2", "botanics_reasoning")

def route_judge(state: EvalState) -> str:
    return _route_by_tool_calls(state, "judge_messages", "judge_tools", "judge_reasoning")


# -- graph builder -------------------------------------------------------------

def build_eval_graph():
    builder = StateGraph(EvalState)

    # No vision - start at orchestrator
    builder.add_node("orchestrator_agent", orchestrator_node)

    # Direct agents (prompt-only - no ToolNode)
    builder.add_node("linguistic_agent", linguistic_node)
    builder.add_node("landscape_agent", landscape_node)
    builder.add_node("infrastructure_agent", infrastructure_node)
    builder.add_node("regulatory_agent", regulatory_node)
    builder.add_node("cultural_agent", cultural_node)
    builder.add_node("meta_agent", meta_node)

    # Tool-using agents (only botanics + judge have tool nodes)
    builder.add_node("botanics_agent", botanics_node)
    builder.add_node("botanics_tools", ToolNode([plant_search, gbif_distribution, powo_distribution], messages_key="botanics_messages"))
    builder.add_node("botanics_continue", botanics_continue_node)
    builder.add_node("botanics_tools_2", ToolNode([gbif_distribution, powo_distribution], messages_key="botanics_messages"))
    builder.add_node("botanics_reasoning", botanics_reasoning_node)

    builder.add_node("judge_agent", judge_node)
    builder.add_node("judge_tools", ToolNode([identify_country, wikidata_search, wikidata_sparql, geocode], messages_key="judge_messages"))
    builder.add_node("judge_reasoning", judge_reasoning_node)

    builder.add_edge(START, "orchestrator_agent")

    # fan-out: seven parallel branches
    builder.add_edge("orchestrator_agent", "linguistic_agent")
    builder.add_edge("orchestrator_agent", "landscape_agent")
    builder.add_edge("orchestrator_agent", "botanics_agent")
    builder.add_edge("orchestrator_agent", "regulatory_agent")
    builder.add_edge("orchestrator_agent", "infrastructure_agent")
    builder.add_edge("orchestrator_agent", "cultural_agent")
    builder.add_edge("orchestrator_agent", "meta_agent")

    # botanics tool loop
    builder.add_conditional_edges("botanics_agent", route_botanics,
                                  {"botanics_tools": "botanics_tools", "botanics_reasoning": "botanics_reasoning"})
    builder.add_edge("botanics_tools", "botanics_continue")
    builder.add_conditional_edges("botanics_continue", route_botanics_continue,
                                  {"botanics_tools_2": "botanics_tools_2", "botanics_reasoning": "botanics_reasoning"})
    builder.add_edge("botanics_tools_2", "botanics_reasoning")

    # fan-in: judge after all branches complete
    builder.add_edge(
        ["linguistic_agent", "landscape_agent", "botanics_reasoning",
         "regulatory_agent", "infrastructure_agent", "cultural_agent",
         "meta_agent"],
        "judge_agent",
    )

    # judge tool loop
    builder.add_conditional_edges("judge_agent", route_judge,
                                  {"judge_tools": "judge_tools", "judge_reasoning": "judge_reasoning"})
    builder.add_edge("judge_tools", "judge_reasoning")
    builder.add_edge("judge_reasoning", END)

    return builder.compile()
