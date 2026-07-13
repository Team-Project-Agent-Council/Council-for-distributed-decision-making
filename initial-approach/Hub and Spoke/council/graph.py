from __future__ import annotations

import asyncio
import operator
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from ollama import Client as OllamaClient
from typing_extensions import TypedDict

from council.orchestrator_agent import run as run_orchestrator
from council.linguistic_agent import run as run_linguistic
from council.landscape_agent import run as run_landscape
from council.infrastructure_agent import run as run_infrastructure
from council.regulatory_agent import run as run_regulatory
from council.botanics_agent import run as run_botanics, reason as reason_botanics
from council.cultural_agent import run as run_cultural
from council.judge_agent import run as run_judge, reason as reason_judge, identify_country
from council.judge_agent import deliberate as judge_deliberate
from council.linguistic_agent import respond_to_followup as followup_linguistic
from council.landscape_agent import respond_to_followup as followup_landscape
from council.botanics_agent import respond_to_followup as followup_botanics
from council.regulatory_agent import respond_to_followup as followup_regulatory
from council.infrastructure_agent import respond_to_followup as followup_infrastructure
from council.cultural_agent import respond_to_followup as followup_cultural
from council.tools import wikidata_search, wikidata_sparql, geocode
from council.tools.biodiversity import plant_search, gbif_distribution, powo_distribution

# Vision pipeline imports (shared root package)
from vision_pipeline.scene_parser import scene_parser
from vision_pipeline.detail_identifier import detail_identifier
from vision_pipeline.detail_extractor import detail_extractor
from vision_pipeline.crop_tool import crop_tool
from vision_pipeline.detail_focusser import detail_focusser
from vision_pipeline.state import Detail
from vision_pipeline.config import PipelineConfig, load_config as load_vision_config


# -- state --------------------------------------------------------------------

class CouncilState(TypedDict):
    image_path: str
    clean_image_path: str | None

    # -- vision pipeline fields --------------------------------------------
    scene_description: str
    detected_objects: list[dict]
    details: list[Detail]
    has_details: bool
    vision_errors: Annotated[list[str], operator.add]

    # -- orchestrator fields -----------------------------------------------
    general_description: str
    crop_descriptions: list[str]
    linguistic_prompt: str
    landscape_prompt: str
    botanics_prompt: str
    regulatory_prompt: str
    infrastructure_prompt: str
    cultural_prompt: str

    # -- agent branches ----------------------------------------------------
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

    # -- deliberation loop fields -----------------------------------------
    deliberation_round: int
    followup_questions: list[dict]
    deliberation_history: Annotated[list[str], operator.add]
    judge_satisfied: bool
    agent_followup_history: dict[str, list[dict]]


# -- vision pipeline nodes ----------------------------------------------------

_vision_config: PipelineConfig | None = None
_ollama_client: OllamaClient | None = None


def _get_vision_config() -> PipelineConfig:
    global _vision_config
    if _vision_config is None:
        _vision_config = load_vision_config()
    return _vision_config


def _get_ollama_client() -> OllamaClient:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient(host=_get_vision_config().ollama_host)
    return _ollama_client


def scene_parser_node(state: CouncilState) -> dict:
    config = _get_vision_config()
    client = _get_ollama_client()
    result = scene_parser(state, client, config)
    if "errors" in result:
        result["vision_errors"] = result.pop("errors")
    return result


def detail_identifier_node(state: CouncilState) -> dict:
    config = _get_vision_config()
    client = _get_ollama_client()
    result = detail_identifier(state, client, config)
    if "errors" in result:
        result["vision_errors"] = result.pop("errors")
    return result


def detail_extractor_node(state: CouncilState) -> dict:
    config = _get_vision_config()
    result = detail_extractor(state, config)
    if "errors" in result:
        result["vision_errors"] = result.pop("errors")
    return result


def crop_tool_node(state: CouncilState) -> dict:
    config = _get_vision_config()
    result = crop_tool(state, config)
    if "errors" in result:
        result["vision_errors"] = result.pop("errors")
    return result


def detail_focusser_node(state: CouncilState) -> dict:
    config = _get_vision_config()
    client = _get_ollama_client()
    result = detail_focusser(state, client, config)
    if "errors" in result:
        result["vision_errors"] = result.pop("errors")
    return result


def vision_mapping_node(state: CouncilState) -> dict:
    general_description = state.get("scene_description", "")
    crop_descriptions = []
    for detail in state.get("details", []):
        desc = detail.get("focused_description")
        if desc:
            crop_descriptions.append(f"{detail['name']}: {desc}")
    return {
        "general_description": general_description,
        "crop_descriptions": crop_descriptions,
    }


def route_after_identifier(state: CouncilState) -> str:
    if state.get("has_details", False) and len(state.get("details", [])) > 0:
        return "detail_extractor"
    return "vision_mapping"


# -- orchestrator --------------------------------------------------------------

async def orchestrator_node(state: CouncilState) -> dict:
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

async def linguistic_node(state: CouncilState) -> dict:
    return {"linguistic_result": await run_linguistic(state["linguistic_prompt"])}

async def landscape_node(state: CouncilState) -> dict:
    return {"landscape_result": await run_landscape(state["landscape_prompt"])}

async def infrastructure_node(state: CouncilState) -> dict:
    return {"infrastructure_result": await run_infrastructure(state["infrastructure_prompt"])}

async def regulatory_node(state: CouncilState) -> dict:
    return {"regulatory_result": await run_regulatory(state["regulatory_prompt"])}

async def cultural_node(state: CouncilState) -> dict:
    prompt = state.get("cultural_prompt", "")
    if not prompt or "no cultural clues" in prompt.lower():
        return {"cultural_result": "No cultural clues visible in this scene."}
    return {"cultural_result": await run_cultural(prompt)}

async def meta_node(_state: CouncilState) -> dict:
    return {"rag_result": "RAG/meta agent skipped (requires Ollama for embeddings)."}

async def botanics_node(state: CouncilState) -> dict:
    response = await run_botanics(state["botanics_prompt"])
    return {"botanics_messages": [response]}

async def botanics_reasoning_node(state: CouncilState) -> dict:
    return {"botanics_result": await reason_botanics(state["botanics_messages"])}

async def botanics_continue_node(state: CouncilState) -> dict:
    from council.botanics_agent import continue_with_tools
    response = await continue_with_tools(state["botanics_messages"])
    return {"botanics_messages": [response]}

async def judge_node(state: CouncilState) -> dict:
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

async def judge_reasoning_node(state: CouncilState) -> dict:
    return {"country_result": await reason_judge(state["judge_messages"])}


# -- deliberation hub-and-spoke -----------------------------------------------

MAX_DELIBERATION_ROUNDS = 3

FOLLOWUP_DISPATCHERS = {
    "linguistic": followup_linguistic,
    "landscape": followup_landscape,
    "botanics": followup_botanics,
    "regulatory": followup_regulatory,
    "infrastructure": followup_infrastructure,
    "cultural": followup_cultural,
}

AGENT_RESULT_KEYS = {
    "linguistic": "linguistic_result",
    "landscape": "landscape_result",
    "botanics": "botanics_result",
    "regulatory": "regulatory_result",
    "infrastructure": "infrastructure_result",
    "cultural": "cultural_result",
}

AGENT_PROMPT_KEYS = {
    "linguistic": "linguistic_prompt",
    "landscape": "landscape_prompt",
    "botanics": "botanics_prompt",
    "regulatory": "regulatory_prompt",
    "infrastructure": "infrastructure_prompt",
    "cultural": "cultural_prompt",
}


async def judge_deliberation_node(state: CouncilState) -> Command:
    """Hub node: Judge reviews evidence, decides to finalize or send follow-ups."""
    round_num = state.get("deliberation_round", 0)

    if round_num >= MAX_DELIBERATION_ROUNDS:
        return Command(
            update={
                "judge_satisfied": True,
                "deliberation_round": round_num,
                "deliberation_history": [f"Round {round_num}: Forced finalization (max rounds reached)."],
            },
            goto="judge_agent",
        )

    decision = await judge_deliberate(
        linguistic_result=state.get("linguistic_result", ""),
        landscape_result=state.get("landscape_result", ""),
        botanics_result=state.get("botanics_result", ""),
        regulatory_result=state.get("regulatory_result", ""),
        rag_result=state.get("rag_result", ""),
        infrastructure_result=state.get("infrastructure_result", ""),
        cultural_result=state.get("cultural_result", ""),
        deliberation_round=round_num,
        deliberation_history=state.get("deliberation_history", []),
    )

    history_entry = f"Round {round_num}: {decision.reasoning}"

    if decision.satisfied or not decision.follow_ups:
        return Command(
            update={
                "judge_satisfied": True,
                "deliberation_round": round_num,
                "deliberation_history": [history_entry],
            },
            goto="judge_agent",
        )

    valid_follow_ups = [fu for fu in decision.follow_ups if fu.agent in FOLLOWUP_DISPATCHERS]
    if not valid_follow_ups:
        return Command(
            update={
                "judge_satisfied": True,
                "deliberation_round": round_num,
                "deliberation_history": [history_entry + " [No valid agents targeted, finalizing]"],
            },
            goto="judge_agent",
        )

    agents_queried = ", ".join(fu.agent for fu in valid_follow_ups)
    history_entry += f" [Follow-ups to: {agents_queried}]"

    followup_questions = [
        {"agent": fu.agent, "question": fu.question}
        for fu in valid_follow_ups
    ]

    return Command(
        update={
            "judge_satisfied": False,
            "deliberation_round": round_num + 1,
            "followup_questions": followup_questions,
            "deliberation_history": [history_entry],
        },
        goto="followup_dispatch",
    )


async def followup_dispatch_node(state: CouncilState) -> Command:
    """Execute all pending follow-ups in parallel, update results, loop back to Judge."""
    questions = state.get("followup_questions", [])
    if not questions:
        return Command(goto="judge_deliberation")

    history = state.get("agent_followup_history", {})

    async def _run_followup(fq: dict) -> tuple[str, str]:
        agent = fq["agent"]
        question = fq["question"]
        result_key = AGENT_RESULT_KEYS[agent]
        prompt_key = AGENT_PROMPT_KEYS[agent]
        original_result = state.get(result_key, "")
        original_prompt = state.get(prompt_key, "")
        prior_exchanges = history.get(agent, [])
        dispatch_fn = FOLLOWUP_DISPATCHERS[agent]
        new_result = await dispatch_fn(original_result, question, original_prompt, prior_exchanges)
        return agent, new_result

    results = await asyncio.gather(*[_run_followup(fq) for fq in questions])

    update = {}
    new_history = dict(history)
    for fq in questions:
        agent = fq["agent"]
        answer = next(r for a, r in results if a == agent)
        update[AGENT_RESULT_KEYS[agent]] = answer
        agent_hist = list(new_history.get(agent, []))
        agent_hist.append({"question": fq["question"], "answer": answer})
        new_history[agent] = agent_hist

    update["agent_followup_history"] = new_history

    return Command(
        update=update,
        goto="judge_deliberation",
    )


# -- routing functions ---------------------------------------------------------

def _route_by_tool_calls(state: dict, messages_key: str, tools_target: str, reasoning_target: str) -> str:
    last = state[messages_key][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return tools_target
    return reasoning_target

def route_botanics(state: CouncilState) -> str:
    return _route_by_tool_calls(state, "botanics_messages", "botanics_tools", "botanics_reasoning")

def route_botanics_continue(state: CouncilState) -> str:
    return _route_by_tool_calls(state, "botanics_messages", "botanics_tools_2", "botanics_reasoning")

def route_judge(state: CouncilState) -> str:
    return _route_by_tool_calls(state, "judge_messages", "judge_tools", "judge_reasoning")


# -- graph builder -------------------------------------------------------------

def build_graph():
    builder = StateGraph(CouncilState)

    # Vision pipeline
    builder.add_node("scene_parser", scene_parser_node)
    builder.add_node("detail_identifier", detail_identifier_node)
    builder.add_node("detail_extractor", detail_extractor_node)
    builder.add_node("crop_tool", crop_tool_node)
    builder.add_node("detail_focusser", detail_focusser_node)
    builder.add_node("vision_mapping", vision_mapping_node)

    # Orchestrator
    builder.add_node("orchestrator_agent", orchestrator_node)

    # Direct agents
    builder.add_node("linguistic_agent", linguistic_node)
    builder.add_node("landscape_agent", landscape_node)
    builder.add_node("infrastructure_agent", infrastructure_node)
    builder.add_node("regulatory_agent", regulatory_node)
    builder.add_node("cultural_agent", cultural_node)
    builder.add_node("meta_agent", meta_node)

    # Tool-using agents
    builder.add_node("botanics_agent", botanics_node)
    builder.add_node("botanics_tools", ToolNode([plant_search, gbif_distribution, powo_distribution], messages_key="botanics_messages"))
    builder.add_node("botanics_continue", botanics_continue_node)
    builder.add_node("botanics_tools_2", ToolNode([gbif_distribution, powo_distribution], messages_key="botanics_messages"))
    builder.add_node("botanics_reasoning", botanics_reasoning_node)

    # Deliberation hub
    builder.add_node("judge_deliberation", judge_deliberation_node)
    builder.add_node("followup_dispatch", followup_dispatch_node)

    # Judge final determination
    builder.add_node("judge_agent", judge_node)
    builder.add_node("judge_tools", ToolNode([identify_country, wikidata_search, wikidata_sparql, geocode], messages_key="judge_messages"))
    builder.add_node("judge_reasoning", judge_reasoning_node)

    # -- Vision pipeline edges ---------------------------------------------
    builder.add_edge(START, "scene_parser")
    builder.add_edge("scene_parser", "detail_identifier")
    builder.add_conditional_edges(
        "detail_identifier",
        route_after_identifier,
        {"detail_extractor": "detail_extractor", "vision_mapping": "vision_mapping"},
    )
    builder.add_edge("detail_extractor", "crop_tool")
    builder.add_edge("crop_tool", "detail_focusser")
    builder.add_edge("detail_focusser", "vision_mapping")
    builder.add_edge("vision_mapping", "orchestrator_agent")

    # -- fan-out: seven parallel branches ----------------------------------
    builder.add_edge("orchestrator_agent", "linguistic_agent")
    builder.add_edge("orchestrator_agent", "landscape_agent")
    builder.add_edge("orchestrator_agent", "botanics_agent")
    builder.add_edge("orchestrator_agent", "regulatory_agent")
    builder.add_edge("orchestrator_agent", "infrastructure_agent")
    builder.add_edge("orchestrator_agent", "cultural_agent")
    builder.add_edge("orchestrator_agent", "meta_agent")

    # -- botanics tool loop ------------------------------------------------
    builder.add_conditional_edges("botanics_agent", route_botanics,
                                  {"botanics_tools": "botanics_tools", "botanics_reasoning": "botanics_reasoning"})
    builder.add_edge("botanics_tools", "botanics_continue")
    builder.add_conditional_edges("botanics_continue", route_botanics_continue,
                                  {"botanics_tools_2": "botanics_tools_2", "botanics_reasoning": "botanics_reasoning"})
    builder.add_edge("botanics_tools_2", "botanics_reasoning")

    # -- fan-in: judge deliberation after all branches complete ---------
    builder.add_edge(
        ["linguistic_agent", "landscape_agent", "botanics_reasoning",
         "regulatory_agent", "infrastructure_agent", "cultural_agent",
         "meta_agent"],
        "judge_deliberation",
    )

    # -- judge tool loop ---------------------------------------------------
    builder.add_conditional_edges("judge_agent", route_judge,
                                  {"judge_tools": "judge_tools", "judge_reasoning": "judge_reasoning"})
    builder.add_edge("judge_tools", "judge_reasoning")
    builder.add_edge("judge_reasoning", END)

    return builder.compile()
