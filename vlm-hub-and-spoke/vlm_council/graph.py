"""LangGraph pipeline for the VLM Council.

Topology:
    prepare_image → [linguistic, landscape, botanics, regulatory, meta] → judge_review ↔ discussion → judge_final
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from vlm_council.state import VLMCouncilState, AgentAssessment, CandidateEntry, DiscussionEntry
from vlm_council.config import VLMCouncilConfig, load_config
from vlm_council.image_utils import encode_image
from vlm_council.agents import linguistic, landscape, botanics, regulatory, meta
from vlm_council.agents import judge as judge_agent


# Agent registry for dynamic dispatch 

AGENT_MODULES = {
    "linguistic": linguistic,
    "landscape": landscape,
    "botanics": botanics,
    "regulatory": regulatory,
    "meta": meta,
}

# VLM call timeout in seconds (600s for thinking-mode models, 300s for instruct)
VLM_CALL_TIMEOUT = int(os.environ.get("VLM_CALL_TIMEOUT", "600"))


# Helper: parse agent assessment JSON 

def _strip_think_tags(text: str) -> tuple[str, str]:
    """Separate thinking chain from the actual response.

    Returns (thinking_chain, response). If no thinking detected, returns ("", text).

    Handles three formats:
    1. Explicit <think>...</think> wrapper (a common thinking-mode convention)
    2. Gemma 4: <|channel>thought reasoning...<channel|>actual answer
    3. vLLM: reasoning...\\n</think>\\nactual answer (no opening tag)
    """
    # Format 1: explicit <think>...</think>
    think_match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), think_match.group(2).strip()

    # Format 2: Gemma 4 <|channel>thought...<channel|>
    channel_match = re.search(r"<\|channel\>thought(.*?)<channel\|>(.*)", text, re.DOTALL)
    if channel_match:
        return channel_match.group(1).strip(), channel_match.group(2).strip()

    # Format 3: </think> without opening tag (vLLM sometimes omits <think>)
    think_end = re.search(r"</think>(.*)", text, re.DOTALL)
    if think_end:
        thinking = text[:think_end.start()].strip()
        response = think_end.group(1).strip()
        return thinking, response

    return "", text


def _extract_country_result(text: str) -> tuple[str, str]:
    """Extract 'Country: X\\nCoordinates: Y\\nReasoning: Z' from text that may
    contain a long thinking chain before the actual answer.

    Returns (thinking_chain, country_result).
    """
    # First try stripping think tags
    thinking, response = _strip_think_tags(text)
    if thinking:
        return thinking, response

    # No think tags => look for "Country:" pattern in the text
    country_match = re.search(r"(Country:\s*.+?)$", text, re.DOTALL)
    if country_match:
        thinking = text[:country_match.start()].strip()
        response = country_match.group(1).strip()
        if thinking:
            return thinking, response

    return "", text


def _parse_assessment(agent_name: str, raw: str) -> AgentAssessment:
    """Parse JSON assessment from agent output, with fallback."""
    thinking, response = _strip_think_tags(raw)

    text = response.strip() if response else raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    # Find JSON objects
    matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL))
    if not matches:
        matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL))

    for match in reversed(matches):
        try:
            data = json.loads(match.group())
            raw_candidates = data.get("candidates", [])
            if not raw_candidates:
                continue

            candidates = [
                CandidateEntry(
                    country=c.get("country", "unknown"),
                    confidence=c.get("confidence", "low"),
                    reasoning=c.get("reasoning", ""),
                )
                for c in raw_candidates if isinstance(c, dict) and c.get("country")
            ]

            if candidates:
                return AgentAssessment(
                    agent_name=agent_name,
                    candidates=candidates,
                    evidence=data.get("evidence", []),
                )
        except json.JSONDecodeError:
            continue

    return AgentAssessment(
        agent_name=agent_name,
        candidates=[],
        evidence=[],
    )


def _log(msg: str):
    """Print timestamped log to stderr."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


# Node: prepare_image

async def prepare_image(state: VLMCouncilState) -> dict:
    """Encode the image once and store in state."""
    _log(f"prepare_image: encoding {state['image_path']}")
    b64, mime = encode_image(state["image_path"])
    _log(f"prepare_image: done ({len(b64) // 1024} KB)")
    return {"image_b64": b64, "image_mime": mime}


# Agent nodes

def _make_agent_node(agent_name: str):
    """Create an agent node that calls the VLM and directly produces an assessment."""
    mod = AGENT_MODULES[agent_name]

    async def node(state: VLMCouncilState) -> dict:
        _log(f"agent/{agent_name}: starting VLM call...")
        t0 = time.time()
        try:
            raw = await asyncio.wait_for(
                mod.assess(state["image_b64"], state["image_mime"]),
                timeout=VLM_CALL_TIMEOUT,
            )
            elapsed = time.time() - t0
            _log(f"agent/{agent_name}: done in {elapsed:.1f}s")
        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            _log(f"agent/{agent_name}: TIMEOUT after {elapsed:.1f}s")
            raw = '{"candidates": ["unknown"], "confidence": "speculative", "reasoning": "Agent timed out", "evidence": []}'
        except Exception as e:
            elapsed = time.time() - t0
            _log(f"agent/{agent_name}: ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")
            raw = f'{{"candidates": ["unknown"], "confidence": "speculative", "reasoning": "Error: {e}", "evidence": []}}'

        assessment = _parse_assessment(agent_name, raw)
        _log(f"agent/{agent_name}: candidates={assessment.get('candidates', [])} ({assessment.get('confidence', '?')})")
        return {f"{agent_name}_assessment": assessment}

    node.__name__ = f"{agent_name}_agent"
    return node


# Hub-and-spoke nodes 

MAX_JUDGE_RETRIES = 2


async def judge_review_node(state: VLMCouncilState) -> dict:
    """Judge reviews all assessments and decides: finalize or ask questions.

    If the judge returns invalid JSON, retry up to MAX_JUDGE_RETRIES times.
    """
    for attempt in range(MAX_JUDGE_RETRIES + 1):
        _log(f"judge/review: starting (attempt {attempt + 1})...")
        t0 = time.time()
        try:
            decision = await asyncio.wait_for(
                judge_agent.review(state),
                timeout=VLM_CALL_TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception) as e:
            _log(f"judge/review: error ({type(e).__name__}), forcing finalize")
            decision = {"action": "finalize"}
            break

        elapsed = time.time() - t0
        action = decision.get("action", "?")
        _log(f"judge/review: done in {elapsed:.1f}s → {action}")

        # Valid decision => use it
        if action in ("finalize", "questions"):
            break

        # Invalid JSON / unrecognized action => retry
        if attempt < MAX_JUDGE_RETRIES:
            _log(f"judge/review: invalid decision ({action}), retrying...")
        else:
            _log(f"judge/review: invalid decision after {MAX_JUDGE_RETRIES + 1} attempts, forcing finalize")
            decision = {"action": "finalize"}

    return {"judge_messages": [AIMessage(content=json.dumps(decision))]}


def route_judge_review(state: VLMCouncilState) -> Literal["judge_final", "judge_question"]:
    """Route based on judge's decision and max rounds.

    The judge decides, no forced discussion overrides.
    """
    config = load_config()
    current_round = state.get("discussion_round", 0)

    if current_round >= config.max_discussion_rounds:
        _log(f"route/judge_review: max rounds ({config.max_discussion_rounds}) reached → finalize")
        return "judge_final"

    messages = state.get("judge_messages", [])
    decision = None
    if messages:
        try:
            decision = json.loads(messages[-1].content)
        except (json.JSONDecodeError, AttributeError):
            pass

    if decision and decision.get("action") == "questions":
        return "judge_question"

    return "judge_final"


async def judge_question_node(state: VLMCouncilState) -> dict:
    """Extract the judge's questions and prepare them for the targeted agents."""
    messages = state.get("judge_messages", [])
    try:
        decision = json.loads(messages[-1].content)
    except (json.JSONDecodeError, AttributeError, IndexError):
        decision = {}

    current_round = state.get("discussion_round", 0) + 1

    entries = []
    if decision.get("action") == "questions" and decision.get("questions"):
        for q in decision["questions"]:
            entries.append(DiscussionEntry(
                round_number=current_round,
                judge_question=q["question"],
                target_agent=q["target_agent"],
                agent_response="",
            ))
    _log(f"judge_question: {len(entries)} question(s) to {[e['target_agent'] for e in entries]}")

    return {
        "discussion_log": entries,
        "discussion_round": current_round,
    }


async def _discuss_single_agent(state: VLMCouncilState, entry: dict) -> DiscussionEntry:
    """Run a single agent's discussion response."""
    target = entry["target_agent"]
    question = entry["judge_question"]

    mod = AGENT_MODULES.get(target)
    if mod is None:
        return DiscussionEntry(
            round_number=entry["round_number"],
            judge_question=question,
            target_agent=target,
            agent_response=f"Unknown agent: {target}",
        )

    assessment = state.get(f"{target}_assessment", {})
    previous = json.dumps(assessment, ensure_ascii=False) if assessment else "(no prior assessment)"

    _log(f"discussion/{target}: starting re-examination...")
    t0 = time.time()
    try:
        response = await asyncio.wait_for(
            mod.discuss(state["image_b64"], state["image_mime"], previous, question),
            timeout=VLM_CALL_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception) as e:
        elapsed = time.time() - t0
        _log(f"discussion/{target}: TIMEOUT/ERROR after {elapsed:.1f}s: {type(e).__name__}")
        response = f"Agent {target} timed out during discussion."
    elapsed = time.time() - t0
    _log(f"discussion/{target}: done in {elapsed:.1f}s")

    return DiscussionEntry(
        round_number=entry["round_number"],
        judge_question=question,
        target_agent=target,
        agent_response=response,
    )


async def discussion_response_node(state: VLMCouncilState) -> dict:
    """Dispatch to targeted agents for re-examination, IN PARALLEL via vLLM."""
    discussion_log = state.get("discussion_log", [])
    current_round = state.get("discussion_round", 0)

    # Find all entries from the current round that need responses
    pending = [e for e in discussion_log if e["round_number"] == current_round and not e["agent_response"]]

    if not pending:
        return {"discussion_log": []}

    if len(pending) == 1:
        updated = await _discuss_single_agent(state, pending[0])
        return {"discussion_log": [updated]}

    _log(f"discussion: dispatching {len(pending)} agents in parallel...")
    tasks = [_discuss_single_agent(state, entry) for entry in pending]
    results = await asyncio.gather(*tasks)
    _log(f"discussion: all {len(results)} agents responded")
    return {"discussion_log": list(results)}


# Judge final node

async def judge_final_node(state: VLMCouncilState) -> dict:
    """Judge makes the final country determination.

    On success, populates:
      - country_result: raw judge text ("Country: ...\\nCoordinates: ...\\nReasoning: ...")
      - coordinates:    {"lat": float, "lng": float} parsed from country_result, or None
      - final_reasoning: thinking chain if the model emitted one

    On judge failure (timeout / exception):
      - country_result: ""
      - coordinates:    None
      - error:          "<ExceptionType>: <message>"

    We do NOT emit a fake "Coordinates: 0.0, 0.0" fallback because that would
    silently poison downstream distance metrics with 15 000+ km outliers.
    """
    _log("judge/final: starting...")
    t0 = time.time()
    judge_error: str | None = None
    result = ""
    try:
        result = await asyncio.wait_for(
            judge_agent.finalize(state),
            timeout=VLM_CALL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        judge_error = f"TimeoutError: judge exceeded {VLM_CALL_TIMEOUT}s"
        _log(f"judge/final: TIMEOUT after {time.time() - t0:.1f}s")
    except Exception as e:
        judge_error = f"{type(e).__name__}: {e}"
        _log(f"judge/final: ERROR ({judge_error})")

    elapsed = time.time() - t0

    if judge_error is not None:
        return {
            "country_result": "",
            "coordinates": None,
            "final_reasoning": "",
            "error": judge_error,
        }

    thinking, response = _extract_country_result(result)
    coordinates = parse_coordinates(response)
    _log(
        f"judge/final: done in {elapsed:.1f}s "
        f"-> {response[:80]} (coords={coordinates})"
    )

    output: dict = {
        "country_result": response,
        "coordinates": coordinates,
    }
    if thinking:
        output["final_reasoning"] = thinking
    return output


def build_graph(config: VLMCouncilConfig | None = None) -> StateGraph:
    """Build the VLM Council graph with parallel agents via vLLM.

    vLLM uses continuous batching, so all 5 agents run in parallel
    on a single GPU.
    """
    if config is None:
        config = load_config()

    builder = StateGraph(VLMCouncilState)

    # Phase 0: Image preparation
    builder.add_node("prepare_image", prepare_image)

    # Phase 1: Agent assessment nodes
    for name in AGENT_MODULES:
        builder.add_node(f"{name}_agent", _make_agent_node(name))

    # Phase 2: Hub-and-spoke
    builder.add_node("judge_review", judge_review_node)
    builder.add_node("judge_question", judge_question_node)
    builder.add_node("discussion_response", discussion_response_node)
    builder.add_node("judge_final", judge_final_node)

    # Edges 

    builder.add_edge(START, "prepare_image")

    agent_names = list(AGENT_MODULES.keys())
    for name in agent_names:
        builder.add_edge("prepare_image", f"{name}_agent")
    for name in agent_names:
        builder.add_edge(f"{name}_agent", "judge_review")

    # Hub-and-spoke loop
    builder.add_conditional_edges(
        "judge_review",
        route_judge_review,
        {"judge_final": "judge_final", "judge_question": "judge_question"},
    )
    builder.add_edge("judge_question", "discussion_response")
    builder.add_edge("discussion_response", "judge_review")

    # Terminal
    builder.add_edge("judge_final", END)

    return builder.compile()
