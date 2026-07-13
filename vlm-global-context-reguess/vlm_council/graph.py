"""LangGraph pipeline for the VLM Council, Global Context Re-guess.

Topology:
    prepare_image → [5 agents Round 1 parallel] → [5 agents Round 2 parallel with global context] → judge_final
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

from langgraph.graph import END, START, StateGraph

from vlm_council.state import VLMCouncilState, AgentAssessment, CandidateEntry
from vlm_council.config import VLMCouncilConfig, load_config
from vlm_council.coordinates import parse_coordinates
from vlm_council.image_utils import encode_image
from vlm_council.agents import linguistic, landscape, botanics, regulatory, meta
from vlm_council.agents import judge as judge_agent


AGENT_MODULES = {
    "linguistic": linguistic,
    "landscape": landscape,
    "botanics": botanics,
    "regulatory": regulatory,
    "meta": meta,
}

VLM_CALL_TIMEOUT = int(os.environ.get("VLM_CALL_TIMEOUT", "600"))


def _strip_think_tags(text: str) -> tuple[str, str]:
    """Separate thinking chain from the actual response."""
    # Gemma-4 format: <|think|>...<|/think|>
    gemma_match = re.search(r"<\|think\|>(.*?)<\|/think\|>(.*)", text, re.DOTALL)
    if gemma_match:
        return gemma_match.group(1).strip(), gemma_match.group(2).strip()

    # Standard format: <think>...</think>
    think_match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), think_match.group(2).strip()

    channel_match = re.search(r"<\|channel\>thought(.*?)<channel\|>(.*)", text, re.DOTALL)
    if channel_match:
        return channel_match.group(1).strip(), channel_match.group(2).strip()

    # Partial think end (response started mid-stream)
    think_end = re.search(r"</think>(.*)", text, re.DOTALL)
    if think_end:
        thinking = text[:think_end.start()].strip()
        response = think_end.group(1).strip()
        return thinking, response

    # Gemma-4 partial end
    gemma_end = re.search(r"<\|/think\|>(.*)", text, re.DOTALL)
    if gemma_end:
        thinking = text[:gemma_end.start()].strip()
        response = gemma_end.group(1).strip()
        return thinking, response

    return "", text


def _extract_country_result(text: str) -> tuple[str, str]:
    """Extract 'Country: X' from text that may contain a thinking chain."""
    thinking, response = _strip_think_tags(text)
    if thinking:
        return thinking, response

    country_match = re.search(r"(Country:\s*.+?)$", text, re.DOTALL)
    if country_match:
        thinking = text[:country_match.start()].strip()
        response = country_match.group(1).strip()
        if thinking:
            return thinking, response

    return "", text


def _parse_assessment(agent_name: str, raw: str) -> AgentAssessment:
    """Parse JSON assessment from agent output, with fallback."""
    _, response = _strip_think_tags(raw)

    text = response.strip() if response else raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    # Try parsing the whole text as JSON first
    for candidate_text in [text, raw]:
        try:
            data = json.loads(candidate_text.strip())
            if isinstance(data, dict) and data.get("candidates"):
                candidates = [
                    CandidateEntry(
                        country=c.get("country", "unknown"),
                        confidence=c.get("confidence", "low"),
                        reasoning=c.get("reasoning", ""),
                    )
                    for c in data["candidates"] if isinstance(c, dict) and c.get("country")
                ]
                if candidates:
                    return AgentAssessment(
                        agent_name=agent_name,
                        candidates=candidates,
                        evidence=data.get("evidence", []),
                    )
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: find JSON objects in text with regex
    for search_text in [text, raw]:
        matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", search_text, re.DOTALL))
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


# === Node: prepare_image ===

async def prepare_image(state: VLMCouncilState) -> dict:
    """Encode the image once and store in state."""
    _log(f"prepare_image: encoding {state['image_path']}")
    b64, mime = encode_image(state["image_path"])
    _log(f"prepare_image: done ({len(b64) // 1024} KB)")
    return {"image_b64": b64, "image_mime": mime}


# === Round 1: Independent assessment nodes ===

def _make_round1_node(agent_name: str):
    """Create a Round 1 agent node that calls the VLM independently."""
    mod = AGENT_MODULES[agent_name]

    async def node(state: VLMCouncilState) -> dict:
        _log(f"round1/{agent_name}: starting VLM call...")
        t0 = time.time()
        try:
            raw = await asyncio.wait_for(
                mod.assess(state["image_b64"], state["image_mime"]),
                timeout=VLM_CALL_TIMEOUT,
            )
            elapsed = time.time() - t0
            _log(f"round1/{agent_name}: done in {elapsed:.1f}s")
        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            _log(f"round1/{agent_name}: TIMEOUT after {elapsed:.1f}s")
            raw = '{"candidates": [], "evidence": []}'
        except Exception as e:
            elapsed = time.time() - t0
            _log(f"round1/{agent_name}: ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")
            raw = f'{{"candidates": [], "evidence": []}}'

        assessment = _parse_assessment(agent_name, raw)
        _log(f"round1/{agent_name}: {len(assessment.get('candidates', []))} candidates")
        return {f"round_1_{agent_name}": assessment}

    node.__name__ = f"round1_{agent_name}"
    return node


# === Round 2: Re-guess with global context ===

def _format_all_round1(state: VLMCouncilState) -> str:
    """Format all Round 1 assessments as text for Round 2 context."""
    parts = []
    for name in AGENT_MODULES:
        assessment = state.get(f"round_1_{name}", {})
        candidates = assessment.get("candidates", [])
        evidence = assessment.get("evidence", [])

        if not candidates:
            parts.append(f"[{name.upper()} AGENT]\n  (insufficient evidence)")
            continue

        cand_lines = []
        for c in candidates:
            country = c.get("country", "?")
            conf = c.get("confidence", "?")
            reasoning = c.get("reasoning", "")
            cand_lines.append(f"  - {country} ({conf}): {reasoning}")

        evidence_str = ", ".join(str(e) for e in evidence) if evidence else "(none)"
        parts.append(
            f"[{name.upper()} AGENT]\n"
            + "\n".join(cand_lines) + "\n"
            f"  Evidence: {evidence_str}"
        )
    return "\n\n".join(parts)


def _make_round2_node(agent_name: str):
    """Create a Round 2 agent node that re-guesses with full Round 1 context."""
    mod = AGENT_MODULES[agent_name]

    async def node(state: VLMCouncilState) -> dict:
        _log(f"round2/{agent_name}: starting re-guess with global context...")
        t0 = time.time()

        own_round1 = json.dumps(state.get(f"round_1_{agent_name}", {}), ensure_ascii=False)
        all_round1 = _format_all_round1(state)

        error_info = ""
        try:
            raw = await asyncio.wait_for(
                mod.re_guess(state["image_b64"], state["image_mime"], own_round1, all_round1),
                timeout=VLM_CALL_TIMEOUT,
            )
            elapsed = time.time() - t0
            _log(f"round2/{agent_name}: done in {elapsed:.1f}s")
        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            error_info = f"TIMEOUT after {elapsed:.1f}s"
            _log(f"round2/{agent_name}: {error_info}")
            raw = '{"candidates": [], "evidence": []}'
        except Exception as e:
            elapsed = time.time() - t0
            error_info = f"{type(e).__name__}: {e}"
            _log(f"round2/{agent_name}: ERROR after {elapsed:.1f}s: {error_info}")
            raw = '{"candidates": [], "evidence": []}'

        assessment = _parse_assessment(agent_name, raw)
        assessment["raw_output"] = raw
        if error_info:
            assessment["error"] = error_info
        n_cands = len(assessment.get('candidates', []))
        _log(f"round2/{agent_name}: {n_cands} candidates parsed")
        if n_cands == 0 and not error_info:
            _log(f"round2/{agent_name}: RAW (first 300): {raw[:300]}")
        return {f"round_2_{agent_name}": assessment}

    node.__name__ = f"round2_{agent_name}"
    return node


# === Judge final node ===

async def judge_final_node(state: VLMCouncilState) -> dict:
    """Judge makes the final country determination based on ALL rounds.

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
    _log("judge/final: starting (with thinking enabled)...")
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
    _log(f"judge/final: done in {elapsed:.1f}s -> {response[:80]} (coords={coordinates})")

    output: dict = {
        "country_result": response,
        "coordinates": coordinates,
    }
    if thinking:
        output["final_reasoning"] = thinking
    return output


# === Barrier node between Round 1 and Round 2 ===

async def round1_complete(state: VLMCouncilState) -> dict:
    """No-op barrier node to synchronize all Round 1 agents before Round 2."""
    _log("round1_complete: all agents finished Round 1, starting Round 2...")
    return {}


# === Build the graph ===

def build_graph(config: VLMCouncilConfig | None = None) -> StateGraph:
    """Build the VLM Council graph with Global Context Re-guess topology.

    Topology:
        prepare_image
            → [5 agents Round 1 in parallel]
            → round1_complete (barrier)
            → [5 agents Round 2 in parallel with global context]
            → judge_final
    """
    if config is None:
        config = load_config()

    builder = StateGraph(VLMCouncilState)

    # Nodes
    builder.add_node("prepare_image", prepare_image)

    for name in AGENT_MODULES:
        builder.add_node(f"round1_{name}", _make_round1_node(name))

    builder.add_node("round1_complete", round1_complete)

    for name in AGENT_MODULES:
        builder.add_node(f"round2_{name}", _make_round2_node(name))

    builder.add_node("judge_final", judge_final_node)

    # Edges: START → prepare_image → [Round 1 agents]
    builder.add_edge(START, "prepare_image")

    agent_names = list(AGENT_MODULES.keys())
    for name in agent_names:
        builder.add_edge("prepare_image", f"round1_{name}")

    # Round 1 agents → barrier
    for name in agent_names:
        builder.add_edge(f"round1_{name}", "round1_complete")

    # Barrier → [Round 2 agents]
    for name in agent_names:
        builder.add_edge("round1_complete", f"round2_{name}")

    # Round 2 agents → judge_final
    for name in agent_names:
        builder.add_edge(f"round2_{name}", "judge_final")

    # Terminal
    builder.add_edge("judge_final", END)

    return builder.compile()
