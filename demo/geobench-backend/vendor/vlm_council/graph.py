"""LangGraph pipeline for the VLM Council.

Topology:
    prepare_image → [5x specialists: initial assess] → region_consensus_check
        → Path A (consensus): → country_hypotheses → [5x evaluate] → country_decision → END
        → Path B (no consensus): → region_hypotheses → [5x evaluate] → region_decision
            → [5x constrained assess] → country_hypotheses → [5x evaluate] → country_decision → END
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from typing import Literal

from langgraph.graph import END, START, StateGraph

from vlm_council.state import (
    VLMCouncilState,
    AgentAssessment,
    CandidateEntry,
    Hypothesis,
    HypothesisEvaluation,
)
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

# VLM call timeout in seconds (600s for Thinking models, 300s for Instruct) relevant for Qwen3-VL-32B-Thinking
VLM_CALL_TIMEOUT = int(os.environ.get("VLM_CALL_TIMEOUT", "600"))


# Helper: parse agent assessment JSON 

def _strip_think_tags(text: str) -> tuple[str, str]:
    """Separate thinking chain from the actual response.

    Returns (thinking_chain, response). If no thinking detected, returns ("", text).

    Handles three formats:
    1. Qwen3: <think>reasoning...</think>actual answer
    2. Gemma 4: <|channel>thought reasoning...<channel|>actual answer
    3. vLLM: reasoning...\\n</think>\\nactual answer (no opening tag)
    """
    # Format 1: Qwen3 explicit <think>...</think>
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

    return AgentAssessment(agent_name=agent_name, candidates=[], evidence=[])


def _parse_evaluations(agent_name: str, raw: str) -> list[HypothesisEvaluation]:
    """Parse JSON array of hypothesis evaluations from agent output."""
    _, response = _strip_think_tags(raw)
    text = response.strip() if response else raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    # Find JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group())
        if not isinstance(data, list):
            return []
        evals = []
        for item in data:
            if isinstance(item, dict) and item.get("hypothesis_id") and item.get("confidence"):
                evals.append(HypothesisEvaluation(
                    agent_name=agent_name,
                    hypothesis_id=item["hypothesis_id"],
                    confidence=item["confidence"],
                    reasoning=item.get("reasoning", ""),
                    key_evidence=item.get("key_evidence", []),
                ))
        return evals
    except json.JSONDecodeError:
        return []


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
    return {"image_b64": b64, "image_mime": mime, "current_phase": "initial"}


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
            _log(f"agent/{agent_name}: TIMEOUT after {time.time() - t0:.1f}s")
            raw = '{"candidates": [], "evidence": []}'
        except Exception as e:
            _log(f"agent/{agent_name}: ERROR: {type(e).__name__}: {e}")
            raw = '{"candidates": [], "evidence": []}'

        assessment = _parse_assessment(agent_name, raw)
        _log(f"agent/{agent_name}: {len(assessment.get('candidates', []))} candidates")
        return {f"{agent_name}_assessment": assessment}

    node.__name__ = f"{agent_name}_agent"
    return node


# Hub-and-spoke nodes 

async def region_consensus_check(state: VLMCouncilState) -> dict:
    """Judge checks if all agents agree on a region."""
    _log("judge/consensus: checking region consensus...")
    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            judge_agent.check_region_consensus(state),
            timeout=VLM_CALL_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception) as e:
        _log(f"judge/consensus: ERROR ({type(e).__name__}), assuming no consensus")
        result = {"consensus": False, "consensus_region": None, "proposed_regions": [], "region_candidates": {}}

    elapsed = time.time() - t0
    consensus = result.get("consensus", False)
    region = result.get("consensus_region")
    proposed = result.get("proposed_regions", [])
    region_candidates = result.get("region_candidates", {})

    _log(f"judge/consensus: {'CONSENSUS' if consensus else 'NO CONSENSUS'} "
         f"(region={region}, proposed={proposed}) in {elapsed:.1f}s")

    output = {
        "region_consensus": consensus,
        "proposed_regions": proposed,
        "region_candidates": region_candidates,
        "current_phase": "region" if not consensus else "country",
    }
    if consensus and region:
        output["confirmed_region"] = region

    return output


def route_after_consensus(state: VLMCouncilState) -> Literal["country_hypotheses", "region_hypotheses"]:
    """Route based on region consensus."""
    if state.get("region_consensus", False):
        return "country_hypotheses"
    return "region_hypotheses"


#  Region Hypotheses Node

async def region_hypotheses_node(state: VLMCouncilState) -> dict:
    """Create Hypothesis objects from proposed_regions."""
    config = load_config()
    proposed = state.get("proposed_regions", [])[:config.max_region_hypotheses]

    hypotheses = []
    for region in proposed:
        hyp_id = "region_" + region.lower().replace(" ", "_").replace("&", "and")
        hypotheses.append(Hypothesis(
            hypothesis_id=hyp_id,
            level="region",
            value=region,
            statement=f"This image is from {region}",
        ))

    _log(f"region_hypotheses: created {len(hypotheses)} hypotheses: {[h['value'] for h in hypotheses]}")
    return {"active_hypotheses": hypotheses}


#  Region Evaluate Node

async def region_evaluate_node(state: VLMCouncilState) -> dict:
    """All 5 specialists evaluate region hypotheses in parallel (clean-slate).

    Each agent receives an independently shuffled copy of the hypothesis list
    so that any positional bias from the LLM (primacy/recency effects when
    listing N hypotheses) does NOT correlate across the 5 specialists. This
    is the "isolated assessment" part of Progressive Narrowing — without the
    shuffle, all 5 agents see the hypotheses in the same order, so any
    position-dependent bias would compound rather than wash out.
    """
    _log("region_evaluate: starting parallel evaluation...")
    hypotheses = state.get("active_hypotheses", [])
    if not hypotheses:
        return {"hypothesis_evaluations": []}

    async def _eval_agent(name: str) -> list[HypothesisEvaluation]:
        mod = AGENT_MODULES[name]
        # Per-agent independent permutation of the hypotheses. The original
        # `hypotheses` list is left untouched for downstream consumers.
        shuffled = list(hypotheses)
        random.shuffle(shuffled)
        t0 = time.time()
        try:
            raw = await asyncio.wait_for(
                mod.evaluate_hypotheses(
                    state["image_b64"], state["image_mime"], shuffled
                ),
                timeout=VLM_CALL_TIMEOUT,
            )
            _log(f"region_evaluate/{name}: done in {time.time() - t0:.1f}s")
            return _parse_evaluations(name, raw)
        except (asyncio.TimeoutError, Exception) as e:
            _log(f"region_evaluate/{name}: ERROR ({type(e).__name__})")
            return []

    tasks = [_eval_agent(name) for name in AGENT_MODULES]
    results = await asyncio.gather(*tasks)

    all_evals = []
    for evals in results:
        all_evals.extend(evals)

    _log(f"region_evaluate: got {len(all_evals)} evaluations total")
    return {"hypothesis_evaluations": all_evals}


#  Region Decision Node

async def region_decision_node(state: VLMCouncilState) -> dict:
    """Judge decides the region based on evaluations."""
    _log("judge/region_decision: deciding region...")
    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            judge_agent.decide_region(state),
            timeout=VLM_CALL_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception) as e:
        _log(f"judge/region_decision: ERROR ({type(e).__name__})")
        proposed = state.get("proposed_regions", [])
        result = {"decided_region": proposed[0] if proposed else "Europe", "reasoning": "Fallback"}

    decided = result.get("decided_region", "Unknown")
    reasoning = result.get("reasoning", "")
    elapsed = time.time() - t0

    # If judge returned a hypothesis_id instead of a region name, resolve it
    hypotheses = state.get("active_hypotheses", [])
    hyp_lookup = {h["hypothesis_id"]: h["value"] for h in hypotheses}
    if decided in hyp_lookup:
        decided = hyp_lookup[decided]

    _log(f"judge/region_decision: decided '{decided}' in {elapsed:.1f}s")

    return {
        "confirmed_region": decided,
        "region_decision_reasoning": reasoning,
        "current_phase": "country",
    }


#  Country Assess Node (Path B only)

async def country_assess_node(state: VLMCouncilState) -> dict:
    """5 specialists do a new assessment constrained to the confirmed region (Path B)."""
    region = state.get("confirmed_region", "Unknown")
    _log(f"country_assess: constrained assessment for region '{region}'...")

    async def _assess_agent(name: str) -> tuple[str, AgentAssessment]:
        mod = AGENT_MODULES[name]
        t0 = time.time()
        try:
            raw = await asyncio.wait_for(
                mod.assess_constrained(state["image_b64"], state["image_mime"], region),
                timeout=VLM_CALL_TIMEOUT,
            )
            _log(f"country_assess/{name}: done in {time.time() - t0:.1f}s")
            return name, _parse_assessment(name, raw)
        except (asyncio.TimeoutError, Exception) as e:
            _log(f"country_assess/{name}: ERROR ({type(e).__name__})")
            return name, AgentAssessment(agent_name=name, candidates=[], evidence=[])

    tasks = [_assess_agent(name) for name in AGENT_MODULES]
    results = await asyncio.gather(*tasks)

    output = {}
    for name, assessment in results:
        output[f"{name}_country_assessment"] = assessment
        _log(f"country_assess/{name}: {len(assessment.get('candidates', []))} candidates")

    return output


#  Country Hypotheses Node ──────────────────────────────────────────────────

async def country_hypotheses_node(state: VLMCouncilState) -> dict:
    """Create country hypotheses from assessments.

    Path A (consensus): Extract from initial assessments, filter to confirmed region via LLM
    Path B (no consensus): Extract from constrained assessments (already region-filtered)

    In both paths: also include candidates that the Judge assigned to the confirmed region
    (via region_candidates) and that were proposed by 3+ agents initially even if they
    didn't make it into the constrained assessments or Top-N filter.
    """
    config = load_config()
    confirmed_region = state.get("confirmed_region", "")

    # Determine which assessments to use
    if state.get("region_consensus", False):
        # Path A: use initial assessments
        prefix = ""
    else:
        # Path B: use constrained assessments (already filtered by region)
        prefix = "country_"

    # Collect all country candidates ranked by frequency
    from collections import Counter
    country_counts: Counter = Counter()

    for name in AGENT_MODULES:
        key = f"{name}_{prefix}assessment" if prefix else f"{name}_assessment"
        assessment = state.get(key, {})
        for c in assessment.get("candidates", []):
            country = c.get("country", "").strip()
            if country:
                country_counts[country] += 1

    all_countries = [c for c, _ in country_counts.most_common()]

    # Filter to confirmed region via LLM (Path A needs this; Path B is already constrained)
    if state.get("region_consensus", False) and confirmed_region and all_countries:
        _log(f"country_hypotheses: filtering {len(all_countries)} candidates to region '{confirmed_region}'...")
        filtered = await asyncio.wait_for(
            judge_agent.filter_countries_to_region(all_countries, confirmed_region),
            timeout=VLM_CALL_TIMEOUT,
        )
        _log(f"country_hypotheses: {len(filtered)} countries after region filter")
        top_countries = filtered[:config.max_country_hypotheses]
    else:
        top_countries = all_countries[:config.max_country_hypotheses]

    region_candidates = state.get("region_candidates", {})
    confirmed_region_candidates = region_candidates.get(confirmed_region, {})
    additions = []
    if isinstance(confirmed_region_candidates, dict):
        for country, count in confirmed_region_candidates.items():
            if isinstance(count, int) and count >= 3 and country not in top_countries:
                additions.append(country)

    if additions:
        _log(f"country_hypotheses: adding {len(additions)} strong candidates (3+ agents) from confirmed region: {additions}")
        top_countries.extend(additions)

    hypotheses = []
    for country in top_countries:
        hyp_id = "country_" + country.lower().replace(" ", "_")
        hypotheses.append(Hypothesis(
            hypothesis_id=hyp_id,
            level="country",
            value=country,
            statement=f"This image is from {country}",
        ))

    _log(f"country_hypotheses: {len(hypotheses)} hypotheses: {[h['value'] for h in hypotheses]}")
    return {"active_hypotheses": hypotheses}


# Country Evaluate Node

async def country_evaluate_node(state: VLMCouncilState) -> dict:
    """All 5 specialists evaluate country hypotheses in parallel (clean-slate).

    As in region_evaluate_node, each agent receives an independently shuffled
    copy of the hypothesis list to decorrelate positional bias across the
    five specialists.
    """
    _log("country_evaluate: starting parallel evaluation...")
    hypotheses = state.get("active_hypotheses", [])
    if not hypotheses:
        return {"hypothesis_evaluations": []}

    async def _eval_agent(name: str) -> list[HypothesisEvaluation]:
        mod = AGENT_MODULES[name]
        shuffled = list(hypotheses)
        random.shuffle(shuffled)
        t0 = time.time()
        try:
            raw = await asyncio.wait_for(
                mod.evaluate_hypotheses(
                    state["image_b64"], state["image_mime"], shuffled
                ),
                timeout=VLM_CALL_TIMEOUT,
            )
            _log(f"country_evaluate/{name}: done in {time.time() - t0:.1f}s")
            return _parse_evaluations(name, raw)
        except (asyncio.TimeoutError, Exception) as e:
            _log(f"country_evaluate/{name}: ERROR ({type(e).__name__})")
            return []

    tasks = [_eval_agent(name) for name in AGENT_MODULES]
    results = await asyncio.gather(*tasks)

    all_evals = []
    for evals in results:
        all_evals.extend(evals)

    _log(f"country_evaluate: got {len(all_evals)} evaluations total")
    return {"hypothesis_evaluations": all_evals}


#  Country Decision Node

async def country_decision_node(state: VLMCouncilState) -> dict:
    """Judge makes the final country determination."""
    _log("judge/country_decision: making final determination...")
    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            judge_agent.decide_country(state),
            timeout=VLM_CALL_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception) as e:
        _log(f"judge/country_decision: ERROR ({type(e).__name__})")
        result = "Country: Unknown\nCoordinates: 0.0, 0.0\nReasoning: Judge error."

    thinking, response = _extract_country_result(result)
    elapsed = time.time() - t0
    _log(f"judge/country_decision: done in {elapsed:.1f}s → {response[:80]}")

    output = {"country_result": response}
    if thinking:
        output["final_reasoning"] = thinking
    return output


# Build Graph

def build_graph(config: VLMCouncilConfig | None = None) -> StateGraph:
    """Build the VLM Council graph with Progressive Narrowing architecture."""
    if config is None:
        config = load_config()

    builder = StateGraph(VLMCouncilState)

    # Phase 0: Image preparation
    builder.add_node("prepare_image", prepare_image)

    # Phase 1: Initial assessment (parallel via vLLM continuous batching)
    for name in AGENT_MODULES:
        builder.add_node(f"{name}_agent", _make_agent_node(name))

    # Region consensus
    builder.add_node("region_consensus_check", region_consensus_check)

    # Region phase (Path B only)
    builder.add_node("region_hypotheses", region_hypotheses_node)
    builder.add_node("region_evaluate", region_evaluate_node)
    builder.add_node("region_decision", region_decision_node)

    # Country assessment with constraint (Path B only)
    builder.add_node("country_assess", country_assess_node)

    # Country phase (both paths)
    builder.add_node("country_hypotheses", country_hypotheses_node)
    builder.add_node("country_evaluate", country_evaluate_node)
    builder.add_node("country_decision", country_decision_node)

    # Edges

    # START => prepare_image => [5x agents] => region_consensus_check
    builder.add_edge(START, "prepare_image")
    for name in AGENT_MODULES:
        builder.add_edge("prepare_image", f"{name}_agent")
    for name in AGENT_MODULES:
        builder.add_edge(f"{name}_agent", "region_consensus_check")

    # Conditional: consensus or not
    builder.add_conditional_edges(
        "region_consensus_check",
        route_after_consensus,
        {
            "country_hypotheses": "country_hypotheses",
            "region_hypotheses": "region_hypotheses",
        },
    )

    # Region phase (Path B)
    builder.add_edge("region_hypotheses", "region_evaluate")
    builder.add_edge("region_evaluate", "region_decision")
    builder.add_edge("region_decision", "country_assess")
    builder.add_edge("country_assess", "country_hypotheses")

    # Country phase (both paths converge here)
    builder.add_edge("country_hypotheses", "country_evaluate")
    builder.add_edge("country_evaluate", "country_decision")

    # Terminal
    builder.add_edge("country_decision", END)

    return builder.compile()
