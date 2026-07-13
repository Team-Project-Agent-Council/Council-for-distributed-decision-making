"""LangGraph pipeline for the VLM Council (Tournament Only).

This is a single-path pipeline that ends in a head-to-head tournament bracket.

Topology:
    prepare_image → [5x specialists: initial assess]
    → country_hypotheses (candidate pool from the initial assessments)
    → country_evaluate (5x specialists, clean-slate, see RAG refs)
    → tournament (4 → 2 → 1 bracket, multimodal Judge with RAG refs)
    → END
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

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
from vlm_council.tournament import tournament_node


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

            candidates_raw = [
                (c.get("country", "").strip(), c.get("confidence", "low"), c.get("reasoning", ""))
                for c in raw_candidates if isinstance(c, dict) and c.get("country")
            ]

            from vlm_council.regions import canonical_country_name

            seen: set[str] = set()
            candidates: list[CandidateEntry] = []
            for country, confidence, reasoning in candidates_raw:
                canon = canonical_country_name(country) or country
                if canon in seen:
                    continue
                seen.add(canon)
                candidates.append(CandidateEntry(
                    country=canon,
                    confidence=confidence,
                    reasoning=reasoning,
                ))

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

    # Find JSON array, scan for `[` and let json.raw_decode find its
    # balanced end. Avoids catastrophic backtracking on malformed output.
    decoder = json.JSONDecoder()
    data = None
    for source in (text, raw):
        idx = source.find("[")
        while idx != -1:
            try:
                obj, _end = decoder.raw_decode(source, idx)
                if isinstance(obj, list):
                    data = obj
                    break
            except json.JSONDecodeError:
                pass
            idx = source.find("[", idx + 1)
        if data is not None:
            break
    if data is None:
        return []

    evals: list[HypothesisEvaluation] = []
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


#  Country Hypotheses Node ──────────────────────────────────────────────────

async def country_hypotheses_node(state: VLMCouncilState) -> dict:
    """Create country hypotheses directly from the 5 initial assessments.

    Take the top-N countries by agent-count across all 5 initial assessments.
    The resulting list is both the active hypotheses and the candidate_pool
    that feeds country_evaluate and the tournament bracket.
    """
    config = load_config()

    from collections import Counter
    country_counts: Counter = Counter()

    for name in AGENT_MODULES:
        assessment = state.get(f"{name}_assessment", {})
        for c in assessment.get("candidates", []):
            country = c.get("country", "").strip()
            if country:
                country_counts[country] += 1

    all_countries = [c for c, _ in country_counts.most_common()]
    top_countries = all_countries[:config.max_country_hypotheses]

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
    return {"active_hypotheses": hypotheses, "candidate_pool": top_countries}


# Country Evaluate Node

async def country_evaluate_node(state: VLMCouncilState) -> dict:
    """All 5 specialists evaluate the candidate_pool in parallel (clean-slate).

    Hypotheses come from candidate_pool (set by country_hypotheses).
    Specialists see RAG reference images per candidate.
    """
    pool = state.get("candidate_pool", [])
    if not pool:
        # Fallback: use raw active_hypotheses if candidate_pool is empty
        hypotheses = state.get("active_hypotheses", [])
        _log(f"country_evaluate: candidate_pool empty, falling back to {len(hypotheses)} active_hypotheses")
    else:
        hypotheses = []
        for country in pool:
            hyp_id = "country_" + country.lower().replace(" ", "_")
            hypotheses.append(Hypothesis(
                hypothesis_id=hyp_id,
                level="country",
                value=country,
                statement=f"This image is from {country}",
            ))
        _log(f"country_evaluate: evaluating {len(hypotheses)} survivors: {pool}")

    if not hypotheses:
        return {"hypothesis_evaluations": []}

    async def _eval_agent(name: str) -> list[HypothesisEvaluation]:
        mod = AGENT_MODULES[name]
        t0 = time.time()
        try:
            raw = await asyncio.wait_for(
                mod.evaluate_hypotheses(
                    state["image_b64"], state["image_mime"], hypotheses
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
    return {"hypothesis_evaluations": all_evals, "current_phase": "tournament"}


#  Country Decision Node, replaced by tournament_node


# Build Graph

def build_graph(config: VLMCouncilConfig | None = None) -> StateGraph:
    """Build the VLM Council Tournament-only graph.

    Topology:
        START → prepare_image → [5x specialists: initial assess]
              → country_hypotheses → country_evaluate → tournament → END
    """
    if config is None:
        config = load_config()

    builder = StateGraph(VLMCouncilState)

    # Phase 0: Image preparation
    builder.add_node("prepare_image", prepare_image)

    # Phase 1: Initial assessment (parallel via vLLM continuous batching)
    for name in AGENT_MODULES:
        builder.add_node(f"{name}_agent", _make_agent_node(name))

    # Country phase (no region filter, direct from initial assessments)
    builder.add_node("country_hypotheses", country_hypotheses_node)

    # Country evaluation + Tournament
    builder.add_node("country_evaluate", country_evaluate_node)
    builder.add_node("tournament", tournament_node)

    # Edges
    builder.add_edge(START, "prepare_image")
    for name in AGENT_MODULES:
        builder.add_edge("prepare_image", f"{name}_agent")
    for name in AGENT_MODULES:
        builder.add_edge(f"{name}_agent", "country_hypotheses")

    builder.add_edge("country_hypotheses", "country_evaluate")
    builder.add_edge("country_evaluate", "tournament")

    builder.add_edge("tournament", END)

    return builder.compile()
