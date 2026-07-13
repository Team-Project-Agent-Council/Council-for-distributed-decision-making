"""LangGraph pipeline for the VLM Council, Debate approach.

Topology:
    prepare_image
        → [5 agents Round 1 parallel]
        → round1_complete (barrier)
        → moderator (identifies contradictions, decides pairings)
        → CONDITIONAL:
            terminate=true  → judge_final → END
            terminate=false → debate_round (all pairings in parallel)
                                → moderator (loop)
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
    DebateMessage,
    DebatePairing,
    ModeratorDecision,
)
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
    gemma_match = re.search(r"<\|think\|>(.*?)<\|/think\|>(.*)", text, re.DOTALL)
    if gemma_match:
        return gemma_match.group(1).strip(), gemma_match.group(2).strip()

    think_match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), think_match.group(2).strip()

    channel_match = re.search(r"<\|channel\>thought(.*?)<channel\|>(.*)", text, re.DOTALL)
    if channel_match:
        return channel_match.group(1).strip(), channel_match.group(2).strip()

    think_end = re.search(r"</think>(.*)", text, re.DOTALL)
    if think_end:
        thinking = text[:think_end.start()].strip()
        response = think_end.group(1).strip()
        return thinking, response

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


def _parse_debate_response(agent_name: str, raw: str) -> DebateMessage:
    """Parse a debate response from an agent."""
    _, response = _strip_think_tags(raw)
    text = response.strip() if response else raw.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    for candidate_text in [text, raw]:
        try:
            data = json.loads(candidate_text.strip())
            if isinstance(data, dict) and "position" in data:
                return DebateMessage(
                    agent_name=agent_name,
                    position=data.get("position", "unknown"),
                    revised=bool(data.get("revised", False)),
                    confidence=data.get("confidence", "low"),
                    argument=data.get("argument", ""),
                    key_evidence=data.get("key_evidence", []),
                )
        except (json.JSONDecodeError, ValueError):
            pass

    for search_text in [text, raw]:
        matches = list(re.finditer(r"\{[^{}]*\}", search_text, re.DOTALL))
        for match in reversed(matches):
            try:
                data = json.loads(match.group())
                if "position" in data:
                    return DebateMessage(
                        agent_name=agent_name,
                        position=data.get("position", "unknown"),
                        revised=bool(data.get("revised", False)),
                        confidence=data.get("confidence", "low"),
                        argument=data.get("argument", ""),
                        key_evidence=data.get("key_evidence", []),
                    )
            except json.JSONDecodeError:
                continue

    return DebateMessage(
        agent_name=agent_name,
        position="unknown",
        revised=False,
        confidence="low",
        argument=raw[:200] if raw else "",
        key_evidence=[],
    )


def _parse_moderator_response(raw: str) -> dict:
    """Parse the moderator's JSON response."""
    _, response = _strip_think_tags(raw)
    text = response.strip() if response else raw.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    for candidate_text in [text, raw]:
        try:
            data = json.loads(candidate_text.strip())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    for search_text in [text, raw]:
        matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", search_text, re.DOTALL))
        for match in reversed(matches):
            try:
                data = json.loads(match.group())
                if "terminate" in data or "pairings" in data:
                    return data
            except json.JSONDecodeError:
                continue

    return {"contradictions": [], "pairings": [], "reasoning": "Failed to parse", "terminate": True, "termination_reason": "parse_failure"}


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


# === Barrier between Round 1 and Debate ===

async def round1_complete(state: VLMCouncilState) -> dict:
    """No-op barrier node to synchronize all Round 1 agents before debate."""
    _log("round1_complete: all agents finished Round 1, starting debate moderation...")
    return {}


# === Moderator node ===

async def moderator_node(state: VLMCouncilState) -> dict:
    """Judge examines current positions, identifies contradictions, decides pairings."""
    current_round = state.get("current_debate_round", 0)
    max_rounds = int(os.environ.get("DEBATE_MAX_ROUNDS", "3"))
    min_confidence = os.environ.get("DEBATE_MIN_CONFIDENCE", "medium")

    _log(f"moderator: debate round {current_round} (max {max_rounds})")

    if current_round >= max_rounds:
        _log("moderator: max rounds reached, terminating debate")
        decision = ModeratorDecision(
            debate_round=current_round,
            contradictions_found=[],
            pairings_opened=[],
            reasoning="Maximum debate rounds reached.",
            terminate=True,
            termination_reason="max_rounds_reached",
        )
        return {
            "moderator_decisions": [decision],
            "debate_terminated": True,
            "current_debate_round": current_round,
        }

    # Check for stalemate: if last round produced no revisions
    if current_round > 0:
        debate_pairings = state.get("debate_pairings", [])
        last_round_pairings = [p for p in debate_pairings if p.get("debate_round") == current_round]
        any_revision = any(
            ex.get("revised", False)
            for p in last_round_pairings
            for ex in p.get("exchanges", [])
        )
        if not any_revision and last_round_pairings:
            _log("moderator: stalemate detected (no revisions last round), terminating")
            decision = ModeratorDecision(
                debate_round=current_round,
                contradictions_found=[],
                pairings_opened=[],
                reasoning="Stalemate: no agent revised their position in the last round.",
                terminate=True,
                termination_reason="stalemate",
            )
            return {
                "moderator_decisions": [decision],
                "debate_terminated": True,
                "current_debate_round": current_round,
            }

    # Programmatic pre-check: get actual top-1 positions
    positions = _get_all_current_positions(state)
    _log(f"moderator: current positions: {positions}")

    unique_countries = set(pos["country"] for pos in positions.values() if pos["country"])
    if len(unique_countries) <= 1:
        _log(f"moderator: consensus on {unique_countries}, terminating")
        decision = ModeratorDecision(
            debate_round=current_round,
            contradictions_found=[],
            pairings_opened=[],
            reasoning=f"All agents agree on {next(iter(unique_countries)) if unique_countries else 'unknown'}.",
            terminate=True,
            termination_reason="consensus",
        )
        return {
            "moderator_decisions": [decision],
            "debate_terminated": True,
            "current_debate_round": current_round,
        }

    # Call the LLM moderator for nuanced pairing decisions
    t0 = time.time()
    try:
        raw = await asyncio.wait_for(
            judge_agent.moderate(state, min_confidence=min_confidence),
            timeout=VLM_CALL_TIMEOUT,
        )
        elapsed = time.time() - t0
        _log(f"moderator: LLM responded in {elapsed:.1f}s")
    except (asyncio.TimeoutError, Exception) as e:
        elapsed = time.time() - t0
        _log(f"moderator: ERROR {type(e).__name__} after {elapsed:.1f}s, terminating")
        decision = ModeratorDecision(
            debate_round=current_round,
            contradictions_found=[],
            pairings_opened=[],
            reasoning=f"Moderator error: {e}",
            terminate=True,
            termination_reason="error",
        )
        return {
            "moderator_decisions": [decision],
            "debate_terminated": True,
            "current_debate_round": current_round,
        }

    parsed = _parse_moderator_response(raw)

    # Validate pairings: ensure agents actually disagree
    valid_pairings = []
    for pairing in parsed.get("pairings", []):
        a = pairing.get("agent_a", "")
        b = pairing.get("agent_b", "")
        if a in AGENT_MODULES and b in AGENT_MODULES and a != b:
            pos_a = positions.get(a, {}).get("country", "")
            pos_b = positions.get(b, {}).get("country", "")
            if pos_a and pos_b and pos_a != pos_b:
                valid_pairings.append({"agent_a": a, "agent_b": b})

    should_terminate = parsed.get("terminate", False) or len(valid_pairings) == 0

    decision = ModeratorDecision(
        debate_round=current_round,
        contradictions_found=parsed.get("contradictions", []),
        pairings_opened=valid_pairings,
        reasoning=parsed.get("reasoning", ""),
        terminate=should_terminate,
        termination_reason=parsed.get("termination_reason", "") if should_terminate else "",
    )

    _log(f"moderator: {len(valid_pairings)} pairings, terminate={should_terminate}")

    return {
        "moderator_decisions": [decision],
        "debate_terminated": should_terminate,
        "current_debate_round": current_round + (0 if should_terminate else 1),
    }


def _get_all_current_positions(state: VLMCouncilState) -> dict:
    """Get each agent's current top-1 position (from debate or Round 1)."""
    positions = {}
    for name in AGENT_MODULES:
        # Check debate history for latest position
        debate_pairings = state.get("debate_pairings", [])
        found = False
        for pairing in reversed(debate_pairings):
            for exchange in reversed(pairing.get("exchanges", [])):
                if exchange.get("agent_name") == name:
                    positions[name] = {
                        "country": exchange["position"],
                        "confidence": exchange["confidence"],
                    }
                    found = True
                    break
            if found:
                break

        if not found:
            assessment = state.get(f"round_1_{name}", {})
            candidates = assessment.get("candidates", [])
            if candidates:
                positions[name] = {
                    "country": candidates[0]["country"],
                    "confidence": candidates[0]["confidence"],
                }
            else:
                positions[name] = {"country": "", "confidence": ""}

    return positions


# === Debate round node ===

async def debate_round_node(state: VLMCouncilState) -> dict:
    """Execute all debate pairings decided by the moderator."""
    moderator_decisions = state.get("moderator_decisions", [])
    if not moderator_decisions:
        return {"debate_pairings": []}

    latest_decision = moderator_decisions[-1]
    pairings_to_run = latest_decision.get("pairings_opened", [])
    current_round = state.get("current_debate_round", 1)

    if not pairings_to_run:
        return {"debate_pairings": []}

    _log(f"debate_round: executing {len(pairings_to_run)} pairings for round {current_round}")

    all_round1_context = _format_all_round1(state)
    tasks = []
    for pairing in pairings_to_run:
        tasks.append(_execute_debate_pairing(
            state, pairing["agent_a"], pairing["agent_b"],
            current_round, all_round1_context,
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    new_pairings = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            pairing = pairings_to_run[i]
            _log(f"debate_round: ERROR in {pairing['agent_a']} vs {pairing['agent_b']}: {result}")
            new_pairings.append(DebatePairing(
                debate_round=current_round,
                agent_a=pairing["agent_a"],
                agent_b=pairing["agent_b"],
                agent_a_initial_position="error",
                agent_b_initial_position="error",
                exchanges=[],
            ))
        else:
            new_pairings.append(result)

    return {"debate_pairings": new_pairings}


async def _execute_debate_pairing(
    state: VLMCouncilState,
    agent_a_name: str,
    agent_b_name: str,
    debate_round: int,
    all_round1_context: str,
) -> DebatePairing:
    """Execute a real back-and-forth debate between two agents.

    Agents take turns: A speaks, B reads A's response and replies,
    A reads B's reply and responds, etc. Continues until one revises
    or max exchanges reached.
    """
    max_exchanges = int(os.environ.get("DEBATE_MAX_EXCHANGES", "6"))
    mod_a = AGENT_MODULES[agent_a_name]
    mod_b = AGENT_MODULES[agent_b_name]

    positions = _get_all_current_positions(state)
    pos_a = positions.get(agent_a_name, {})
    pos_b = positions.get(agent_b_name, {})

    # Build prior debate history between these two (from previous rounds)
    prior_history = _get_pairing_history(state, agent_a_name, agent_b_name)

    _log(f"debate: {agent_a_name}({pos_a.get('country')}) vs {agent_b_name}({pos_b.get('country')}), multi-turn")

    exchanges: list[DebateMessage] = []
    t0 = time.time()

    # Current positions as JSON for the prompt
    current_pos_a = json.dumps({"country": pos_a.get("country", ""), "confidence": pos_a.get("confidence", "")})
    current_pos_b = json.dumps({"country": pos_b.get("country", ""), "confidence": pos_b.get("confidence", "")})

    for turn in range(max_exchanges):
        # Build conversation so far (this round's exchanges)
        conversation_so_far = _format_conversation(exchanges)
        full_history = prior_history + ("\n" + conversation_so_far if conversation_so_far else "")

        # Determine whose turn it is (alternate, A starts)
        if turn % 2 == 0:
            # Agent A's turn
            speaking_agent = agent_a_name
            speaking_mod = mod_a
            own_pos = current_pos_a
            opponent_name = agent_b_name
            opponent_pos = current_pos_b
        else:
            # Agent B's turn
            speaking_agent = agent_b_name
            speaking_mod = mod_b
            own_pos = current_pos_b
            opponent_name = agent_a_name
            opponent_pos = current_pos_a

        try:
            raw = await asyncio.wait_for(
                speaking_mod.debate(
                    state["image_b64"], state["image_mime"],
                    own_pos, opponent_name, opponent_pos,
                    full_history, all_round1_context,
                ),
                timeout=VLM_CALL_TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception) as e:
            _log(f"debate: {speaking_agent} ERROR on turn {turn}: {e}")
            raw = '{"position": "unknown", "revised": false, "confidence": "low", "argument": "error", "key_evidence": []}'

        msg = _parse_debate_response(speaking_agent, raw)
        exchanges.append(msg)

        revised_str = " (REVISED)" if msg["revised"] else ""
        _log(f"debate: turn {turn+1}/{max_exchanges}, {speaking_agent} → {msg['position']}{revised_str}")

        # Update current position if revised
        if msg["revised"]:
            new_pos = json.dumps({"country": msg["position"], "confidence": msg["confidence"]})
            if turn % 2 == 0:
                current_pos_a = new_pos
            else:
                current_pos_b = new_pos

        # Early termination: if agent revised, give opponent one final reply
        if msg["revised"] and turn < max_exchanges - 1:
            # Let the other agent have the last word
            final_turn = turn + 1
            conversation_so_far = _format_conversation(exchanges)
            full_history = prior_history + ("\n" + conversation_so_far if conversation_so_far else "")

            if final_turn % 2 == 0:
                final_agent = agent_a_name
                final_mod = mod_a
                final_own_pos = current_pos_a
                final_opp_name = agent_b_name
                final_opp_pos = current_pos_b
            else:
                final_agent = agent_b_name
                final_mod = mod_b
                final_own_pos = current_pos_b
                final_opp_name = agent_a_name
                final_opp_pos = current_pos_a

            try:
                raw_final = await asyncio.wait_for(
                    final_mod.debate(
                        state["image_b64"], state["image_mime"],
                        final_own_pos, final_opp_name, final_opp_pos,
                        full_history, all_round1_context,
                    ),
                    timeout=VLM_CALL_TIMEOUT,
                )
            except (asyncio.TimeoutError, Exception) as e:
                _log(f"debate: {final_agent} ERROR on final reply: {e}")
                raw_final = '{"position": "unknown", "revised": false, "confidence": "low", "argument": "error", "key_evidence": []}'

            final_msg = _parse_debate_response(final_agent, raw_final)
            exchanges.append(final_msg)
            _log(f"debate: final reply, {final_agent} → {final_msg['position']}")
            break

    elapsed = time.time() - t0
    _log(f"debate: {agent_a_name} vs {agent_b_name} done, {len(exchanges)} exchanges in {elapsed:.1f}s")

    return DebatePairing(
        debate_round=debate_round,
        agent_a=agent_a_name,
        agent_b=agent_b_name,
        agent_a_initial_position=pos_a.get("country", "unknown"),
        agent_b_initial_position=pos_b.get("country", "unknown"),
        exchanges=exchanges,
    )


def _format_conversation(exchanges: list[DebateMessage]) -> str:
    """Format the current round's exchanges as readable conversation."""
    if not exchanges:
        return ""
    lines = []
    for ex in exchanges:
        revised_str = " (REVISED)" if ex.get("revised") else ""
        lines.append(
            f"{ex['agent_name']}{revised_str}: {ex.get('position', '?')} "
            f"({ex.get('confidence', '?')}), {ex.get('argument', '')}"
        )
    return "\n".join(lines)


def _get_pairing_history(state: VLMCouncilState, agent_a: str, agent_b: str) -> str:
    """Get previous debate exchanges between these two agents."""
    debate_pairings = state.get("debate_pairings", [])
    relevant = []
    for pairing in debate_pairings:
        agents_in_pairing = {pairing.get("agent_a"), pairing.get("agent_b")}
        if agent_a in agents_in_pairing and agent_b in agents_in_pairing:
            for ex in pairing.get("exchanges", []):
                revised_str = " (REVISED)" if ex.get("revised") else ""
                relevant.append(
                    f"{ex['agent_name']}{revised_str}: {ex.get('position', '?')} "
                    f"({ex.get('confidence', '?')}), {ex.get('argument', '')}"
                )
    return "\n".join(relevant)


def _format_all_round1(state: VLMCouncilState) -> str:
    """Format all Round 1 assessments as text for debate context."""
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


# === Judge final node ===

async def judge_final_node(state: VLMCouncilState) -> dict:
    """Judge makes the final country determination based on Round 1 + debate.

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


# === Routing function ===

def should_continue_debate(state: VLMCouncilState) -> str:
    """Routing function: decide whether to continue debating or finalize."""
    if state.get("debate_terminated", False):
        return "finalize"
    return "debate"


# === Build the graph ===

def build_graph(config: VLMCouncilConfig | None = None) -> StateGraph:
    """Build the VLM Council graph with Debate topology.

    Topology:
        prepare_image
            → [5 agents Round 1 in parallel]
            → round1_complete (barrier)
            → moderator (identifies contradictions)
            → CONDITIONAL: debate_round ↔ moderator (loop) OR judge_final
            → END
    """
    if config is None:
        config = load_config()

    builder = StateGraph(VLMCouncilState)

    # Nodes
    builder.add_node("prepare_image", prepare_image)

    for name in AGENT_MODULES:
        builder.add_node(f"round1_{name}", _make_round1_node(name))

    builder.add_node("round1_complete", round1_complete)
    builder.add_node("moderator", moderator_node)
    builder.add_node("debate_round", debate_round_node)
    builder.add_node("judge_final", judge_final_node)

    # Edges: START → prepare_image → [Round 1 agents]
    builder.add_edge(START, "prepare_image")

    agent_names = list(AGENT_MODULES.keys())
    for name in agent_names:
        builder.add_edge("prepare_image", f"round1_{name}")

    # Round 1 agents → barrier
    for name in agent_names:
        builder.add_edge(f"round1_{name}", "round1_complete")

    # Barrier → moderator
    builder.add_edge("round1_complete", "moderator")

    # Moderator → conditional: debate or finalize
    builder.add_conditional_edges(
        "moderator",
        should_continue_debate,
        {
            "debate": "debate_round",
            "finalize": "judge_final",
        },
    )

    # Debate round → back to moderator (loop)
    builder.add_edge("debate_round", "moderator")

    # Judge final → END
    builder.add_edge("judge_final", END)

    return builder.compile()
