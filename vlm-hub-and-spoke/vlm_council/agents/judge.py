"""Judge Agent: Leads hub-and-spoke discussion and makes final determination."""

from __future__ import annotations

import json
import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from vlm_council.llm import get_vlm

# Gemma 4 thinking token is injected before system prompt when VLM_JUDGE_THINKING=true
_THINKING_ENABLED = os.environ.get("VLM_JUDGE_THINKING", "false").lower() in ("true", "1", "yes")
_THINK_PREFIX = "<|think|>\n" if _THINKING_ENABLED else ""

# Phase 2a: Review assessments and decide next action

REVIEW_SYSTEM_PROMPT = """\
You are the Judge of a GeoGuessr council. 5 specialist agents have each analyzed the same Google Street View image.

Your task: use the agents' evidence to eliminate impossible candidates, resolve disagreements, and determine the country. Targeted discussion with agents is your primary tool, use it actively. Finalizing without discussion is reserved for cases where the answer is already locked in by hard, country-specific evidence.

ANALYSIS, before asking questions or finalizing:
1. Check for ELIMINATING evidence first. Some evidence rules out entire groups of countries:
   - Left-hand traffic eliminates all right-hand traffic countries (and vice versa)
   - A specific script (Cyrillic, Arabic, Thai, etc.) eliminates countries that don't use it
   - A specific license plate format eliminates countries with different formats
   If any agent provides such evidence, immediately discard contradicted candidates, even if other agents ranked them highly.

2. Among the remaining candidates, count which country appears across multiple agents at any position.

3. When an agent has HIGH confidence based on specific evidence (not regional), investigate whether that evidence truly distinguishes their pick from alternatives.

4. When top candidates are neighboring countries, ask agents what specifically distinguishes one from the other in their domain.

ASK QUESTIONS (do NOT finalize yet) whenever ANY of the following holds:
   a. Agents disagree on the top candidate, OR no single country is shared by ≥3 agents at high/medium confidence.
   b. The leading candidate has plausible neighbors or look-alikes (e.g., Belgium/Netherlands, Argentina/Uruguay, Malaysia/Indonesia, Czechia/Slovakia, Austria/Germany) that have not been explicitly ruled out by an agent.
   c. The leading candidate's support comes mainly from generic regional cues (climate, vegetation type, road style) without specific identifying evidence (text, sign, plate, endemic species, brand).
   d. Any agent flagged contradicting evidence against the leading candidate that has not been resolved.
   e. No agent provided HARD eliminating evidence narrowing the answer to a single country.
   Prefer parallel questions across multiple agents in one round.

FINALIZE only when ALL of the following are true:
   - At least 3 agents independently named the SAME top country at high or medium confidence.
   - At least one agent provided HARD identifying evidence (script, driving side, plate format, country-specific text/sign/brand, endemic species) consistent with that country.
   - No agent provided evidence that contradicts that country.
   - No plausible neighboring or look-alike country remains untested.

You can ask MULTIPLE agents in parallel.

Respond with JSON only:
- {"action": "questions", "questions": [{"target_agent": "<name>", "question": "<question>"}, ...]}
- {"action": "finalize"}

Valid target_agent values: "linguistic", "landscape", "botanics", "regulatory", "meta"\
"""


def _format_assessments(state: dict) -> str:
    """Format all agent assessments for the judge."""
    agents = ["linguistic", "landscape", "botanics", "regulatory", "meta"]
    parts = []
    for name in agents:
        assessment = state.get(f"{name}_assessment", {})
        candidates = assessment.get("candidates", [])
        evidence = assessment.get("evidence", [])
        evidence_str = ", ".join(str(e) for e in evidence) if evidence else "(none)"

        if not candidates:
            parts.append(f"[{name.upper()} AGENT]\n  (insufficient evidence)")
            continue

        cand_lines = []
        for c in candidates:
            country = c.get("country", "?")
            conf = c.get("confidence", "?")
            reasoning = c.get("reasoning", "")
            cand_lines.append(f"  - {country} ({conf}): {reasoning}")
        parts.append(
            f"[{name.upper()} AGENT]\n"
            + "\n".join(cand_lines) + "\n"
            f"  Evidence: {evidence_str}"
        )
    return "\n\n".join(parts)


def _format_discussion_log(discussion_log: list[dict]) -> str:
    """Format prior discussion rounds for context."""
    if not discussion_log:
        return "(No prior discussion)"
    parts = []
    for entry in discussion_log:
        parts.append(
            f"Round {entry['round_number']}:\n"
            f"  Judge asked {entry['target_agent']}: {entry['judge_question']}\n"
            f"  Response: {entry['agent_response']}"
        )
    return "\n\n".join(parts)


def parse_review_decision(text: str) -> dict:
    """Parse the judge's review decision from JSON output.

    Handles Thinking model output by stripping thinking chains first.
    Supports: explicit <think>...</think> and Gemma 4 <|channel>thought...<channel|>
    Supports both single-question and multi-question formats:
    - {"action": "question", "target_agent": "...", "question": "..."}
    - {"action": "questions", "questions": [{"target_agent": "...", "question": "..."}, ...]}
    - {"action": "finalize"}
    """
    # Strip explicit <think>...</think> wrapper
    think_match = re.search(r"<think>.*?</think>(.*)", text, re.DOTALL)
    if think_match:
        text = think_match.group(1)
    # Strip Gemma 4 thinking channel
    channel_match = re.search(r"<\|channel\>thought.*?<channel\|>(.*)", text, re.DOTALL)
    if channel_match:
        text = channel_match.group(1)
    # Strip </think> without opening tag (vLLM format)
    think_end = re.search(r"</think>(.*)", text, re.DOTALL)
    if think_end:
        text = think_end.group(1)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"action": "finalize"}
    try:
        decision = json.loads(match.group())
        action = decision.get("action")
        if action == "finalize":
            return decision
        if action == "question" and "target_agent" in decision and "question" in decision:
            # Single question => normalize to multi-question format
            return {
                "action": "questions",
                "questions": [{"target_agent": decision["target_agent"], "question": decision["question"]}],
            }
        if action == "questions" and "questions" in decision:
            # Multi-question: validate entries
            valid = [q for q in decision["questions"] if "target_agent" in q and "question" in q]
            if valid:
                return {"action": "questions", "questions": valid}
        return decision
    except json.JSONDecodeError:
        pass
    return {"action": "finalize"}


async def review(state: dict, llm=None) -> dict:
    """Review all assessments and decide: finalize or ask a question."""
    if llm is None:
        llm = get_vlm("judge")
    assessments_text = _format_assessments(state)
    discussion_text = _format_discussion_log(state.get("discussion_log", []))

    response = await llm.ainvoke([
        SystemMessage(content=_THINK_PREFIX + REVIEW_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Agent Assessments:\n\n{assessments_text}\n\n"
            f"Prior Discussion:\n{discussion_text}\n\n"
            "Decide: finalize or ask a targeted question. Respond with JSON only."
        )),
    ])
    return parse_review_decision(response.content)


# Phase 2b: Final determination of the country 

FINAL_SYSTEM_PROMPT = """\
You are the Judge making the final country determination for a GeoGuessr image.

You have assessments from five specialist agents plus any discussion clarifications.

Decision process:
1. ELIMINATE first: check if any agent's evidence rules out candidates. Driving side, script, license plate format, or other hard constraints can immediately discard countries, regardless of how many agents suggested them.
2. Evaluate ALL remaining candidates across all agents. Specific evidence (identified text, unique road sign, endemic species) outweighs generic regional evidence (temperate climate, flat terrain) that applies to multiple countries equally.
3. For your chosen country, verify that no agent provided evidence that contradicts it.

Estimate coordinates based on your chosen country and any regional clues from the agents.

Respond with EXACTLY this format:
Country: <name>
Coordinates: <lat>, <lon>
Reasoning: <2-3 sentences explaining your choice and what evidence supported it>\
"""


async def finalize(state: dict, llm=None) -> str:
    """Make the final country determination"""
    if llm is None:
        llm = get_vlm("judge")
    assessments_text = _format_assessments(state)
    discussion_text = _format_discussion_log(state.get("discussion_log", []))

    response = await llm.ainvoke([
        SystemMessage(content=_THINK_PREFIX + FINAL_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Agent Assessments:\n\n{assessments_text}\n\n"
            f"Discussion Clarifications:\n{discussion_text}\n\n"
            "Make your final determination. Provide Country, Coordinates, and Reasoning."
        )),
    ])
    return response.content
