"""LLM-as-a-judge, Hub-and-Spoke Stage 2 evaluation.

The judge receives the Street View image plus the full Hub-and-Spoke trace
(initial assessments, judge questions, agent responses, judge synthesis) and
ground truth, then produces ONE structured verdict per image covering all
five agents at once. Per-agent fields become ``dict[agent_name, value]``;
image-level fields stay scalar.

Resume-capable: skips images where ``<image_id>.json`` already exists.
Async via ``asyncio.Semaphore``; OpenAI direct client.

Env vars:
  VLM_JUDGE_LLM_MODEL, model name
  VLM_JUDGE_LLM_API_BASE, API base URL
  VLM_JUDGE_MAX_TOKENS, output token budget (default 2500)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from eval_hubspoke.loader import AGENT_NAMES, RunRecord, load_run, parse_discussion_for_agent
from vlm_council.image_utils import encode_image


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    """Per-image verdict. Per-agent fields are dicts keyed by agent name."""

    # Per-agent
    role_adherence: dict[str, bool] = Field(default_factory=dict)
    hallucination_score: dict[str, float] = Field(default_factory=dict)
    visual_consistency_score: dict[str, float] = Field(default_factory=dict)
    confidence_calibration_score: dict[str, float] = Field(default_factory=dict)
    question_relevance_score: dict[str, float] = Field(default_factory=dict)
    question_relevance_notes: dict[str, str] = Field(default_factory=dict)
    response_update_quality: dict[str, float] = Field(default_factory=dict)
    response_update_notes: dict[str, str] = Field(default_factory=dict)
    targeted_agent_addressed_question: dict[str, bool] = Field(default_factory=dict)
    hallucination_examples: dict[str, list[str]] = Field(default_factory=dict)

    # Image-level
    judge_question_strategy_score: float = 0.5
    judge_question_strategy_notes: str = ""
    judge_synthesis_quality: float = 0.5
    judge_synthesis_notes: str = ""
    discussion_convergence_score: float | None = None
    discussion_convergence_notes: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_AGENT_DOMAINS_LIST = (
    "  - linguistic:   language/script analysis, text on signs, place names, writing systems\n"
    "  - landscape:    terrain, climate cues, soil, topography, settlement patterns\n"
    "  - botanics:     plant species, vegetation, flora native ranges\n"
    "  - regulatory:   road signs, lane markings, infrastructure standards, driving side, license plates\n"
    "  - meta:         camera generation/model, Street View coverage patterns, cross-domain synthesis"
)

_SYSTEM_PROMPT = f"""You are an evaluation judge for a Hub-and-Spoke multi-agent geo-localization pipeline.

Five specialist agents (linguistic, landscape, botanics, regulatory, meta) each independently analyse a Google Street View image and propose country candidates. A judge (the hub) then reviews all assessments and asks targeted follow-up questions to specific agents (spokes). This repeats for up to 3 rounds. You receive the IMAGE, the GROUND TRUTH, and the full trace, and score the QUALITY of this process.

Agent domains:
{_AGENT_DOMAINS_LIST}

IMAGE-GROUNDED FIELDS, inspect the attached image directly before scoring these:
  - hallucination_score      (compare each agent's specific claims against what is actually visible)
  - hallucination_examples   (a quote only counts if the image does not support it)
  - visual_consistency_score (does the agent's evidence match what you see in the image?)
Do NOT infer these from agent text or trace alone. Cross-check every specific
visual claim, text on signs, species, road infrastructure, terrain features, against the image before scoring.

You must output ONE JSON object per image. Every per-agent dict MUST have ALL FIVE keys:
linguistic, landscape, botanics, regulatory, meta.

JSON schema:
{{
  "role_adherence":                     {{"<agent>": <bool>, ...}},
  "hallucination_score":                {{"<agent>": <float 0-1>, ...}},     // INVERTED: higher is worse
  "visual_consistency_score":           {{"<agent>": <float 0-1>, ...}},
  "confidence_calibration_score":       {{"<agent>": <float 0-1>, ...}},
  "question_relevance_score":           {{"<agent>": <float 0-1>, ...}},
  "question_relevance_notes":           {{"<agent>": "<str>", ...}},
  "response_update_quality":            {{"<agent>": <float 0-1>, ...}},
  "response_update_notes":              {{"<agent>": "<str>", ...}},
  "targeted_agent_addressed_question":  {{"<agent>": <bool>, ...}},
  "hallucination_examples":             {{"<agent>": [<str>, ...], ...}},   // up to 3 verbatim quotes per agent
  "judge_question_strategy_score":      <float 0-1>,
  "judge_question_strategy_notes":      "<str>",
  "judge_synthesis_quality":            <float 0-1>,
  "judge_synthesis_notes":              "<str>",
  "discussion_convergence_score":       <float 0-1>,
  "discussion_convergence_notes":       "<str>"
}}

SCORING ANCHORS (use consistently for every 0-1 field):
  0.00 = total failure on this dimension
  0.25 = poor, major problems
  0.50 = mixed/neutral midpoint (use this when agent was NOT questioned for question_relevance_score and response_update_quality)
  0.75 = good, minor weaknesses
  1.00 = flawless on this dimension

hallucination_score (INVERTED, higher is worse):
  Cross-check against the IMAGE before scoring.
  0.00 = no invented claims; all specific claims verifiable in the image
  0.25 = one minor unsupported claim
  0.50 = several unsupported claims OR one major fabrication
  0.75 = pervasive invention
  1.00 = agent fabricated nearly everything

visual_consistency_score (per agent, image-grounded):
  Inspect the attached IMAGE directly and do NOT infer from agent text alone.
  Compare each concrete visual claim the agent made against what is visible.
  1.00 = every concrete visual claim is verifiable in the image
  0.75 = mostly verifiable; one minor claim is vague or slightly off
  0.50 = some claims verifiable, others vague or mildly off
  0.25 = several claims contradict or cannot be verified against the image
  0.00 = agent's evidence directly contradicts what is visible

response_update_quality:
  1.0 = agent gave concrete new observations, directly addressed the question
  0.5 = agent responded but was vague or restated priors (also use 0.5 when NOT questioned)
  0.0 = agent response was empty, refused, or added nothing

question_relevance_score:
  1.0 = judge's question was squarely in this agent's domain
  0.5 = somewhat relevant (also use 0.5 when NOT questioned)
  0.0 = judge asked this agent about something completely outside their domain

judge_question_strategy_score (image-level):
  1.0 = judge identified the key uncertainty and targeted exactly the right agent
  0.5 = questions were reasonable but imprecise
  0.0 = judge asked irrelevant questions or targeted wrong agents

discussion_convergence_score (image-level):
  ONLY meaningful when discussion_rounds > 0. If discussion_rounds == 0, set this
  field to null (no discussion to evaluate).
  1.0 = clear convergence: candidates eliminated across rounds, final decision followed directly from discussion
  0.5 = partial convergence
  0.0 = no convergence: discussion went in circles or judge ignored it

targeted_agent_addressed_question (per agent, ONLY include agents that were questioned):
  true  = agent gave a substantive response specifically addressing the judge's question
  false = agent dodged, gave a generic response, or restated priors without engaging
  IMPORTANT: OMIT agents that were not questioned from this dict entirely. Do not
  default them to true or false, leave the key missing.

role_adherence (per agent, boolean):
  Judge whether the agent stayed within its declared expertise, INDEPENDENT of whether it was questioned.
  Each agent has a specific role:
    - linguistic:   language/script analysis, text on signs, place names, writing systems
    - landscape:    terrain, vegetation-macro, climate cues, geological features
    - botanics:     plant species identification, flora characteristics
    - regulatory:   road infrastructure, license plates, signs, traffic conventions
    - meta:         cross-cutting synthesis, camera artefacts, general geolocation cues
  true  = agent stayed in its role. This includes:
          * Providing analysis within its domain
          * Correctly declaring "no evidence available" or "insufficient" when the
            image contains nothing relevant to its role (e.g. linguistic on an
            image with no legible text). Declining to speculate outside one's
            expertise IS rollentreu.
  false = agent stepped outside its domain by making claims that belong to another
          agent's role (e.g. linguistic identifying plant species; botanics
          discussing text on signs). Being silent or "not questioned" is NEVER
          grounds for false, only positive out-of-domain claims are.

hallucination_examples (per agent, image-grounded):
  Up to 3 verbatim quotes from THIS agent's assessments/responses that are NOT supported
  by the image. A quote only counts if you can confirm by looking at the image that the
  claim is wrong or unsupported. Empty list if none. Quote the agent verbatim, do not
  paraphrase.

Output ONLY the JSON object. No prose, no markdown fences."""


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------

def _short(s: str, n: int = 600) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[:n] + " [...]"


def _format_assessment(agent: str, assessment: dict | None) -> str:
    if not assessment:
        return f"### {agent}\n  (no initial assessment)\n"
    cands = assessment.get("candidates") or []
    evidence = assessment.get("evidence") or []
    lines = [f"### {agent}", "  Candidates:"]
    for i, c in enumerate(cands[:5], 1):
        lines.append(
            f"    {i}. {c.get('country', '?')} "
            f"(conf={c.get('confidence', '?')}): {_short(c.get('reasoning', ''), 300)}"
        )
    if evidence:
        lines.append("  Evidence:")
        for e in evidence[:6]:
            lines.append(f"    - {_short(str(e), 150)}")
    return "\n".join(lines) + "\n"


def build_trace(record: RunRecord) -> str:
    """Build the per-image evaluation trace (all five agents + full discussion)."""
    parts: list[str] = []

    parts.append("## Initial assessments (all five agents)\n")
    for agent in AGENT_NAMES:
        parts.append(_format_assessment(agent, (record.assessments or {}).get(agent)))

    parts.append("## Discussion log (judge questions + agent responses)")
    if record.discussion_log:
        for entry in record.discussion_log:
            rnd = entry.get("round_number", "?")
            tgt = entry.get("target_agent", "?")
            q = _short(entry.get("judge_question", ""), 400)
            resp = _short(entry.get("agent_response", ""), 400)
            parts.append(f"  Round {rnd} → {tgt}:")
            parts.append(f"    Q: {q}")
            parts.append(f"    A: {resp if resp else '(empty)'}")
        parts.append(f"  Total discussion_rounds: {record.discussion_rounds}")
    else:
        parts.append(f"  (no discussion; discussion_rounds={record.discussion_rounds})")

    parts.append("\n## Per-agent questioned flag (derived from trace)")
    for agent in AGENT_NAMES:
        was_q = bool(parse_discussion_for_agent(record, agent))
        parts.append(f"  {agent}: was_questioned={was_q}")

    parts.append("\n## Judge final reasoning")
    final = _short(record.raw.get("country_result", "") or record.final_reasoning or "", 800)
    parts.append(f"  {final}")

    correct = "YES" if record.is_correct else "NO"
    parts.append(
        f"\n## Ground truth\n  {record.truth_country_name} ({record.truth_country_code.upper()}) "
        f"| Prediction: {record.pred_country} | Correct: {correct}"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json(content: str) -> dict | None:
    text = (content or "").strip()
    m = re.search(r"<think>.*?</think>(.*)", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start: i + 1])
                except Exception:
                    return None
    return None


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

def _resolve_image_path(record: RunRecord, image_root: Path | None) -> Path | None:
    raw = record.image_path
    if image_root is not None and raw:
        cand = image_root / Path(raw).name
        if cand.exists():
            return cand
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    if image_root is not None:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            cand = image_root / f"{record.image_id}{ext}"
            if cand.exists():
                return cand
    return None


# ---------------------------------------------------------------------------
# Async judge call (one per image)
# ---------------------------------------------------------------------------

_MAX_TOKENS = int(os.environ.get("VLM_JUDGE_MAX_TOKENS", "2500"))


def _build_openai_messages(record: RunRecord, image_b64: str, image_mime: str) -> list[dict]:
    truth_block = (
        f"[GROUND TRUTH, known to you, hidden from the agents]\n"
        f"  country: {record.truth_country_name} ({record.truth_country_code.upper()})\n"
        f"  coordinates: {record.truth_lat:.4f}, {record.truth_lon:.4f}\n"
    )
    trace = build_trace(record)
    text = (
        truth_block
        + "\n[STREET-VIEW IMAGE, what the agents saw]\n"
        + "(image attached above this prompt)\n\n"
        + "[HUB-AND-SPOKE TRACE]\n"
        + trace
        + "\n\n[YOUR TASK]\n"
        + "Score ALL five agents and the judge orchestration against the rubric. "
        + "Every per-agent dict MUST contain all five keys: linguistic, landscape, "
        + "botanics, regulatory, meta. Output ONLY the JSON object, no prose."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
                {"type": "text", "text": text},
            ],
        },
    ]


async def _judge_one(
    record: RunRecord,
    out_dir: Path,
    image_root: Path | None,
    client,
    model: str,
    sem: asyncio.Semaphore,
) -> tuple[str, str]:
    """Judge one image. Returns (image_id, status)."""
    out_file = out_dir / f"{record.image_id}.json"
    if out_file.exists():
        return record.image_id, "skipped"

    img_path = _resolve_image_path(record, image_root)
    if img_path is None:
        with open(out_file, "w") as f:
            json.dump(
                {"image_id": record.image_id, "error": "image not found",
                 "image_path": record.image_path},
                f, indent=2,
            )
        return record.image_id, "no_image"

    try:
        b64, mime = encode_image(str(img_path))
    except Exception as e:
        with open(out_file, "w") as f:
            json.dump({"image_id": record.image_id,
                       "error": f"encode failed: {e}"}, f, indent=2)
        return record.image_id, "encode_error"

    messages = _build_openai_messages(record, b64, mime)

    async with sem:
        try:
            _extra_body: dict = {}
            # Qwen3.6 defaults to thinking-mode which prepends 2-5k reasoning
            # tokens before the answer. For the judge we want direct JSON, so
            # we disable it explicitly. This kwarg is a no-op for other models.
            import os as _os
            if _os.environ.get("VLM_JUDGE_DISABLE_THINKING", "").lower() in ("1", "true", "yes"):
                _extra_body["chat_template_kwargs"] = {"enable_thinking": False}

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=_MAX_TOKENS,
                extra_body=_extra_body or None,
            )
        except Exception as e:
            with open(out_file, "w") as f:
                json.dump({"image_id": record.image_id,
                           "error": f"llm failed: {e}"}, f, indent=2)
            return record.image_id, "llm_error"

    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = choice.finish_reason

    parsed = _parse_json(content)
    if parsed is None:
        with open(out_file, "w") as f:
            json.dump({"image_id": record.image_id,
                       "error": "parse_error",
                       "finish_reason": finish_reason,
                       "raw": content[:1500]}, f, indent=2)
        return record.image_id, "parse_error"

    try:
        verdict = JudgeVerdict.model_validate(parsed)
        verdict_dump = verdict.model_dump()
        validation_err = None
    except ValidationError as e:
        verdict_dump = parsed
        validation_err = str(e)[:500]

    # was_questioned is derived from the trace, not asked of the judge.
    was_questioned = {
        a: bool(parse_discussion_for_agent(record, a)) for a in AGENT_NAMES
    }

    payload: dict = {
        "image_id": record.image_id,
        "truth_country": record.truth_country_name,
        "pred_country": record.pred_country,
        "is_correct": record.is_correct,
        "discussion_rounds": record.discussion_rounds,
        "was_questioned": was_questioned,
        "finish_reason": finish_reason,
        "verdict": verdict_dump,
    }
    if validation_err:
        payload["validation_error"] = validation_err

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    return record.image_id, "ok"


async def _run_async(
    records: list[RunRecord],
    out_dir: Path,
    image_root: Path | None,
    model: str,
    api_base: str,
    concurrency: int,
) -> dict[str, int]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=api_base,
    )
    sem = asyncio.Semaphore(max(1, concurrency))

    tasks = [
        asyncio.create_task(
            _judge_one(r, out_dir, image_root, client, model, sem)
        )
        for r in records
    ]

    counts: dict[str, int] = {}
    for fut in asyncio.as_completed(tasks):
        image_id, status = await fut
        counts[status] = counts.get(status, 0) + 1
        print(f"[judge] {image_id}: {status}")
    return counts


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(
    *,
    results: Path,
    gt: Path,
    out: Path,
    image_root: Path | None = None,
    model: str | None = None,
    api_base: str | None = None,
    concurrency: int = 1,
    limit: int | None = None,
    file_list: Path | None = None,
) -> dict:
    judge_dir = out / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)

    _model = model or os.environ.get("VLM_JUDGE_LLM_MODEL", "")
    _api_base = api_base or os.environ.get("VLM_JUDGE_LLM_API_BASE", "")

    if not _model:
        raise ValueError("Judge model not set. Use --model or VLM_JUDGE_LLM_MODEL env var.")
    if not _api_base:
        raise ValueError("Judge API base not set. Use --api-base or VLM_JUDGE_LLM_API_BASE env var.")

    records = load_run(results, gt)

    # If a file-list whitelist was provided, restrict to those image_ids.
    # Used by the parallel short-gpu launcher.
    if file_list is not None:
        with open(file_list) as f:
            wanted = {line.strip() for line in f if line.strip()}
        before = len(records)
        records = [r for r in records if r.image_id in wanted]
        print(
            f"[judge] file-list restricted {before} loaded records to "
            f"{len(records)} (of {len(wanted)} requested image_ids)"
        )

    if limit:
        records = records[:limit]

    print(f"[judge] evaluating {len(records)} images (one call per image, model={_model})")

    counts = asyncio.run(
        _run_async(records, judge_dir, image_root, _model, _api_base, concurrency)
    )
    print(f"[judge] done: {counts}")
    return counts
