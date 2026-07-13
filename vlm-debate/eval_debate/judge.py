"""LLM-as-judge evaluation for the Debate approach.

The judge receives the Street View image plus the full Debate trace
(R1 assessments, all pairings + exchanges, moderator decisions, final
judge reasoning) and ground truth, then produces ONE structured verdict
per image covering all five agents + image-level orchestration. Per-agent
fields become ``dict[agent_name, value]``; image-level fields stay scalar.

Resume-capable: skips images where ``<image_id>.json`` already exists.
Async via ``asyncio.Semaphore``.

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

from eval_debate.loader import AGENT_NAMES, RunRecord, load_run
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
    argument_quality_score: dict[str, float] = Field(default_factory=dict)
    argument_quality_notes: dict[str, str] = Field(default_factory=dict)
    revision_justification_score: dict[str, float] = Field(default_factory=dict)
    revision_justification_notes: dict[str, str] = Field(default_factory=dict)
    debate_contribution_score: dict[str, float] = Field(default_factory=dict)
    debate_contribution_notes: dict[str, str] = Field(default_factory=dict)
    hallucination_examples: dict[str, list[str]] = Field(default_factory=dict)

    # Image-level
    moderator_pairing_quality_score: float = 0.5
    moderator_pairing_notes: str = ""
    judge_synthesis_quality: float = 0.5
    judge_synthesis_notes: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_AGENT_DOMAINS = (
    "  - linguistic:  language text, scripts, written signs, place names, license plate letter patterns\n"
    "  - landscape:   terrain, soil colour, climate, elevation, agricultural patterns, hemisphere/sun angle\n"
    "  - botanics:    vegetation species, trees, grass types, endemic flora\n"
    "  - regulatory:  road infrastructure, signs (non-text visual aspects), driving side, road markings,\n"
    "                 utility poles, bollards, fence styles, license plate formats\n"
    "  - meta:        camera generation/rig type, Street View coverage patterns, image quality artefacts"
)

_SYSTEM_PROMPT = f"""You are an evaluation judge for a multi-agent geo-localization pipeline that uses an adversarial debate stage.

Five specialist agents independently propose country candidates in Round 1. A moderator then picks pairings of agents who disagree and runs short debates over up to a few rounds. A final judge synthesises everything. You receive the IMAGE, the GROUND TRUTH, and the full trace.

Agent domains:
{_AGENT_DOMAINS}

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
  "role_adherence":                 {{"<agent>": <bool>, ...}},
  "hallucination_score":            {{"<agent>": <float 0-1>, ...}},   // INVERTED: higher is worse
  "visual_consistency_score":       {{"<agent>": <float 0-1>, ...}},
  "confidence_calibration_score":   {{"<agent>": <float 0-1>, ...}},
  "argument_quality_score":         {{"<agent>": <float 0-1>, ...}},
  "argument_quality_notes":         {{"<agent>": "<str>", ...}},
  "revision_justification_score":   {{"<agent>": <float 0-1>, ...}},
  "revision_justification_notes":   {{"<agent>": "<str>", ...}},
  "debate_contribution_score":      {{"<agent>": <float 0-1>, ...}},
  "debate_contribution_notes":      {{"<agent>": "<str>", ...}},
  "hallucination_examples":         {{"<agent>": [<str>, ...], ...}},  // up to 3 verbatim quotes per agent
  "moderator_pairing_quality_score": <float 0-1>,
  "moderator_pairing_notes":         "<str>",
  "judge_synthesis_quality":         <float 0-1>,
  "judge_synthesis_notes":           "<str>"
}}

SCORING ANCHORS (use consistently for every 0-1 field):
  0.00 = total failure on this dimension
  0.25 = poor, major problems
  0.50 = mixed/neutral midpoint (use this when agent did NOT debate for the three debate-specific scores)
  0.75 = good, minor weaknesses
  1.00 = flawless on this dimension

role_adherence (bool, per agent):
  true  = agent answered exclusively (or overwhelmingly) within their domain
  false = agent substantially answered outside their domain

hallucination_score (INVERTED, higher is worse, per agent):
  Cross-check against the IMAGE before scoring.
  0.00 = no invented claims; all specific claims verifiable in the image
  0.50 = one or two dubious specific claims (e.g. reading blurry text with high confidence)
  1.00 = agent fabricated specific details (named text, described objects not plausibly visible)

visual_consistency_score (per agent, image-grounded):
  Inspect the attached IMAGE directly and do NOT infer from agent text alone.
  Compare each concrete visual claim the agent made against what is visible.
  1.00 = every concrete visual claim is verifiable in the image
  0.75 = mostly verifiable; one minor claim is vague or slightly off
  0.50 = some claims verifiable, others vague or mildly off
  0.25 = several claims contradict or cannot be verified against the image
  0.00 = agent's evidence directly contradicts what is visible

argument_quality_score (per agent, ONLY meaningful when agent debated; otherwise 0.5):
  1.0 = each debate turn cited specific image evidence within the agent's domain; no repetition
  0.5 = plausible arguments but partially generic or one turn repeated prior argument
        (also use 0.5 when agent did NOT debate)
  0.0 = agent repeated the same argument verbatim across turns, produced no domain-relevant evidence,
        or argued outside their domain in debate

revision_justification_score (per agent, ONLY meaningful when agent debated; otherwise 0.5):
  IF the agent REVISED at any point:
    1.0 = revision followed specific opponent evidence that genuinely contradicts the prior position
    0.5 = revision occurred but trigger was vague or reasoning unconvincing
    0.0 = rubber-stamp revision (opponent just repeated themselves)
  IF the agent DID NOT revise:
    1.0 = agent clearly articulated why their evidence was stronger, engaged with opponent's points
    0.5 = maintained position but repetitive (also use 0.5 when agent did NOT debate)
    0.0 = stubborn restatement without engaging opponent's evidence

debate_contribution_score (per agent, ONLY meaningful when agent debated; otherwise 0.5):
  1.0 = agent surfaced new specific image observations IN DEBATE not present in R1
  0.5 = mostly reiterated R1 but framed more sharply (also use 0.5 when agent did NOT debate)
  0.0 = no new information beyond R1; just argued in circles

moderator_pairing_quality_score (image-level):
  Pairing priority rule (from the moderator's own system prompt):
    BEST: Linguistic or Regulatory vs. any disagreeing agent (hard constraints)
    GOOD: Botanics vs. any disagreeing agent (endemic species)
    AVOID: Landscape vs. Meta (both soft evidence)
  1.0 = moderator picked the highest-priority available pairing
  0.5 = pairings reasonable but missed a better option (also use 0.5 when no debate happened)
  0.0 = paired same-domain agents, already-agreeing agents, or chose Landscape vs. Meta when better was available

judge_synthesis_quality (image-level):
  Evidence hierarchy from the judge's own prompt (strongest to weakest):
    1. Concessions in debate (explicit revisions)
    2. Hard constraints: transcribed text, specific script, driving side, license plate format
    3. Specific evidence: endemic species, region-specific crops
    4. Soft evidence: general terrain, climate, soil colour
  1.0 = final reasoning explicitly follows this hierarchy
  0.5 = plausible but inconsistent application
  0.0 = contradicts the evidence or ignores a concession

hallucination_examples (per agent, image-grounded):
  Up to 3 verbatim quotes from THIS agent's R1 assessment or debate exchanges that are
  NOT supported by the image. A quote only counts if you can confirm by looking at the
  image that the claim is wrong or unsupported. Empty list if none. Quote the agent
  verbatim, do not paraphrase.

Output ONLY the JSON object. No prose, no markdown fences."""


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------

def _short(s: str, n: int = 500) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[:n] + " [...]"


def _format_r1(agent: str, assessment: dict | None) -> str:
    if not assessment:
        return f"### {agent}\n  (no Round 1 assessment)\n"
    cands = assessment.get("candidates") or []
    evid = assessment.get("evidence") or []
    lines = [f"### {agent}", "  Candidates:"]
    for c in cands[:5]:
        lines.append(
            f"    - {c.get('country', '?')} (conf={c.get('confidence', '?')}): "
            f"{_short(c.get('reasoning', ''), 280)}"
        )
    if evid:
        lines.append("  Evidence:")
        for e in evid[:6]:
            lines.append(f"    - {_short(str(e), 140)}")
    return "\n".join(lines) + "\n"


def _agent_debated(record: RunRecord, agent: str) -> bool:
    for p in record.debate.get("pairings", []):
        if agent in (p.get("agent_a"), p.get("agent_b")) and p.get("exchanges"):
            return True
    return False


def build_trace(record: RunRecord) -> str:
    """Build the per-image evaluation trace (all five agents + full debate)."""
    parts: list[str] = []

    parts.append("## Round 1 assessments (all five agents)\n")
    for agent in AGENT_NAMES:
        parts.append(_format_r1(agent, record.r1_assessments.get(agent)))

    parts.append("## Moderator decisions")
    decisions = record.debate.get("moderator_decisions") or []
    if decisions:
        for md in decisions:
            parts.append(
                f"  Round {md.get('debate_round', '?')}: "
                f"terminate={md.get('terminate', False)} "
                f"reason={md.get('termination_reason', '') or '-'}"
            )
            pairings = md.get("pairings_opened") or []
            if pairings:
                parts.append(f"    Pairings opened: {pairings}")
            contras = md.get("contradictions_found") or []
            if contras:
                parts.append(f"    Contradictions: {contras}")
            if md.get("reasoning"):
                parts.append(f"    Reasoning: {_short(md.get('reasoning', ''), 280)}")
    else:
        parts.append("  (no moderator decisions, no debate initiated)")
    parts.append("")

    parts.append("## Debate exchanges (all pairings)")
    pairings = record.debate.get("pairings") or []
    if pairings:
        for p in pairings:
            a = p.get("agent_a", "?")
            b = p.get("agent_b", "?")
            parts.append(
                f"  {a} ({p.get('agent_a_initial_position', '?')}) vs "
                f"{b} ({p.get('agent_b_initial_position', '?')}) "
                f", debate_round={p.get('debate_round', '?')}"
            )
            for ex in p.get("exchanges", []):
                revised = " [REVISED]" if ex.get("revised") else ""
                speaker = ex.get("agent_name", "?")
                parts.append(
                    f"    [{speaker}{revised}] pos={ex.get('position', '?')} "
                    f"conf={ex.get('confidence', '?')}"
                )
                parts.append(f"      arg: {_short(ex.get('argument', ''), 320)}")
                kev = ex.get("key_evidence") or []
                if kev:
                    parts.append(f"      key_evidence: {', '.join(str(k) for k in kev[:5])}")
    else:
        parts.append("  (no pairings)")
    parts.append("")

    parts.append("## Per-agent debated flag (derived from trace)")
    for agent in AGENT_NAMES:
        parts.append(f"  {agent}: agent_debated={_agent_debated(record, agent)}")
    parts.append("")

    parts.append("## Final judge reasoning")
    parts.append(f"  {_short(record.final_reasoning or '(none)', 800)}")

    correct = "YES" if record.is_correct else "NO"
    parts.append(
        f"\n## Ground truth\n  {record.truth_country_name} ({record.truth_country_code}) "
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
        f"  country: {record.truth_country_name} ({record.truth_country_code})\n"
        f"  coordinates: {record.truth_lat:.4f}, {record.truth_lon:.4f}\n"
    )
    trace = build_trace(record)
    text = (
        truth_block
        + "\n[STREET-VIEW IMAGE, what the agents saw]\n"
        + "(image attached above this prompt)\n\n"
        + "[DEBATE TRACE]\n"
        + trace
        + "\n\n[YOUR TASK]\n"
        + "Score ALL five agents and the moderator/judge orchestration against the rubric. "
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
    client: Any,
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
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=_MAX_TOKENS,
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

    agent_debated = {a: _agent_debated(record, a) for a in AGENT_NAMES}

    payload: dict = {
        "image_id": record.image_id,
        "truth_country": record.truth_country_name,
        "pred_country": record.pred_country,
        "is_correct": record.is_correct,
        "debate_happened": record.debate_happened,
        "debate_rounds": record.total_debate_rounds,
        "agent_debated": agent_debated,
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

def _load_file_list(file_list: Path) -> set[str]:
    """Read a file with one image filename or image_id per line.

    Accepts either bare image_ids (e.g. ``1NJsXTxIF9GGMDxC_1``) or full image
    filenames (e.g. ``1NJsXTxIF9GGMDxC_1.png``); the extension is stripped so
    the remaining stem matches ``RunRecord.image_id``.
    """
    ids: set[str] = set()
    with open(file_list) as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            stem = Path(name).stem if "." in name else name
            ids.add(stem)
    return ids


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
    skip_done: bool = True,
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

    if file_list is not None:
        wanted = _load_file_list(file_list)
        before = len(records)
        records = [r for r in records if r.image_id in wanted]
        print(f"[judge] file-list filter: {len(records)}/{before} records selected "
              f"(from {len(wanted)} ids in {file_list})")

    if skip_done:
        before = len(records)
        records = [r for r in records if not (judge_dir / f"{r.image_id}.json").exists()]
        skipped = before - len(records)
        if skipped:
            print(f"[judge] resume: skipping {skipped} images already judged "
                  f"(re-run with skip_done=False to force re-judge)")

    if limit:
        records = records[:limit]

    if not records:
        print("[judge] nothing to do.")
        return {}

    print(f"[judge] evaluating {len(records)} images (one call per image, model={_model})")

    counts = asyncio.run(
        _run_async(records, judge_dir, image_root, _model, _api_base, concurrency)
    )
    print(f"[judge] done: {counts}")
    return counts
