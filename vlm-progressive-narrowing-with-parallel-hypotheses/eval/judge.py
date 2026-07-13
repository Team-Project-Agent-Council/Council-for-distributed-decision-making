"""LLM-as-a-judge, Stage 2 evaluation for VLM Council (Progressive Narrowing).

Per image, sends (ground truth, image, full discussion trace) to the judge VLM.
Pydantic-enforced verdict covers:

  Carried over from v12:
    role_adherence, did each agent stay within its expertise?
    argumentative_quality, Very Weak … Very Strong per agent
    hallucination_score, agent claims not supported by image [0=clean, 1=severe]
    hallucination_examples, verbatim quotes of unsupported claims
    visual_consistency_score, do agent descriptions match the image? [0,1]
    visual_consistency_notes, per-agent notes
    confidence_calibration, did agent rank truth country highest? [0,1]
    constructive_synthesis, bool: did judge meaningfully use all specialists?

  New for Progressive Narrowing:
    region_narrowing_quality, [0,1] was the region decision well-reasoned
                                given the evidence? (only meaningful for Path B;
                                for Path A this scores the consensus check)
    region_narrowing_notes, free-text explanation
    hypothesis_pool_quality, [0,1] were the generated country hypotheses
                                the right candidates for this image, regardless
                                of whether the correct answer was included?
    hypothesis_pool_notes, free-text explanation

Resume-capable: skips images that already have a verdict file.
Concurrent (asyncio.Semaphore).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from vlm_council.image_utils import encode_image
from vlm_council.llm import get_vlm
from eval.loader import AGENT_NAMES, RunRecord, load_run


# Pydantic schema

QualityScale = Literal["very_weak", "weak", "normal", "strong", "very_strong"]


class JudgeVerdict(BaseModel):
    """Structured verdict for a single image."""

    # Per-agent (all dicts must have all 5 agent keys)
    role_adherence: dict[str, bool]
    role_adherence_notes: dict[str, str]
    argumentative_quality: dict[str, QualityScale]
    hallucination_score: dict[str, float]
    hallucination_examples: dict[str, list[str]]
    visual_consistency_score: dict[str, float]
    visual_consistency_notes: dict[str, str]
    confidence_calibration: dict[str, float]

    # Run-level
    constructive_synthesis: bool
    overall_verdict_notes: str = Field(max_length=400)

    # New PN-specific
    region_narrowing_quality: float = Field(ge=0.0, le=1.0)
    region_narrowing_notes: str = Field(default="", max_length=300)
    hypothesis_pool_quality: float = Field(ge=0.0, le=1.0)
    hypothesis_pool_notes: str = Field(default="", max_length=300)


def _judge_json_schema() -> dict:
    agent_keys = list(AGENT_NAMES)
    bool_map = {
        "type": "object",
        "properties": {a: {"type": "boolean"} for a in agent_keys},
        "required": agent_keys,
    }
    str_map = {
        "type": "object",
        "properties": {a: {"type": "string", "maxLength": 100} for a in agent_keys},
        "required": agent_keys,
    }
    quality_map = {
        "type": "object",
        "properties": {
            a: {"type": "string", "enum": list(QualityScale.__args__)}
            for a in agent_keys
        },
        "required": agent_keys,
    }
    score_map = {
        "type": "object",
        "properties": {a: {"type": "number", "minimum": 0.0, "maximum": 1.0} for a in agent_keys},
        "required": agent_keys,
    }
    str_array_map = {
        "type": "object",
        "properties": {
            a: {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 3}
            for a in agent_keys
        },
        "required": agent_keys,
    }
    return {
        "type": "object",
        "properties": {
            "role_adherence": bool_map,
            "role_adherence_notes": str_map,
            "argumentative_quality": quality_map,
            "hallucination_score": score_map,
            "hallucination_examples": str_array_map,
            "visual_consistency_score": score_map,
            "visual_consistency_notes": str_map,
            "confidence_calibration": score_map,
            "constructive_synthesis": {"type": "boolean"},
            "overall_verdict_notes": {"type": "string", "maxLength": 400},
            "region_narrowing_quality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "region_narrowing_notes": {"type": "string", "maxLength": 300},
            "hypothesis_pool_quality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "hypothesis_pool_notes": {"type": "string", "maxLength": 300},
        },
        "required": [
            "role_adherence", "role_adherence_notes", "argumentative_quality",
            "hallucination_score", "hallucination_examples",
            "visual_consistency_score", "visual_consistency_notes",
            "confidence_calibration", "constructive_synthesis",
            "overall_verdict_notes", "region_narrowing_quality",
            "hypothesis_pool_quality",
        ],
    }


def _parse_judge_json(content: str | list) -> dict | None:
    if isinstance(content, list):
        text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    else:
        text = content or ""
    text = text.strip()
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


# Trace formatting

def _short(s: str, n: int = 800) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[:n] + " […]"


def _format_assessment(name: str, a: dict | None) -> str:
    if not a:
        return f"### {name}\n  (no assessment recorded)\n"
    cands = a.get("candidates") or []
    evidence = a.get("evidence") or []
    lines = [f"### {name}"]
    lines.append("Candidates:")
    for i, c in enumerate(cands, 1):
        lines.append(
            f"  {i}. {c.get('country', '?')}  "
            f"(confidence={c.get('confidence', '?')}): {_short(c.get('reasoning', ''), 400)}"
        )
    if evidence:
        lines.append("Evidence (observable clues cited):")
        for e in evidence[:8]:
            lines.append(f"  - {_short(e, 200)}")
    return "\n".join(lines) + "\n"


def _format_hypothesis_evaluations(evs: list[dict]) -> str:
    if not evs:
        return ""
    by_hyp: dict[str, list[dict]] = {}
    for ev in evs:
        by_hyp.setdefault(ev.get("hypothesis_id", "?"), []).append(ev)
    lines = ["### Hypothesis evaluations"]
    for hid, group in by_hyp.items():
        lines.append(f"  Hypothesis: {hid}")
        for ev in group:
            lines.append(
                f"    [{ev.get('agent_name', '?'):11s}] "
                f"{ev.get('confidence', '?'):22s}, "
                f"{_short(ev.get('reasoning', ''), 280)}"
            )
    return "\n".join(lines) + "\n"


def build_trace(record: RunRecord) -> str:
    parts: list[str] = []
    pn = record.raw.get("progressive_narrowing") or {}

    parts.append("## Phase 1, Initial assessments (each agent independent)\n")
    for agent in AGENT_NAMES:
        parts.append(_format_assessment(agent, (record.assessments or {}).get(agent)))

    path = record.path or "?"
    parts.append(
        f"## Region decision, path {path}\n"
        f"  region_consensus={pn.get('region_consensus')}, "
        f"confirmed_region={pn.get('confirmed_region', '')}\n"
        f"  proposed_regions={pn.get('proposed_regions', [])}\n"
        f"  reasoning: {_short(pn.get('region_decision_reasoning', ''), 600)}\n"
    )

    if record.country_assessments:
        parts.append("## Phase 2, Country-round assessments (Path B, region-constrained)\n")
        for agent in AGENT_NAMES:
            parts.append(_format_assessment(agent, record.country_assessments.get(agent)))

    # Active country hypotheses
    if record.active_hypotheses:
        parts.append("## Country hypotheses presented for evaluation\n")
        for h in record.active_hypotheses:
            parts.append(f"  - {h.get('value', '?')} (id={h.get('hypothesis_id', '?')})")
        parts.append("")

    parts.append(_format_hypothesis_evaluations(record.hypothesis_evaluations))

    parts.append(
        "## Final answer\n"
        f"  predicted country: {record.pred_country}\n"
        f"  predicted coordinates: {record.pred_lat}, {record.pred_lng}\n"
        f"  final reasoning: {_short(record.final_reasoning, 800)}\n"
    )
    return "\n".join(parts)


_SYSTEM = """You are an evaluation judge for a multi-agent geo-localization pipeline called VLM Council.

Five specialist agents, linguistic, landscape, botanics, regulatory, meta, analyse a street-view image and propose country candidates. A judge (separate LLM) then:
  1. Checks whether the agents agree on a world region (region consensus).
  2. If no consensus: generates region hypotheses, has all agents evaluate them, decides a region, then has agents do a constrained country assessment (Path B).
  3. If consensus: skips directly to country hypotheses (Path A).
  4. Generates country hypotheses from the assessments, has all agents evaluate them, and makes a final country decision via weighted scoring.

Agent expertise:
  - linguistic:  scripts, languages on signs, place-name etymology
  - landscape:   terrain, vegetation, climate cues
  - botanics:    plant species native ranges
  - regulatory:  road signs, markings, infrastructure standards, driving side
  - meta:        cross-domain synthesis, country distinguishing features

You must produce ONE JSON object. Every per-agent dict MUST have ALL FIVE keys:
linguistic, landscape, botanics, regulatory, meta.


===========================================================================
RUBRIC DIMENSIONS, what to look at and how to score each
===========================================================================

  role_adherence / role_adherence_notes
    SOURCE: Phase 1 + Phase 2 agent assessments only.
    QUESTION: Did this agent stay inside its declared expertise, or did it
    argue from a domain it does not own?

  argumentative_quality
    SOURCE: Phase 1 + Phase 2 evidence strings of THIS agent.
    QUESTION: Is the reasoning concrete, image-grounded, internally consistent?
    Vague hedging or unsupported leaps lower the score.

  hallucination_score (INVERTED, higher is WORSE)
    SOURCE: Compare each agent's evidence claims AGAINST THE IMAGE.
    QUESTION: Did the agent claim things not actually visible in the image?
    (e.g. "eucalyptus visible" when there is none, "Cyrillic script on signs"
    when signs are Latin). RAG references are NOT hallucinations, only
    image-vs-claim mismatch counts.

  hallucination_examples
    Short verbatim quotes (≤3 per agent) of agent CLAIMS not supported by the
    image. Empty list when none.

  visual_consistency_score / visual_consistency_notes
    SOURCE: Same as hallucination but framed positively, does what the agent
    describes MATCH the image? An agent can be visually consistent while still
    being wrong about which country it is.

  confidence_calibration
    SOURCE: hypothesis_evaluations confidence labels (strongly_support /
    support / neutral / contradicts / strongly_contradicts) for THIS agent
    across all country hypotheses, vs. the ground truth.
    QUESTION: Did the agent assign its highest confidence to the truth country
    (when the truth was in the pool) or appropriately hedge?

  constructive_synthesis
    SOURCE: Region decision + country decision reasoning.
    QUESTION: Did the judge meaningfully integrate and weigh the specialists'
    evidence, or was it a rubber-stamp of the plurality vote?

  region_narrowing_quality  [0,1]
    SOURCE: Region decision block + the agents' initial assessments.
    For PATH B: Was the region decision well-reasoned given the multi-agent
    evidence? Did the judge correctly identify the dominant region signal, or
    did it get confused by outlier agents? Score the quality of the reasoning
    process, not just whether it happened to be correct.
    For PATH A (consensus): Was the consensus check sound? Did it correctly
    confirm that all agents agreed, and was the confirmed region correct?
    0.00 = completely unsound / wrong for obvious reasons
    0.25 = poor, major reasoning gaps
    0.50 = acceptable but shallow
    0.75 = solid reasoning with minor gaps
    1.00 = exemplary, well-anchored in specific agent evidence

  region_narrowing_notes
    1-2 sentence explanation of region_narrowing_quality.

  hypothesis_pool_quality  [0,1]
    SOURCE: The "Country hypotheses presented for evaluation" block + the image.
    QUESTION: Were the generated country hypotheses the right set of plausible
    candidates for this specific image, given the visual evidence? Look at the
    image and judge whether a knowledgeable human would have included roughly
    these countries. The truth country being in the pool is a POSITIVE signal
    but NOT required for a high score, the question is about the plausibility
    of the pool as a whole.
    0.00 = pool is completely implausible for the image (wrong continent, etc.)
    0.25 = pool has one or two plausible candidates but many implausible ones
    0.50 = mixed, roughly half plausible
    0.75 = pool is mostly plausible with one or two questionable entries
    1.00 = all candidates are visually plausible and diverse within the region

  hypothesis_pool_notes
    1-2 sentence explanation of hypothesis_pool_quality.


===========================================================================
SCORING ANCHORS for [0,1] float scores
===========================================================================

  0.00  total failure on this dimension
  0.25  poor, major problems, mostly unhelpful
  0.50  mixed, partial success, partial failure (NEUTRAL midpoint)
  0.75  good, solid with minor weaknesses
  1.00  flawless on this dimension

  hallucination_score is INVERTED (higher = worse):
    0.00 = no invented claims
    0.25 = one minor unsupported claim
    0.50 = several unsupported OR one major fabrication
    0.75 = pervasive invention
    1.00 = agent fabricated nearly everything

  confidence_calibration:
    1.00 = high confidence on truth country, low on others (correct case)
    0.75 = right answer but over-hedged, OR wrong with humility
    0.50 = neutral / no clear signal
    0.25 = high confidence on wrong country, no hedge for truth
    0.00 = strongly_supports wrong AND strongly_contradicts truth


===========================================================================
EXAMPLE OUTPUT (fill in your own values)
===========================================================================

{
  "role_adherence": {"linguistic": true, "landscape": true, "botanics": false, "regulatory": true, "meta": true},
  "role_adherence_notes": {
    "linguistic": "stayed within scripts/place-name analysis",
    "landscape": "valid terrain & climate cues",
    "botanics": "speculated about road signs, outside domain",
    "regulatory": "discussed road markings appropriately",
    "meta": "synthesized cross-domain hints correctly"
  },
  "argumentative_quality": {"linguistic": "strong", "landscape": "normal", "botanics": "weak", "regulatory": "strong", "meta": "very_strong"},
  "hallucination_score": {"linguistic": 0.0, "landscape": 0.1, "botanics": 0.5, "regulatory": 0.0, "meta": 0.05},
  "hallucination_examples": {"linguistic": [], "landscape": [], "botanics": ["claimed eucalyptus visible, not present"], "regulatory": [], "meta": []},
  "visual_consistency_score": {"linguistic": 0.9, "landscape": 0.85, "botanics": 0.4, "regulatory": 1.0, "meta": 0.9},
  "visual_consistency_notes": {"linguistic": "scripts match image", "landscape": "biome plausible", "botanics": "species not supported by image", "regulatory": "road markings as described", "meta": "ok"},
  "confidence_calibration": {"linguistic": 0.7, "landscape": 0.5, "botanics": 0.2, "regulatory": 0.9, "meta": 0.85},
  "constructive_synthesis": true,
  "overall_verdict_notes": "Brief 1-2 sentence run summary.",
  "region_narrowing_quality": 0.75,
  "region_narrowing_notes": "Judge correctly identified Europe from 4/5 agents; one outlier (botanics→Asia) was appropriately discounted.",
  "hypothesis_pool_quality": 0.75,
  "hypothesis_pool_notes": "Pool included Germany, Austria, Czech Republic, all visually plausible. Hungary felt like a stretch given the architecture."
}

Allowed enum values for argumentative_quality:
  very_weak, weak, normal, strong, very_strong

Do NOT include any prose, markdown, or commentary outside the JSON."""


def build_messages(record: RunRecord, image_b64: str, image_mime: str) -> list:
    truth_block = (
        f"[GROUND TRUTH, known to you, hidden from the agents]\n"
        f"  country: {record.truth_country_name} ({record.truth_country_code.upper()})\n"
        f"  coordinates: {record.truth_lat:.4f}, {record.truth_lng:.4f}\n"
        f"  truth in hypothesis pool: {record.truth_in_hypothesis_pool}\n"
    )
    trace = build_trace(record)
    text = (
        truth_block
        + "\n[STREET-VIEW IMAGE, what the agents saw]\n"
        + "(image attached above this prompt)\n\n"
        + "[AGENT DISCUSSION TRACE]\n"
        + trace
        + "\n[YOUR TASK]\n"
        + "Score the discussion against the rubric. Be specific and "
        + "evidence-grounded. Output ONLY the JSON object, no prose."
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
                {"type": "text", "text": text},
            ],
        },
    ]


# Runtime

def _resolve_image_path(record: RunRecord, image_root: Path | None) -> Path | None:
    raw = record.image_path
    if image_root is not None:
        cand = image_root / Path(raw).name
        if cand.exists():
            return cand
    p = Path(raw)
    if p.exists():
        return p
    if image_root is not None:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            cand = image_root / f"{record.image_id}{ext}"
            if cand.exists():
                return cand
    return None


_MAX_JUDGE_TOKENS = int(os.environ.get("VLM_JUDGE_MAX_TOKENS", "2500"))


async def _judge_one(
    record: RunRecord,
    out_dir: Path,
    image_root: Path | None,
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
) -> tuple[str, str]:
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
            json.dump({"image_id": record.image_id, "error": f"encode failed: {e}"}, f, indent=2)
        return record.image_id, "encode_error"

    messages = build_messages(record, b64, mime)

    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=_MAX_JUDGE_TOKENS,
            )
        except Exception as e:
            with open(out_file, "w") as f:
                json.dump({"image_id": record.image_id, "error": f"llm failed: {e}"}, f, indent=2)
            return record.image_id, "llm_error"

    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = choice.finish_reason

    parsed = _parse_judge_json(content)
    if parsed is None:
        with open(out_file, "w") as f:
            json.dump(
                {"image_id": record.image_id,
                 "error": "could not parse judge JSON",
                 "finish_reason": finish_reason,
                 "raw": content[:2000]},
                f, indent=2,
            )
        return record.image_id, "parse_error"

    try:
        verdict = JudgeVerdict.model_validate(parsed)
        verdict_dump = verdict.model_dump()
    except ValidationError as e:
        verdict_dump = parsed
        validation_err = str(e)[:500]
    else:
        validation_err = None

    payload = {
        "image_id": record.image_id,
        "truth_country": record.truth_country_name,
        "pred_country": record.pred_country,
        "is_correct": record.is_correct,
        "path": record.path,
        "truth_in_hypothesis_pool": record.truth_in_hypothesis_pool,
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
    model: str | None,
    api_base: str | None,
    concurrency: int,
) -> dict[str, int]:
    if model is not None:
        os.environ["VLM_JUDGE_LLM_MODEL"] = model
    if api_base is not None:
        os.environ["VLM_JUDGE_LLM_API_BASE"] = api_base

    llm = get_vlm("judge_llm")
    model_name = llm.model_name
    api_base_url = str(llm.openai_api_base)

    client = AsyncOpenAI(api_key="EMPTY", base_url=api_base_url)
    sem = asyncio.Semaphore(max(1, concurrency))

    tasks = [
        asyncio.create_task(_judge_one(r, out_dir, image_root, client, model_name, sem))
        for r in records
    ]
    counts: dict[str, int] = {}
    for fut in asyncio.as_completed(tasks):
        image_id, status = await fut
        counts[status] = counts.get(status, 0) + 1
        print(f"[judge] {image_id}: {status}")
    return counts


def run(
    *,
    results: Path,
    gt: Path,
    out: Path,
    image_root: Path | None,
    model: str | None,
    api_base: str | None,
    concurrency: int,
    limit: int | None = None,
    file_list: Path | None = None,
) -> dict:
    out_dir = out / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    counts = asyncio.run(
        _run_async(records, out_dir, image_root, model, api_base, concurrency)
    )
    print(f"[judge] done: {counts}")
    return counts
