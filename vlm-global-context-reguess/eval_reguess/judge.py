"""LLM-as-judge evaluation for the Global Context Re-guess approach.

The judge receives the Street View image plus the full R1+R2 trace from all
five specialist agents and the final Judge reasoning, then produces ONE
structured verdict per image. Per-agent fields become ``dict[agent_name, value]``;
image-level fields (e.g. judge synthesis) stay scalar.

Resume-capable: skips images where ``<image_id>.json`` already exists.
Async via ``asyncio.Semaphore``. Reads VLM_JUDGE_LLM_MODEL and
VLM_JUDGE_LLM_API_BASE env vars.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from eval_reguess.loader import AGENT_NAMES, RunRecord, load_run
from vlm_council.image_utils import encode_image


# ── Pydantic verdict model ────────────────────────────────────────────────────


class JudgeVerdict(BaseModel):
    """Per-image verdict. Per-agent fields are dicts keyed by agent name."""

    # Per-agent
    role_adherence: dict[str, bool] = Field(default_factory=dict)
    hallucination_score: dict[str, float] = Field(default_factory=dict)
    visual_consistency_score: dict[str, float] = Field(default_factory=dict)
    confidence_calibration_score: dict[str, float] = Field(default_factory=dict)
    round2_improvement: dict[str, float] = Field(default_factory=dict)
    round2_improvement_notes: dict[str, str] = Field(default_factory=dict)
    role_leakage_score: dict[str, float] = Field(default_factory=dict)
    role_leakage_notes: dict[str, str] = Field(default_factory=dict)
    hallucination_examples: dict[str, list[str]] = Field(default_factory=dict)

    # Image-level
    judge_synthesis_quality: float = 0.5
    judge_synthesis_notes: str = ""


# ── System prompt ──────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are an evaluation judge for a multi-agent geo-localization pipeline using the "Global Context Re-guess" approach.

Five specialist agents (linguistic, landscape, botanics, regulatory, meta) each analyzed a Google Street View image independently in Round 1, then saw all other agents' Round 1 results and re-evaluated in Round 2. A Judge made the final decision using all Round 1 + Round 2 evidence. You receive the IMAGE, the GROUND TRUTH, and the full trace.

Each agent's expertise:
  - linguistic: scripts, languages on signs, place-name text, writing systems
  - landscape: terrain, vegetation, climate cues, soil type, topography
  - botanics: plant species, native ranges, seasonal vegetation patterns
  - regulatory: road signs, road markings, infrastructure standards, driving side
  - meta: meta-level patterns, camera generation, Google car coverage quirks, cross-domain synthesis

IMAGE-GROUNDED FIELDS, inspect the attached image directly before scoring these:
  - hallucination_score      (compare each agent's specific claims against what is actually visible)
  - hallucination_examples   (a quote only counts if the image does not support it)
  - visual_consistency_score (does the agent's evidence match what you see in the image?)
Do NOT infer these from agent text or trace alone. Cross-check every specific
visual claim, text on signs, species, road infrastructure, terrain features, against the image before scoring.

You must output ONE JSON object per image. Every per-agent dict MUST have ALL FIVE keys:
linguistic, landscape, botanics, regulatory, meta.

JSON schema:
{
  "role_adherence":                {"<agent>": <bool>, ...},
  "hallucination_score":           {"<agent>": <float 0-1>, ...},
  "visual_consistency_score":      {"<agent>": <float 0-1>, ...},
  "confidence_calibration_score":  {"<agent>": <float 0-1>, ...},
  "round2_improvement":            {"<agent>": <float 0-1>, ...},
  "round2_improvement_notes":      {"<agent>": "<str>", ...},
  "role_leakage_score":            {"<agent>": <float 0-1>, ...},
  "role_leakage_notes":            {"<agent>": "<str>", ...},
  "hallucination_examples":        {"<agent>": [<str>, ...], ...},
  "judge_synthesis_quality":       <float 0-1>,
  "judge_synthesis_notes":         "<str>"
}

=== SCORING RUBRICS ===

For visual_consistency_score and hallucination_examples, inspect the attached
image directly and do NOT infer from agent text alone.

role_adherence (bool, per agent):
  true = agent only argued from evidence within their domain
  false = agent drifted outside their domain (e.g. linguistic agent discussing road markings)

hallucination_score (0=clean, 1=severe, per agent):
  Cross-check against the IMAGE before scoring.
  0.00 = no invented claims; all specific claims verifiable in the image
  0.25 = one minor unsupported claim
  0.50 = several unsupported claims OR one major fabrication
  0.75 = pervasive invention
  1.00 = agent fabricated nearly everything

visual_consistency_score (0-1, higher=better, per agent, image-grounded):
  Inspect the attached IMAGE directly and do NOT infer from agent text alone.
  Compare each concrete visual claim the agent made against what is visible.
  1.00 = every concrete visual claim is verifiable in the image
  0.75 = mostly verifiable; one minor claim is vague or slightly off
  0.50 = some claims verifiable, others vague or mildly off
  0.25 = several claims contradict or cannot be verified against the image
  0.00 = agent's evidence directly contradicts what is visible

confidence_calibration_score (0-1, higher=better, per agent):
  1.00 = high confidence on correct country, appropriate hedging on others
  0.75 = correct answer but over-hedged, OR wrong answer with appropriate humility
  0.50 = neutral / no clear signal
  0.25 = high confidence on wrong country, no hedging
  0.00 = strongly asserts wrong country, dismisses correct one

round2_improvement (0-1, higher=better, per agent):
  1.00 = genuine synthesis of cross-agent evidence, refined reasoning
  0.50 = acknowledged others but didn't substantively change reasoning
  0.00 = rubber-stamped others OR just copy-pasted Round 1

role_leakage_score (0-1, lower=better, per agent, specialist agents only):
  Did this agent import another domain's evidence into their R2 reasoning?
  0.00 = clean: stayed entirely in their own domain
  0.50 = minor leakage: one or two cross-domain points alongside own reasoning
  1.00 = heavy leakage: R2 reasons primarily using another domain's evidence

role_leakage_notes (str, per agent):
  Name the specific cross-domain content that leaked. "none" if 0.0.

judge_synthesis_quality (0-1, image-level):
  1.00 = explicit elimination of wrong countries + evidence weighting + clear chain
  0.75 = good reasoning but one gap
  0.50 = plausible but vague
  0.25 = weak justification
  0.00 = conclusion unsupported, contradicts agents, or circular

hallucination_examples (per agent, image-grounded):
  Up to 3 verbatim quotes from THIS agent's assessments that are NOT supported by the
  image. A quote only counts if you can confirm by looking at the image that the claim
  is wrong or unsupported. Empty list if none. Quote the agent verbatim, do not
  paraphrase.

Output ONLY the JSON object. No prose, no markdown, no commentary outside the JSON."""


# ── Trace builder ─────────────────────────────────────────────────────────────


def _short(s: str, n: int = 600) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[:n] + " [...]"


def _format_assessment(assessment: dict | None) -> str:
    if not assessment:
        return f"  (no assessment)\n"
    cands = assessment.get("candidates") or []
    evidence = assessment.get("evidence") or []
    lines = ["  Candidates:"]
    for i, c in enumerate(cands, 1):
        lines.append(
            f"    {i}. {c.get('country', '?')} "
            f"(confidence={c.get('confidence', '?')}): "
            f"{_short(c.get('reasoning', ''), 350)}"
        )
    if evidence:
        lines.append("  Evidence:")
        for e in evidence[:8]:
            lines.append(f"    - {_short(str(e), 150)}")
    return "\n".join(lines) + "\n"


def build_trace(record: RunRecord) -> str:
    """Per-image trace covering all five agents (R1 + R2) plus the final judge."""
    parts: list[str] = []

    for agent in AGENT_NAMES:
        parts.append(f"=== {agent} ===\n")

        r1 = (record.r1_assessments or {}).get(agent)
        parts.append("--- Round 1 ---")
        parts.append(_format_assessment(r1))

        r2 = (record.r2_assessments or {}).get(agent)
        parts.append("--- Round 2 (after seeing other agents' R1) ---")
        parts.append(_format_assessment(r2))

        if r1 and r2:
            r1_cands = r1.get("candidates") or []
            r2_cands = r2.get("candidates") or []
            r1_top = r1_cands[0].get("country", "?") if r1_cands else "n/a"
            r2_top = r2_cands[0].get("country", "?") if r2_cands else "n/a"
            changed = "CHANGED" if r1_top != r2_top else "SAME"
            parts.append(f"  Top-1 change: {r1_top} → {r2_top} ({changed})\n")

    parts.append("=== Final Judge reasoning ===")
    parts.append(_short(record.final_reasoning or "(none)", 1200))
    parts.append("")

    parts.append("=== Ground truth (hidden from the agents) ===")
    parts.append(f"  Actual country: {record.truth_country_name} ({record.truth_country_code.upper()})")
    parts.append(f"  Actual coordinates: {record.truth_lat:.4f}, {record.truth_lon:.4f}")
    parts.append(f"  Predicted country: {record.pred_country}")
    parts.append(f"  Prediction correct: {record.is_correct}")
    parts.append("")

    return "\n".join(parts)


# ── JSON parsing ───────────────────────────────────────────────────────────────


def _parse_json(content: str) -> dict | None:
    if not content:
        return None
    text = content.strip()
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


# ── Image path resolution ──────────────────────────────────────────────────────


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


# ── Async judge runner ─────────────────────────────────────────────────────────


_MAX_JUDGE_TOKENS = int(os.environ.get("VLM_JUDGE_MAX_TOKENS", "2500"))


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
        + "[RE-GUESS TRACE, R1 + R2 for every agent + final judge reasoning]\n"
        + trace
        + "\n[YOUR TASK]\n"
        + "Score ALL five agents (and the judge's synthesis) against the rubric. "
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
                max_tokens=_MAX_JUDGE_TOKENS,
            )
        except Exception as e:
            with open(out_file, "w") as f:
                json.dump({"image_id": record.image_id,
                           "error": f"llm failed: {e}"}, f, indent=2)
            return record.image_id, "llm_error"

    choice = response.choices[0]
    content = (choice.message.content or "")
    finish_reason = choice.finish_reason

    parsed = _parse_json(content)
    if parsed is None:
        with open(out_file, "w") as f:
            json.dump({"image_id": record.image_id,
                       "error": "could not parse JSON",
                       "finish_reason": finish_reason,
                       "raw": content[:2000]}, f, indent=2)
        return record.image_id, "parse_error"

    try:
        verdict = JudgeVerdict.model_validate(parsed)
        verdict_dump = verdict.model_dump()
        validation_err = None
    except ValidationError as e:
        verdict_dump = parsed
        validation_err = str(e)[:500]

    payload: dict = {
        "image_id": record.image_id,
        "truth_country": record.truth_country_name,
        "pred_country": record.pred_country,
        "is_correct": record.is_correct,
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
    client = AsyncOpenAI(api_key="EMPTY", base_url=api_base)
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

    resolved_model = (
        model
        or os.environ.get("VLM_JUDGE_LLM_MODEL")
        or os.environ.get("VLM_MODEL")
        or "gpt-4o"
    )
    resolved_base = (
        api_base
        or os.environ.get("VLM_JUDGE_LLM_API_BASE")
        or os.environ.get("VLM_API_BASE")
        or "https://api.openai.com/v1"
    )

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

    print(f"[judge] evaluating {len(records)} images (one call per image, "
          f"model={resolved_model})")

    counts = asyncio.run(
        _run_async(records, judge_dir, image_root, resolved_model, resolved_base, concurrency)
    )
    print(f"[judge] done: {counts}")
    return counts
