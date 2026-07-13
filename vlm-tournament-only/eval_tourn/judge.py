"""LLM-as-a-judge, Stage 2 evaluation.

Per the evaluation slide, the judge sees:
  • Ground truth (country code/name + coordinates), explicitly labelled
  • The street-view image
  • The full agent discussion trace (initial assessments, region decision,
    country-round assessments, hypothesis evaluations, tournament log,
    final answer)

It produces a structured verdict (Pydantic-enforced JSON) covering:
  • role_adherence, did each agent stay within its expertise?
  • argumentative_quality, Very Weak ... Very Strong per agent
  • constructive_synthesis, single bool for the whole run
  • cross_agent_influence, correct/incorrect change counts per agent
  • role_adherence_notes, one-line justification per agent
  • overall_verdict_notes, brief free-text summary

Resume-capable: skips images that already have a verdict file.
Concurrent (asyncio.Semaphore) so we can keep the GPU saturated.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from vlm_council.image_utils import encode_image
from vlm_council.llm import get_vlm
from eval_tourn.loader import AGENT_NAMES, RunRecord, load_run


# Pydantic schema

QualityScale = Literal["very_weak", "weak", "normal", "strong", "very_strong"]
AssertScale = Literal["very_low", "low", "medium", "high", "very_high"]
FailureReason = Literal[
    "judge_misjudgment",
    "missing_rag_refs",
    "ambiguous_evidence",
    "agent_misled",
    "not_applicable",
    "other",
]


class TournamentFailure(BaseModel):
    """Why did the truth lose its tournament match (if it was even in the pool)?

    ``not_applicable`` means: truth wasn't in the pool, OR the run was correct,
    so there was no failure to attribute.
    """

    truth_in_pool: bool = False
    truth_lost_to: str | None = None
    failure_match_round: str | None = None
    failure_reason: FailureReason = "not_applicable"
    failure_reasoning: str = ""
    counterfactual_winnable: bool = False


class JudgeVerdict(BaseModel):
    """Structured verdict produced for a single image.

    Backward-compatible: every new (v2) field has a default, so the older
    JSON shape from prior judge runs still validates.
    """

    # ── v1 fields (kept) ────────────────────────────────────────────────
    role_adherence: dict[str, bool]
    role_adherence_notes: dict[str, str]
    argumentative_quality: dict[str, QualityScale]
    constructive_synthesis: bool
    overall_verdict_notes: str = Field(max_length=400)

    # Legacy / deprecated, no longer requested in the prompt, kept
    # nullable so older judge_summary.json files still validate.
    assertiveness: dict[str, AssertScale] | None = None
    assertiveness_score: dict[str, float] = Field(default_factory=dict)
    overall_quality_score: float | None = None

    # ── v2 quantitative scores per agent (all in [0,1]) ─────────────────
    role_adherence_score: dict[str, float] = Field(default_factory=dict)
    argumentative_quality_score: dict[str, float] = Field(default_factory=dict)
    hallucination_score: dict[str, float] = Field(default_factory=dict)
    hallucination_examples: dict[str, list[str]] = Field(default_factory=dict)
    visual_consistency_score: dict[str, float] = Field(default_factory=dict)
    visual_consistency_notes: dict[str, str] = Field(default_factory=dict)
    confidence_calibration: dict[str, float] = Field(default_factory=dict)

    # ── v2 run-level ────────────────────────────────────────────────────
    tournament_failure: TournamentFailure = Field(default_factory=TournamentFailure)


# JSON schema for vLLM guided JSON (same pattern as tournament_judge.py)

def _judge_json_schema() -> dict:
    agent_keys = list(AGENT_NAMES)
    bool_map = {
        "type": "object",
        "properties": {a: {"type": "boolean"} for a in agent_keys},
        "required": agent_keys,
    }
    str_map = {
        "type": "object",
        "properties": {
            a: {"type": "string", "maxLength": 100} for a in agent_keys
        },
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
        "properties": {
            a: {"type": "number", "minimum": 0.0, "maximum": 1.0}
            for a in agent_keys
        },
        "required": agent_keys,
    }
    str_array_map = {
        "type": "object",
        "properties": {
            a: {"type": "array", "items": {"type": "string", "maxLength": 200},
                "maxItems": 3}
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
            "constructive_synthesis": {"type": "boolean"},
            "overall_verdict_notes": {"type": "string", "maxLength": 400},
            "role_adherence_score": score_map,
            "argumentative_quality_score": score_map,
            "hallucination_score": score_map,
            "hallucination_examples": str_array_map,
            "visual_consistency_score": score_map,
            "visual_consistency_notes": str_map,
            "confidence_calibration": score_map,
            "tournament_failure": {
                "type": "object",
                "properties": {
                    "truth_in_pool": {"type": "boolean"},
                    "truth_lost_to": {"type": ["string", "null"]},
                    "failure_match_round": {"type": ["string", "null"]},
                    "failure_reason": {
                        "type": "string",
                        "enum": [
                            "judge_misjudgment", "missing_rag_refs",
                            "ambiguous_evidence", "agent_misled",
                            "not_applicable", "other",
                        ],
                    },
                    "failure_reasoning": {"type": "string", "maxLength": 400},
                    "counterfactual_winnable": {"type": "boolean"},
                },
            },
        },
        "required": [
            "role_adherence",
            "role_adherence_notes",
            "argumentative_quality",
            "constructive_synthesis",
            "overall_verdict_notes",
        ],
    }


def _parse_judge_json(content: str | list) -> dict | None:
    """Extract a JSON object from a model response."""
    if isinstance(content, list):
        text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    else:
        text = content or ""
    text = text.strip()
    # Strip <think>...</think> if any
    m = re.search(r"<think>.*?</think>(.*)", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first balanced {…}
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
                    return json.loads(text[start:i + 1])
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


def _format_tournament(matches: list[dict]) -> str:
    if not matches:
        return ""
    lines = ["### Tournament bracket"]
    for m in matches:
        lines.append(
            f"  [{m.get('round_label', '?'):8s}] "
            f"{m.get('country_a', '?')} vs {m.get('country_b', '?')}  "
            f"→ winner: {m.get('winner', '?')}"
        )
        lines.append(f"      {_short(m.get('reasoning', ''), 500)}")
    return "\n".join(lines) + "\n"


def _format_rag_findings(findings: list[dict]) -> str:
    if not findings:
        return ""
    lines = ["### RAG pre-filter findings"]
    for f in findings[:25]:
        kind = f.get("kind", "?")
        country = f.get("country", "")
        detail = _short(f.get("detail", ""), 200)
        lines.append(f"  [{kind:18s}] {country}: {detail}")
    return "\n".join(lines) + "\n"


def build_trace(record: RunRecord) -> str:
    """Compose the full agent discussion trace as a single text block."""
    parts: list[str] = []
    pn = record.raw.get("progressive_narrowing") or {}
    parts.append("## Phase 1, Initial assessments (each agent independent)\n")
    for agent in AGENT_NAMES:
        parts.append(_format_assessment(agent, (record.assessments or {}).get(agent)))

    parts.append(
        f"## Region decision, path {record.path or '?'}\n"
        f"  region_consensus={pn.get('region_consensus')}, "
        f"confirmed_region={pn.get('confirmed_region', '')}\n"
        f"  proposed_regions={pn.get('proposed_regions', [])}\n"
        f"  reasoning: {_short(pn.get('region_decision_reasoning', ''), 600)}\n"
    )

    if record.country_assessments:
        parts.append("## Phase 2, Country-round assessments (Path B, region-constrained)\n")
        for agent in AGENT_NAMES:
            parts.append(_format_assessment(agent, record.country_assessments.get(agent)))

    parts.append(_format_hypothesis_evaluations(record.hypothesis_evaluations))
    parts.append(_format_rag_findings(record.rag_findings))
    parts.append(_format_tournament(record.tournament_log))

    parts.append(
        "## Final answer (after tournament)\n"
        f"  predicted country: {record.pred_country}\n"
        f"  predicted coordinates: {record.pred_lat}, {record.pred_lng}\n"
        f"  final reasoning: {_short(record.final_reasoning, 800)}\n"
    )
    return "\n".join(parts)


# Prompt construction

_SYSTEM = """You are an evaluation judge for a multi-agent geo-localization pipeline.

Five specialist agents, linguistic, landscape, botanics, regulatory, meta, analyse a street-view image and propose countries. A judge then narrows the candidates and runs a pairwise tournament. Your job is to score the QUALITY of that discussion against the IMAGE and the GROUND TRUTH (both shown to you), not to predict the country yourself.

Each agent's expertise:
  - linguistic: scripts, languages on signs, place-name etymology
  - landscape: terrain, vegetation, climate cues
  - botanics: plant species native ranges
  - regulatory: road signs, markings, infrastructure standards, driving side
  - meta: cross-domain knowledge, country distinguishing features

You must produce ONE JSON object. Every per-agent dict MUST have ALL FIVE keys:
linguistic, landscape, botanics, regulatory, meta.


===========================================================================
WHAT TO LOOK AT FOR EACH RUBRIC DIMENSION
===========================================================================

The trace contains five sections; each rubric field is graded against a
SPECIFIC subset. Do NOT mix sources, e.g. "role_adherence" is judged from
the agents' own assessments, not from the tournament outcome.

  role_adherence / role_adherence_notes
    SOURCE: Phase 1 + Phase 2 agent assessments only.
    QUESTION: Did this agent stay inside its declared expertise (see list
    above), or did it argue from a domain it does not own?

  argumentative_quality / argumentative_quality_score
    SOURCE: Phase 1 + Phase 2 evidence strings of THIS agent.
    QUESTION: Is the reasoning concrete, image-grounded, and internally
    consistent? Vague hedging or unsupported leaps lower the score.

  hallucination_score / hallucination_examples
    SOURCE: Compare each agent's evidence claims AGAINST THE IMAGE.
    QUESTION: Did the agent claim things that aren't actually visible
    (e.g. "eucalyptus visible" when there is none, "Cyrillic script
    on signs" when signs are Latin)? RAG-ref existence is NOT a
    hallucination signal, only image-vs-claim mismatch is.

  visual_consistency_score / visual_consistency_notes
    SOURCE: Same as hallucination but framed positively, does what the
    agent describes MATCH the image? An agent can be visually consistent
    while still being wrong about which country it is.

  confidence_calibration
    SOURCE: hypothesis_evaluations confidence labels (strongly_support /
    support / neutral / contradicts / strongly_contradicts) for THIS
    agent across all candidate countries, vs. the ground truth.
    QUESTION: Did the agent assign its highest confidence to the truth
    country (when correct) or appropriately hedge (when uncertain)?

  constructive_synthesis
    SOURCE: Region decision + tournament reasoning.
    QUESTION: Did the orchestration meaningfully USE the specialists, or
    was it a rubber-stamp of one agent's conclusion?

  tournament_failure
    SOURCE: tournament_log + rag_findings + final answer.
    Only meaningful when the truth was in the candidate pool but the
    final prediction was wrong. See FAILURE-REASON RULES below.


===========================================================================
SCORING ANCHORS, every [0,1] float MUST follow these reference points
===========================================================================

Use these five anchors. Pick the closest, then nudge ±0.05 if needed.
Do NOT spread scores randomly across the [0,1] range.

  0.00  total failure on this dimension. (e.g. agent argued only outside
        its domain; every claim contradicted the image; etc.)
  0.25  poor, major problems, mostly unhelpful but not zero signal.
  0.50  mixed, partial success, partial failure. NEUTRAL midpoint.
  0.75  good, solid execution with minor weaknesses.
  1.00  flawless on this dimension.

Specific dimension overrides:

  hallucination_score (INVERTED, higher is WORSE):
    0.00 = no invented claims at all.
    0.25 = one minor unsupported claim.
    0.50 = several unsupported claims OR one major fabrication.
    0.75 = pervasive invention (most evidence is unsupported).
    1.00 = the agent fabricated nearly everything it said.

  confidence_calibration:
    1.00 = high confidence on truth country, low on others (correct case).
    0.75 = right answer but over-hedged, OR wrong answer with humility.
    0.50 = neutral / no clear signal.
    0.25 = high confidence on a wrong country, no hedge for truth.
    0.00 = strongly_supports a wrong country AND strongly_contradicts truth.


===========================================================================
FAILURE-REASON RULES, apply IN ORDER, take the FIRST that matches
===========================================================================

Only fill tournament_failure when truth_in_pool=true AND prediction != truth.
Otherwise set failure_reason="not_applicable" and leave fields empty/false.

Decision tree (evaluate top-down, stop at first match):

  1. agent_misled
     PRECONDITION: Look at hypothesis_evaluations for the truth country.
     TRIGGER: Two or more specialists assigned `contradicts` or
     `strongly_contradicts` to the truth country, OR they collectively
     gave HIGHER confidence to the wrong winner than to the truth.
     MEANING: The specialists pointed away from truth; the tournament
     was set up to fail before it even started.

  2. missing_rag_refs
     PRECONDITION: agent_misled didn't fire.
     TRIGGER: In the tournament_log match where truth lost, the truth
     side had zero or far fewer verified RAG references than the winner.
     Check rag_findings for [tournament_match] entries and the match
     reasoning for "X has Y refs, Z has 0".
     MEANING: Asymmetric retrieval evidence, the tournament judge
     decided on ref count rather than visual judgment.

  3. judge_misjudgment
     PRECONDITION: 1+2 didn't fire.
     TRIGGER: The truth side had comparable RAG refs AND specialist
     support, AND the tournament reasoning shows the judge actually
     considered both, but still picked the wrong country with a
     conclusion the IMAGE does not support.
     MEANING: A defensible-on-paper decision that contradicts what's
     actually visible. The judge had what it needed and chose wrong.

  4. ambiguous_evidence
     PRECONDITION: 1+2+3 didn't fire.
     TRIGGER: The tournament reasoning explicitly acknowledges the
     match is close ("could be either", "both have similar bollards",
     "vegetation matches both"), AND from the IMAGE it really IS hard
     to tell. This is the bucket for "the council did its job but the
     image is genuinely under-determined".
     MEANING: Not the council's fault, the visual evidence is not
     enough to disambiguate.

  5. other
     None of the above. Briefly explain in failure_reasoning.

Tie-breaking note: ambiguous_evidence is the LAST resort, not the default.
If you can attribute the failure to a specific structural cause (1, 2, 3),
use that instead.

counterfactual_winnable: true ONLY if a smarter judge with the SAME inputs
could have picked truth. agent_misled + ambiguous_evidence usually imply
false. judge_misjudgment usually implies true.


===========================================================================
HALLUCINATION_EXAMPLES
===========================================================================

Short direct quotes (≤3 per agent) of agent CLAIMS that are not supported
by the image. Empty list when none. Quote the agent verbatim, do not
paraphrase.

EXAMPLE OUTPUT (use this as the structural template, fill in your own values):
{
  "role_adherence": {"linguistic": true, "landscape": true, "botanics": false, "regulatory": true, "meta": true},
  "role_adherence_notes": {
    "linguistic": "stayed within scripts/place-name analysis",
    "landscape": "valid terrain & climate cues",
    "botanics": "speculated about regulatory items, drifted from domain",
    "regulatory": "discussed road markings appropriately",
    "meta": "synthesized cross-domain hints correctly"
  },
  "argumentative_quality": {"linguistic": "strong", "landscape": "normal", "botanics": "weak", "regulatory": "strong", "meta": "very_strong"},
  "constructive_synthesis": true,
  "overall_verdict_notes": "brief 1-2 sentence summary",
  "role_adherence_score": {"linguistic": 0.9, "landscape": 0.85, "botanics": 0.4, "regulatory": 0.9, "meta": 0.95},
  "argumentative_quality_score": {"linguistic": 0.8, "landscape": 0.6, "botanics": 0.3, "regulatory": 0.8, "meta": 0.9},
  "hallucination_score": {"linguistic": 0.0, "landscape": 0.1, "botanics": 0.5, "regulatory": 0.0, "meta": 0.05},
  "hallucination_examples": {"linguistic": [], "landscape": [], "botanics": ["claimed eucalyptus visible, not present"], "regulatory": [], "meta": []},
  "visual_consistency_score": {"linguistic": 0.9, "landscape": 0.85, "botanics": 0.4, "regulatory": 1.0, "meta": 0.9},
  "visual_consistency_notes": {"linguistic": "scripts match", "landscape": "biome plausible", "botanics": "species identification not supported by image", "regulatory": "road markings as described", "meta": "ok"},
  "confidence_calibration": {"linguistic": 0.7, "landscape": 0.5, "botanics": 0.2, "regulatory": 0.9, "meta": 0.85},
  "tournament_failure": {
    "truth_in_pool": true,
    "truth_lost_to": "Poland",
    "failure_match_round": "final",
    "failure_reason": "missing_rag_refs",
    "failure_reasoning": "truth side had zero verified bollard refs while opponent had three",
    "counterfactual_winnable": true
  }
}

Allowed enum values:
  argumentative_quality: very_weak, weak, normal, strong, very_strong
  failure_reason:        judge_misjudgment, missing_rag_refs, ambiguous_evidence, agent_misled, not_applicable, other
                         (apply the FAILURE-REASON RULES decision tree above, ambiguous_evidence is the LAST resort, not the default)

Do NOT include any prose, markdown, or commentary outside the JSON. Do NOT use a different shape (e.g. nested per-agent objects with score/notes inside)."""


def build_messages(record: RunRecord, image_b64: str, image_mime: str) -> list:
    truth_block = (
        f"[GROUND TRUTH, known to you, hidden from the agents]\n"
        f"  country: {record.truth_country_name} ({record.truth_country_code.upper()})\n"
        f"  coordinates: {record.truth_lat:.4f}, {record.truth_lng:.4f}\n"
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
        + "evidence-grounded, cite phrases from the trace when assigning "
        + "role_adherence_notes. Output ONLY the JSON object, no prose."
    )
    msg = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
            {"type": "text", "text": text},
        ]
    )
    return [SystemMessage(content=_SYSTEM), msg]


# Runtime

def _resolve_image_path(record: RunRecord, image_root: Path | None) -> Path | None:
    """Find the image on the current host."""
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


def _build_openai_messages(record: RunRecord, image_b64: str, image_mime: str) -> list:
    """Same content as build_messages, but in raw OpenAI dict format."""
    truth_block = (
        f"[GROUND TRUTH, known to you, hidden from the agents]\n"
        f"  country: {record.truth_country_name} ({record.truth_country_code.upper()})\n"
        f"  coordinates: {record.truth_lat:.4f}, {record.truth_lng:.4f}\n"
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
        + "evidence-grounded, cite phrases from the trace when assigning "
        + "role_adherence_notes. Output ONLY the JSON object, no prose."
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


async def _judge_one(
    record: RunRecord,
    out_dir: Path,
    image_root: Path | None,
    client: AsyncOpenAI,
    model: str,
    response_format: dict,
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
            json.dump(
                {"image_id": record.image_id, "error": f"encode failed: {e}"},
                f, indent=2,
            )
        return record.image_id, "encode_error"

    messages = _build_openai_messages(record, b64, mime)

    async with sem:
        try:
            # Raw OpenAI Chat Completions call. Bypasses langchain's
            # _create_chat_result, which raises on finish_reason='length'.
            #
            # NOTE: We do NOT pass response_format / guided_json here. Both
            # paths trigger xgrammar guided decoding in vLLM, and gemma-4-31B
            # falls into infinite-whitespace loops on our nested-object schema.
            # Instead we steer purely via the system prompt (which contains a
            # concrete example) and parse + validate manually.
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=_MAX_JUDGE_TOKENS,
            )
        except Exception as e:
            with open(out_file, "w") as f:
                json.dump(
                    {"image_id": record.image_id, "error": f"llm failed: {e}"},
                    f, indent=2,
                )
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
        # Keep the raw parsed object, it's still useful for aggregation
        verdict_dump = parsed
        validation_err = str(e)[:500]
    else:
        validation_err = None

    payload = {
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
    model: str | None,
    api_base: str | None,
    concurrency: int,
) -> dict[str, int]:
    if model is not None:
        os.environ["VLM_JUDGE_LLM_MODEL"] = model
    if api_base is not None:
        os.environ["VLM_JUDGE_LLM_API_BASE"] = api_base

    # Reuse get_vlm just to derive the model + api_base (env-resolution lives there)
    llm = get_vlm("judge_llm")
    model_name = llm.model_name
    api_base_url = str(llm.openai_api_base)

    schema = _judge_json_schema()
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "JudgeVerdict",
            "schema": schema,
            "strict": False,
        },
    }

    client = AsyncOpenAI(api_key="EMPTY", base_url=api_base_url)

    sem = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        asyncio.create_task(
            _judge_one(r, out_dir, image_root, client, model_name, response_format, sem)
        )
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
) -> dict:
    out_dir = out / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_run(results, gt)
    if limit:
        records = records[:limit]

    counts = asyncio.run(
        _run_async(records, out_dir, image_root, model, api_base, concurrency)
    )
    print(f"[judge] done: {counts}")
    return counts
