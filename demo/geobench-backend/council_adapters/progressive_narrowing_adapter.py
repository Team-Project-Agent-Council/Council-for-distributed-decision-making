"""Adapter that wraps the vendored VLM-Council LangGraph pipeline so the
GeoBench backend can drive it from a /api/demo/run endpoint and emit SSE-style
progress events.

We don't reach inside the graph — we monkey-patch thin tracing shims around
the vendored module's internal `_eval_agent` closure (only reachable via the
node functions). To get per-agent streaming we instead patch the agent
modules' `evaluate_hypotheses` directly and emit the raw evaluations after
each per-agent return.

For everything else we use LangGraph's `astream(stream_mode="updates")` and
key off the node name in the chunk to decide which SSE event to emit.

The adapter is **stateless** — it's instantiated per run by demo_service.
Concurrent runs are isolated via the temp file path + the per-run callback
closure passed by the service layer.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

# The vendored `vlm_council` package uses internal absolute imports
# (`from vlm_council.state import …`), so we need to prepend the
# directory *containing* vlm_council/, not backend/ itself.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_ROOT = _BACKEND_ROOT / "vendor"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))


EventEmitter = Callable[[str, dict[str, Any]], Awaitable[None]]
"""Async function `(event_type, data) -> None` injected by the demo service."""


# Kept as bool for backwards compatibility with the HypothesisMatrix UI,
# which still expects a binary "supports" flag alongside the 5-level scale.
EVAL_CONFIDENCE_TO_SUPPORT = {
    "strongly_support": True,
    "support": True,
    "neutral": False,
    "contradicts": False,
    "strongly_contradicts": False,
}


_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] | None = None


def _load_centroids() -> dict[str, tuple[float, float]]:
    global _COUNTRY_CENTROIDS
    if _COUNTRY_CENTROIDS is not None:
        return _COUNTRY_CENTROIDS
    path = _BACKEND_ROOT / "data" / "country_centroids.json"
    with open(path) as f:
        raw = json.load(f)
    _COUNTRY_CENTROIDS = {k: (v[0], v[1]) for k, v in raw.items()}
    return _COUNTRY_CENTROIDS


def lookup_country_latlng(country: str) -> tuple[float, float] | None:
    """Resolve a country name to a representative lat/lng.

    Falls back to fuzzy lowercase + trimmed match before giving up so common
    capitalisation differences don't drop a guess off the map.
    """
    centroids = _load_centroids()
    if country in centroids:
        return centroids[country]
    key = country.strip().lower()
    for name, coords in centroids.items():
        if name.lower() == key:
            return coords
    return None


# Different specialists (and Gemma across calls) refer to the same country by
# different strings. Without normalisation these end up as duplicate country
# hypotheses ("USA" and "United States" evaluated in parallel).
_COUNTRY_ALIASES: dict[str, str] = {
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "us": "United States",
    "united states of america": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "south korea": "South Korea",
    "korea, south": "South Korea",
    "korea (south)": "South Korea",
    "republic of korea": "South Korea",
    "north korea": "North Korea",
    "korea, north": "North Korea",
    "korea (north)": "North Korea",
    "czech republic": "Czechia",
    "russian federation": "Russia",
    "myanmar (burma)": "Myanmar",
    "burma": "Myanmar",
    "the netherlands": "Netherlands",
    "holland": "Netherlands",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "cape verde": "Cabo Verde",
    "swaziland": "Eswatini",
    "east timor": "Timor-Leste",
    "vatican": "Vatican City",
    "holy see": "Vatican City",
    "u.a.e.": "United Arab Emirates",
    "uae": "United Arab Emirates",
    "p.r. china": "China",
    "prc": "China",
    "people's republic of china": "China",
    "roc": "Taiwan",
    "republic of china": "Taiwan",
    "chinese taipei": "Taiwan",
}


def _normalise_country(name: str) -> str:
    """Return the canonical country name for `name` (case-insensitive lookup)."""
    if not name:
        return name
    key = name.strip().lower()
    return _COUNTRY_ALIASES.get(key, name.strip())


_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


def _temp_image_path(image_bytes: bytes, mime: str) -> Path:
    ext = _MIME_TO_EXT.get(mime.lower(), ".jpg")
    tmp = tempfile.NamedTemporaryFile(prefix="vlm-demo-", suffix=ext, delete=False)
    try:
        tmp.write(image_bytes)
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


async def run_progressive_narrowing(
    image_bytes: bytes,
    mime: str,
    emit: EventEmitter,
) -> dict[str, Any]:
    """Drive the VLM-Council graph and emit progress events.

    Yields events via `emit(event_type, data)`. The emitter is responsible for
    queuing those into the SSE stream consumed by the frontend.

    Returns the final result snapshot for callers that want a synchronous
    result.
    """
    # Deferred imports — vlm_council pulls in langchain + langgraph.
    from vlm_council import graph as vlm_graph
    from vlm_council.agents import (
        botanics,
        landscape,
        linguistic,
        meta,
        regulatory,
    )
    from vlm_council.state import VLMCouncilState

    agent_modules = {
        "linguistic": linguistic,
        "landscape": landscape,
        "botanics": botanics,
        "regulatory": regulatory,
        "meta": meta,
    }

    image_path = _temp_image_path(image_bytes, mime)

    # Shared mutable phase indicator — the patched `evaluate_hypotheses`
    # needs it to distinguish region_evaluation from country_evaluation
    # events. Updated by the streaming loop below.
    phase_state = {"current": "initial"}

    original_evaluate = {
        name: mod.evaluate_hypotheses for name, mod in agent_modules.items()
    }

    def make_traced_evaluate(name: str, mod):  # noqa: ANN001
        original = original_evaluate[name]

        async def traced(image_b64, image_mime, hypotheses, llm=None):  # noqa: ANN001
            # Per-agent hypothesis shuffling already happens upstream in
            # vlm_council.graph — nothing to do here.
            raw = await original(image_b64, image_mime, hypotheses, llm=llm)
            try:
                evals = vlm_graph._parse_evaluations(name, raw)
            except Exception:
                evals = []

            expected = len(hypotheses)
            if len(evals) < expected:
                # The vendored parser requires both `hypothesis_id` and
                # `confidence` per item. Fall back to a permissive parser
                # that tolerates common Gemma variants.
                rescued = _rescue_evaluations(name, raw, hypotheses)
                if len(rescued) > len(evals):
                    print(
                        f"[adapter] {name} evaluate_hypotheses: "
                        f"strict parse got {len(evals)}/{expected}, "
                        f"rescued {len(rescued)}/{expected}",
                        file=sys.stderr,
                        flush=True,
                    )
                    evals = rescued

            if len(evals) < expected:
                snippet = (raw or "").strip()
                if len(snippet) > 800:
                    snippet = snippet[:800] + "…"
                print(
                    f"[adapter] {name} evaluate_hypotheses: "
                    f"INCOMPLETE — got {len(evals)}/{expected} "
                    f"evaluations. Raw response (first 800 chars):\n"
                    f"--- begin ---\n{snippet}\n--- end ---",
                    file=sys.stderr,
                    flush=True,
                )

            event_type = (
                "region_evaluation"
                if phase_state["current"] == "region"
                else "country_evaluation"
            )
            for ev in evals:
                try:
                    await emit(event_type, _evaluation_to_event(ev, hypotheses))
                except Exception:
                    pass
            return raw

        return traced

    for name, mod in agent_modules.items():
        mod.evaluate_hypotheses = make_traced_evaluate(name, mod)

    # Normalise country aliases so "USA" and "United States" don't produce
    # two separate hypotheses. Patches both Phase-1 (`{agent}_assessment`)
    # and Path-B (`{agent}_country_assessment`) slots because the vendored
    # node picks whichever matches via its `prefix` logic.
    original_country_hypotheses_node = vlm_graph.country_hypotheses_node

    async def patched_country_hypotheses_node(state):  # noqa: ANN001
        patched: dict[str, Any] = dict(state)
        agents = ("linguistic", "landscape", "botanics", "regulatory", "meta")
        for agent in agents:
            for key in (f"{agent}_assessment", f"{agent}_country_assessment"):
                assessment = patched.get(key)
                if not isinstance(assessment, dict):
                    continue
                candidates = assessment.get("candidates")
                if not isinstance(candidates, list):
                    continue
                # New list on top of the state — don't mutate the shared
                # assessment object because the frontend already saw the
                # original names in `agent_assessment` events.
                new_candidates = []
                for cand in candidates:
                    if isinstance(cand, dict) and "country" in cand:
                        new_candidates.append({
                            **cand,
                            "country": _normalise_country(cand["country"]),
                        })
                    else:
                        new_candidates.append(cand)
                patched[key] = {**assessment, "candidates": new_candidates}
        return await original_country_hypotheses_node(patched)

    vlm_graph.country_hypotheses_node = patched_country_hypotheses_node

    try:
        await emit("phase1_started", {
            "agents": list(agent_modules.keys()),
        })

        compiled = vlm_graph.build_graph()

        initial_state: VLMCouncilState = {
            "image_path": str(image_path),
            "image_b64": "",
            "image_mime": "",
        }  # type: ignore[typeddict-item]

        final_state: dict[str, Any] = {}
        seen_assessments: set[str] = set()
        seen_country_assessments: set[str] = set()

        async for chunk in compiled.astream(
            initial_state, stream_mode="updates"
        ):
            # `chunk` is a dict keyed by node name with the partial state
            # update that node produced. Multiple keys can show up in a
            # single chunk for parallel branches.
            for node_name, update in chunk.items():
                final_state.update(update)
                await _emit_for_node(
                    node_name=node_name,
                    update=update,
                    emit=emit,
                    seen_assessments=seen_assessments,
                    seen_country_assessments=seen_country_assessments,
                    phase_state=phase_state,
                )

        country, lat, lng, reasoning = _parse_country_result(
            final_state.get("country_result", "")
        )
        final = {
            "country": country,
            "lat": lat,
            "lng": lng,
            "reasoning": reasoning or final_state.get("final_reasoning", ""),
        }
        await emit("final_result", final)
        return final
    finally:
        for name, mod in agent_modules.items():
            mod.evaluate_hypotheses = original_evaluate[name]
        vlm_graph.country_hypotheses_node = original_country_hypotheses_node
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            pass


_AGENT_NAMES = ("linguistic", "landscape", "botanics", "regulatory", "meta")


async def _emit_for_node(
    node_name: str,
    update: dict[str, Any],
    emit: EventEmitter,
    seen_assessments: set[str],
    seen_country_assessments: set[str],
    phase_state: dict[str, str],
) -> None:
    # Phase-1 assessments arrive on separate `{agent}_agent` node updates.
    for agent in _AGENT_NAMES:
        key = f"{agent}_assessment"
        if key in update and agent not in seen_assessments:
            seen_assessments.add(agent)
            await emit("agent_assessment", _assessment_to_event(agent, update[key]))

    if node_name == "region_consensus_check":
        consensus = bool(update.get("region_consensus"))
        confirmed = update.get("confirmed_region")
        proposed = list(update.get("proposed_regions") or [])
        region_candidates = update.get("region_candidates") or {}
        await emit("region_consensus_result", {
            "consensus": consensus,
            "confirmedRegion": confirmed,
            "proposedRegions": proposed,
            "regionCandidates": region_candidates,
        })
        if not consensus:
            phase_state["current"] = "region"

    elif node_name == "region_hypotheses":
        hyps = update.get("active_hypotheses") or []
        await emit("region_hypotheses_generated", {
            "hypotheses": [_hypothesis_summary(h) for h in hyps],
        })

    elif node_name == "region_evaluate":
        # Per-agent events were emitted via the traced `evaluate_hypotheses`;
        # this is the round-complete summary roll-up.
        evals = update.get("hypothesis_evaluations") or []
        await emit("region_evaluation_complete", {
            "summary": _summarise_evaluations(evals, level="region"),
        })

    elif node_name == "region_decision":
        await emit("region_decision", {
            "confirmedRegion": update.get("confirmed_region", "Unknown"),
            "reasoning": update.get("region_decision_reasoning", ""),
        })
        phase_state["current"] = "country"

    elif node_name == "country_assess":
        for agent in _AGENT_NAMES:
            key = f"{agent}_country_assessment"
            if key in update and agent not in seen_country_assessments:
                seen_country_assessments.add(agent)
                await emit(
                    "country_assessment",
                    _assessment_to_event(agent, update[key]),
                )

    elif node_name == "country_hypotheses":
        hyps = update.get("active_hypotheses") or []
        # Path A skips the region phase entirely, so we may need to set
        # the phase indicator for the first time here.
        phase_state["current"] = "country"
        await emit("country_hypotheses_generated", {
            "hypotheses": [_hypothesis_summary(h) for h in hyps],
        })

    elif node_name == "country_evaluate":
        evals = update.get("hypothesis_evaluations") or []
        await emit("country_evaluation_complete", {
            "summary": _summarise_evaluations(evals, level="country"),
        })

    elif node_name == "country_decision":
        # The final structured event with lat/lng is emitted after astream
        # completes; this marker just tells the UI the judge is closing out.
        await emit("final_started", {})


def _assessment_to_event(agent_name: str, assessment: dict[str, Any]) -> dict[str, Any]:
    candidates_out = []
    for cand in assessment.get("candidates", []) or []:
        country = cand.get("country") or "Unknown"
        latlng = lookup_country_latlng(country)
        if latlng is None:
            lat, lng = (0.0, 0.0)
        else:
            lat, lng = latlng
        candidates_out.append({
            "country": country,
            "lat": lat,
            "lng": lng,
            "confidence": cand.get("confidence", "low"),
            "reasoning": cand.get("reasoning", ""),
        })
    return {
        "agentName": agent_name,
        "candidates": candidates_out,
        "evidence": list(assessment.get("evidence", []) or []),
    }


def _hypothesis_summary(h: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": h.get("hypothesis_id", ""),
        "level": h.get("level", ""),
        "value": h.get("value", ""),
        "statement": h.get("statement", ""),
    }


def _evaluation_to_event(
    eval_obj: dict[str, Any], hypotheses: list[dict[str, Any]]
) -> dict[str, Any]:
    """Convert a HypothesisEvaluation dict into the SSE event payload.

    Looks up the hypothesis statement/value via hypothesis_id so the frontend
    doesn't need a side lookup map.
    """
    hyp_id = eval_obj.get("hypothesis_id", "")
    hyp = next(
        (h for h in hypotheses if h.get("hypothesis_id") == hyp_id),
        None,
    )
    confidence = eval_obj.get("confidence", "neutral")
    return {
        "agentName": eval_obj.get("agent_name", ""),
        "hypothesisId": hyp_id,
        "hypothesisValue": (hyp.get("value") if hyp else hyp_id),
        "level": (hyp.get("level") if hyp else ""),
        "confidence": confidence,
        "supports": EVAL_CONFIDENCE_TO_SUPPORT.get(confidence, False),
        "reasoning": eval_obj.get("reasoning", ""),
        "evidence": list(eval_obj.get("key_evidence", []) or []),
    }


def _summarise_evaluations(
    evaluations: list[dict[str, Any]], *, level: str
) -> list[dict[str, Any]]:
    """Roll up evaluations into per-hypothesis support counts.

    Filters by `level` ("region" or "country") because the graph state's
    hypothesis_evaluations list accumulates both phases.
    """
    prefix = f"{level}_"
    by_hyp: dict[str, dict[str, Any]] = {}
    for ev in evaluations:
        hyp_id = ev.get("hypothesis_id", "")
        if not hyp_id.startswith(prefix):
            continue
        bucket = by_hyp.setdefault(
            hyp_id,
            {
                "hypothesisId": hyp_id,
                "hypothesisValue": _value_from_id(hyp_id, prefix),
                "supportCount": 0,
                "totalAgents": 0,
                "byConfidence": {},
            },
        )
        conf = ev.get("confidence", "neutral")
        bucket["totalAgents"] += 1
        bucket["byConfidence"][conf] = bucket["byConfidence"].get(conf, 0) + 1
        if EVAL_CONFIDENCE_TO_SUPPORT.get(conf, False):
            bucket["supportCount"] += 1
    return list(by_hyp.values())


def _value_from_id(hyp_id: str, prefix: str) -> str:
    """Reconstruct a human-readable value from a hypothesis_id.

    The graph generates IDs like "region_western_europe" or "country_spain"
    via `value.lower().replace(" ", "_")`. We can't perfectly invert that
    (e.g. "United States" → "united_states"), but title-casing the suffix
    is a passable display fallback.
    """
    suffix = hyp_id[len(prefix) :] if hyp_id.startswith(prefix) else hyp_id
    return suffix.replace("_", " ").title()


def _rescue_evaluations(
    agent_name: str, raw: str, hypotheses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Salvage hypothesis evaluations when the strict vendored parser fails.

    Handles common Gemma variants: nested `{"evaluations": [...]}`, items
    keyed by `id` / `hypothesis` instead of `hypothesis_id`, and the
    Phase-1 assessment format regression where the LLM returns
    `{"candidates": [{"country": ..., "confidence": "high|medium|..."}]}`
    instead of hypothesis evaluations (observed when an agent's
    SystemMessage still carries the assessment JSON schema at evaluate
    time). For the last case we synthesise
    strongly_support / support / contradicts evaluations from the
    candidate list.

    We try harder to match items to the known hypothesis ids before giving
    up. Only items that resolve to a known hypothesis id AND a non-empty
    confidence string survive.
    """
    if not raw:
        return []

    # Build lookup tables so we can resolve a candidate item to a known
    # hypothesis_id even if the LLM used an alternate key.
    hyp_ids = {h.get("hypothesis_id") for h in hypotheses}
    by_value: dict[str, str] = {}
    by_statement: dict[str, str] = {}
    for h in hypotheses:
        hid = h.get("hypothesis_id")
        if not hid:
            continue
        v = (h.get("value") or "").strip().lower()
        s = (h.get("statement") or "").strip().lower()
        if v:
            by_value[v] = hid
        if s:
            by_statement[s] = hid

    try:
        from vlm_council.graph import _strip_think_tags
        _, response = _strip_think_tags(raw)
        text = response.strip() if response else raw.strip()
    except Exception:
        text = raw.strip()

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]

    import json
    import re

    # ORDER MATTERS: a greedy `\[.*\]` on `{"candidates": [...]}` matches
    # the INNER array first, so we must probe the top-level object before
    # the array regex.
    items: list[Any] = []
    is_assessment_format = False
    candidates: list[str] = []

    m_obj = re.search(r"\{.*\}", text, re.DOTALL)
    if m_obj:
        candidates.append(m_obj.group())
    m_arr = re.search(r"\[.*\]", text, re.DOTALL)
    if m_arr:
        candidates.append(m_arr.group())

    for blob in candidates:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            items = data
            break
        if isinstance(data, dict):
            if isinstance(data.get("candidates"), list):
                # Phase-1 assessment format regression.
                items = data["candidates"]
                is_assessment_format = True
                break
            for key in ("evaluations", "results", "hypothesis_evaluations", "items"):
                inner = data.get(key)
                if isinstance(inner, list):
                    items = inner
                    break
            if items:
                break
            items = [data]
            break

    if is_assessment_format:
        return _rescue_from_assessment_format(agent_name, items, hypotheses)

    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        hid = (
            item.get("hypothesis_id")
            or item.get("id")
            or item.get("hyp_id")
        )
        if not hid:
            for key in ("hypothesis", "value", "statement"):
                v = item.get(key)
                if isinstance(v, str):
                    vk = v.strip().lower()
                    hid = by_value.get(vk) or by_statement.get(vk)
                    if hid:
                        break
        if not hid or hid not in hyp_ids:
            continue
        confidence = item.get("confidence") or item.get("level")
        if not confidence:
            continue
        out.append(
            {
                "agent_name": agent_name,
                "hypothesis_id": hid,
                "confidence": str(confidence),
                "reasoning": item.get("reasoning")
                or item.get("explanation")
                or "",
                "key_evidence": list(
                    item.get("key_evidence")
                    or item.get("evidence")
                    or []
                ),
            }
        )

    return out


# Deliberately conservative: a "medium" candidate becomes "support" rather
# than "strongly_support" so a fallback interpretation isn't over-amplified.
_ASSESSMENT_TO_EVAL_CONFIDENCE = {
    "high": "strongly_support",
    "medium": "support",
    "low": "support",
    "speculative": "neutral",
}


def _rescue_from_assessment_format(
    agent_name: str,
    candidates: list[Any],
    hypotheses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate a Phase-1 `{"candidates": [...]}` blob into hypothesis
    evaluations.

    Emits exactly one evaluation per country-level hypothesis:
    - If the candidate list mentions the hypothesis country, use the
      mapped confidence.
    - If it lists other countries but not this one, emit "contradicts".
    - If the candidate list is empty (agent has no evidence), emit
      "neutral" for all hypotheses.

    Region-level hypotheses are skipped because assessment candidates are
    countries, not regions.
    """
    country_hyps = [h for h in hypotheses if h.get("level") == "country"]
    if not country_hyps:
        return []

    mentioned: dict[str, str] = {}
    for c in candidates:
        if not isinstance(c, dict):
            continue
        country = (c.get("country") or "").strip().lower()
        conf = (c.get("confidence") or "").strip().lower()
        if country and conf:
            mentioned[country] = conf

    out: list[dict[str, Any]] = []
    for h in country_hyps:
        hid = h.get("hypothesis_id")
        value = (h.get("value") or "").strip().lower()
        if not hid or not value:
            continue

        if not mentioned:
            eval_conf = "neutral"
            reasoning = f"{agent_name} reported no domain evidence for this image."
        elif value in mentioned:
            eval_conf = _ASSESSMENT_TO_EVAL_CONFIDENCE.get(
                mentioned[value], "support"
            )
            reasoning = (
                f"{agent_name} listed {h.get('value')} as a candidate "
                f"with '{mentioned[value]}' confidence in the initial assessment."
            )
        else:
            eval_conf = "contradicts"
            reasoning = (
                f"{agent_name} identified other countries "
                f"({', '.join(sorted(mentioned.keys()))}) but not "
                f"{h.get('value')}."
            )

        out.append(
            {
                "agent_name": agent_name,
                "hypothesis_id": hid,
                "confidence": eval_conf,
                "reasoning": reasoning,
                "key_evidence": [],
            }
        )

    return out


def _parse_country_result(text: str) -> tuple[str, float, float, str]:
    """Parse the judge's `Country: X\\nCoordinates: lat, lng\\nReasoning: ...`.

    Falls back to a centroid lookup if coordinates are missing or garbled.
    """
    country = "Unknown"
    lat: float | None = None
    lng: float | None = None
    reasoning = ""

    for line in (text or "").splitlines():
        line = line.strip()
        if line.lower().startswith("country:"):
            country = line.split(":", 1)[1].strip()
        elif line.lower().startswith("coordinates:"):
            payload = line.split(":", 1)[1].strip()
            try:
                parts = [p.strip() for p in payload.split(",")]
                if len(parts) >= 2:
                    lat = float(parts[0])
                    lng = float(parts[1])
            except (ValueError, TypeError):
                pass
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

    if lat is None or lng is None:
        fallback = lookup_country_latlng(country)
        if fallback:
            lat, lng = fallback
        else:
            lat, lng = 0.0, 0.0

    return country, lat, lng, reasoning


async def _cli_main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: python -m council_adapters.progressive_narrowing_adapter <image>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    path = Path(sys.argv[1])
    image_bytes = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    async def emit(t: str, d: dict[str, Any]) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {t}: {json.dumps(d)[:200]}")

    result = await run_progressive_narrowing(image_bytes, mime, emit)
    print("\nFinal:", json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(_cli_main())
