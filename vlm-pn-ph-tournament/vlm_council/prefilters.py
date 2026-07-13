"""Deterministic RAG pre-filters that sit between country_hypotheses and country_evaluate.

Two nodes:
- ``road_marking_prefilter_node``  uses RAGToolbox.road_line_check_structured
- ``driving_side_prefilter_node``  uses RAGToolbox.driving_side_filter

Both consume ``state['road_evidence']`` (produced by the road_evidence_extractor)
rather than parsing the regulatory agent's free-text evidence list. Both are
pure dictionary lookups (no LLM). Both surface findings via ``rag_findings`` and
never silently destroy candidates: if strict filtering would eliminate every
candidate, recovery kicks in (v10 behavior), the candidates are kept and a
``road_filter_warnings`` entry is recorded.

candidate_pool flow:
- After country_hypotheses, ``active_hypotheses`` holds top-K country candidates.
- road_marking_prefilter consumes that list, writes a smaller list back to
  ``candidate_pool`` (or the same list under recovery).
- driving_side_prefilter consumes ``candidate_pool`` and trims further.
- After both filters: country_evaluate sees the survivor set as hypotheses.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from vlm_council.config import load_config
from vlm_council.rag_toolbox import RAGToolbox
from vlm_council.state import RAGFinding, VLMCouncilState


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


_VALID_OBSERVED_COLORS = {"white", "yellow", "red", "blue"}


def _hypotheses_to_countries(hypotheses: list[dict]) -> list[str]:
    return [h.get("value", "") for h in hypotheses if h.get("value")]


def _toolbox_or_none() -> RAGToolbox | None:
    cfg = load_config()
    if not cfg.rag_data_dir:
        _log("prefilter: VLM_DATA_DIR not set, pre-filters disabled (no-op).")
        return None
    try:
        return RAGToolbox(cfg.rag_data_dir)
    except Exception as e:  # noqa: BLE001
        _log(f"prefilter: failed to init RAGToolbox ({type(e).__name__}: {e}); skipping.")
        return None


async def road_marking_prefilter_node(state: VLMCouncilState) -> dict[str, Any]:
    """Eliminate countries whose road-line patterns contradict observed colors.

    Reads observed colors from ``state['road_evidence']`` (extractor output).
    Reads candidate countries from ``active_hypotheses`` (set by country_hypotheses).
    Writes survivors to ``candidate_pool``.
    """
    hypotheses = state.get("active_hypotheses", [])
    candidates = _hypotheses_to_countries(hypotheses)
    if not candidates:
        _log("road_marking_prefilter: no candidates; skipping.")
        return {"candidate_pool": [], "current_phase": "prefilter"}

    toolbox = _toolbox_or_none()
    if toolbox is None:
        return {"candidate_pool": list(candidates), "current_phase": "prefilter"}

    evidence = state.get("road_evidence") or {}
    outside = str(evidence.get("outside_color", "unclear")).lower()
    inside = str(evidence.get("inside_color", "unclear")).lower()

    if outside not in _VALID_OBSERVED_COLORS or inside not in _VALID_OBSERVED_COLORS:
        _log(
            f"road_marking_prefilter: observation not actionable "
            f"(outside={outside}, inside={inside}); passing through {len(candidates)} candidates."
        )
        return {"candidate_pool": list(candidates), "current_phase": "prefilter"}

    result = toolbox.road_line_check_structured(candidates, outside, inside)

    findings: list[RAGFinding] = []
    survivors: list[str] = []
    for country in candidates:
        verdict, pattern = result.by_country.get(country, ("UNKNOWN", ""))
        if verdict == "MISMATCH":
            findings.append(RAGFinding(
                kind="elim_road_marking",
                country=country,
                detail=f"observed outside={outside}, inside={inside} vs table='{pattern}'",
            ))
        else:
            # MATCH or UNKNOWN keep
            survivors.append(country)

    warnings: list[str] = []
    if result.warning:
        warnings.append(result.warning)
        findings.append(RAGFinding(kind="recovery", country="", detail=result.warning))
        # Recovery: every MISMATCH was already turned into UNKNOWN by the toolbox,
        # so survivors above is the full list.

    _log(
        f"road_marking_prefilter: outside={outside}, inside={inside} → "
        f"survivors {len(survivors)}/{len(candidates)}"
        + (f" [recovery: {result.warning}]" if result.warning else "")
        + f" eliminated={[f['country'] for f in findings if f['kind'] == 'elim_road_marking']}"
    )

    return {
        "candidate_pool": survivors,
        "rag_findings": findings,
        "road_filter_warnings": warnings,
        "current_phase": "prefilter",
    }


async def driving_side_prefilter_node(state: VLMCouncilState) -> dict[str, Any]:
    """Eliminate countries whose driving side contradicts the observed side.

    Reads observed side from ``state['road_evidence']``. The extractor enforces
    the rule that basis=none ⇒ driving_side=UNCLEAR, so an UNCLEAR value here
    means the agent was honest about not knowing.
    """
    pool = state.get("candidate_pool") or _hypotheses_to_countries(state.get("active_hypotheses", []))
    if not pool:
        _log("driving_side_prefilter: empty pool; skipping.")
        return {"candidate_pool": [], "current_phase": "prefilter"}

    toolbox = _toolbox_or_none()
    if toolbox is None:
        return {"candidate_pool": list(pool), "current_phase": "prefilter"}

    evidence = state.get("road_evidence") or {}
    observed = str(evidence.get("driving_side", "UNCLEAR")).upper()
    basis = str(evidence.get("driving_side_basis", "none")).lower()

    if observed == "UNCLEAR" or basis == "none":
        _log(f"driving_side_prefilter: observed={observed} (basis={basis}), skipping.")
        return {"candidate_pool": list(pool), "current_phase": "prefilter"}

    result = toolbox.driving_side_filter(pool, observed)  # type: ignore[arg-type]

    findings: list[RAGFinding] = []
    for country, reason in result.eliminated:
        findings.append(RAGFinding(kind="elim_driving", country=country, detail=reason))

    warnings = list(state.get("road_filter_warnings") or [])
    if result.warning:
        warnings.append(result.warning)
        findings.append(RAGFinding(kind="recovery", country="", detail=result.warning))

    _log(
        f"driving_side_prefilter: observed={observed} (basis={basis}) → "
        f"survivors {len(result.kept)}/{len(pool)}"
        + (f" [recovery: {result.warning}]" if result.warning else "")
        + f" eliminated={[c for c, _ in result.eliminated]}"
    )

    return {
        "candidate_pool": list(result.kept),
        "rag_findings": findings,
        "road_filter_warnings": warnings,
        "current_phase": "prefilter",
    }
