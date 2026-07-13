"""Graph nodes that consume the road_evidence_extractor output.

Two nodes live here:

- ``road_evidence_node`` runs the dedicated VLM extractor (color + driving side
  with basis) and writes the result to ``state['road_evidence']``.

- ``region_road_filter_node`` takes the top-2 regions from ``surviving_regions``
  and asks the structured RAG check whether ANY country in each region has a
  road-line pattern matching the observed (outside, inside) colors. Regions
  with zero matching countries are eliminated. Recovery: if BOTH regions would
  be eliminated, keep both with a warning (mirrors v10's all-eliminated
  recovery rule for road markings).

The driving_side check is applied PER COUNTRY downstream in
``driving_side_prefilter_node``; we don't try to use it as a region-level
filter because most regions span both sides.
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from vlm_council.agents import road_evidence_extractor
from vlm_council.config import load_config
from vlm_council.rag_toolbox import RAGToolbox
from vlm_council.regions import countries_in_region
from vlm_council.state import VLMCouncilState


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


VLM_CALL_TIMEOUT = 600


async def road_evidence_node(state: VLMCouncilState) -> dict[str, Any]:
    """Run the road_evidence_extractor and store its structured output."""
    _log("road_evidence: extracting structured road observations...")
    t0 = time.time()
    try:
        evidence = await asyncio.wait_for(
            road_evidence_extractor.extract(state["image_b64"], state["image_mime"]),
            timeout=VLM_CALL_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
        _log(f"road_evidence: ERROR ({type(e).__name__}); using empty evidence")
        evidence = {
            "outside_color": "unclear",
            "inside_color": "unclear",
            "driving_side": "UNCLEAR",
            "driving_side_basis": "none",
        }

    elapsed = time.time() - t0
    _log(
        f"road_evidence: outside={evidence['outside_color']}, "
        f"inside={evidence['inside_color']}, "
        f"side={evidence['driving_side']} "
        f"(basis={evidence['driving_side_basis']}) in {elapsed:.1f}s"
    )
    return {"road_evidence": evidence}


def _toolbox_or_none() -> RAGToolbox | None:
    cfg = load_config()
    if not cfg.rag_data_dir:
        _log("region_road_filter: VLM_DATA_DIR not set, filter disabled.")
        return None
    try:
        return RAGToolbox(cfg.rag_data_dir)
    except Exception as e:  # noqa: BLE001
        _log(f"region_road_filter: failed to init RAGToolbox ({type(e).__name__}: {e}); skipping.")
        return None


async def region_road_filter_node(state: VLMCouncilState) -> dict[str, Any]:
    """Eliminate any region in which NO country matches the observed road markings.

    Recovery rule: if all regions would be eliminated, keep all with a warning.
    """
    survivors_in = list(state.get("surviving_regions") or [])
    if not survivors_in:
        # Fall back to confirmed_region (single) if surviving_regions wasn't populated
        cr = state.get("confirmed_region")
        if cr and cr != "Unknown":
            survivors_in = [cr]

    if not survivors_in:
        _log("region_road_filter: no surviving_regions; skipping.")
        return {}

    evidence = state.get("road_evidence") or {}
    outside = str(evidence.get("outside_color", "unclear")).lower()
    inside = str(evidence.get("inside_color", "unclear")).lower()

    # Non-actionable observation → keep all regions, no warning needed
    if outside not in {"white", "yellow", "red", "blue"} or inside not in {"white", "yellow", "red", "blue"}:
        _log(
            f"region_road_filter: observation not actionable "
            f"(outside={outside}, inside={inside}); keeping all {len(survivors_in)} regions."
        )
        return {"surviving_regions": survivors_in}

    toolbox = _toolbox_or_none()
    if toolbox is None:
        return {"surviving_regions": survivors_in}

    kept: list[str] = []
    eliminated: list[tuple[str, str]] = []
    per_region_match_counts: dict[str, int] = {}

    for region in survivors_in:
        countries = countries_in_region(region)
        if not countries:
            # Unknown region → keep defensively
            kept.append(region)
            per_region_match_counts[region] = -1
            continue

        result = toolbox.road_line_check_structured(countries, outside, inside)
        match_count = sum(1 for c, (verdict, _) in result.by_country.items() if verdict == "MATCH")
        per_region_match_counts[region] = match_count

        if match_count > 0:
            kept.append(region)
        else:
            eliminated.append((region, f"0/{len(countries)} countries match outside={outside},inside={inside}"))

    warnings = list(state.get("road_filter_warnings") or [])
    if not kept and eliminated:
        warning = (
            f"Region road-marking filter would eliminate ALL regions {survivors_in} "
            f"given observed outside={outside}, inside={inside}. "
            f"Recovery: keeping all regions; the road_evidence_extractor reading may be wrong."
        )
        _log(f"region_road_filter: {warning}")
        warnings.append(warning)
        return {
            "surviving_regions": survivors_in,
            "road_filter_warnings": warnings,
        }

    if eliminated:
        _log(
            f"region_road_filter: kept {[r for r in kept]} "
            f"(matches={[per_region_match_counts[r] for r in kept]}); "
            f"eliminated {[r for r, _ in eliminated]}"
        )
    else:
        _log(f"region_road_filter: all {len(kept)} regions have matching countries, no eliminations.")

    return {"surviving_regions": kept}
