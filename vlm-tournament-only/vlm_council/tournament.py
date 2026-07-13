"""Tournament bracket.

Pairwise comparison of the candidate_pool. Dynamic bracket size:

  ≥4 candidates → trim to top-N (config.tournament_finalists, default 4)
                 → semi-1: seed1 vs seedN, semi-2: seed2 vs seedN-1
                 → final: w1 vs w2     (3 matches for N=4)
  3 candidates → A vs B → winner vs C   (2 matches)
  2 candidates → 1 match
  1 candidate  → no match, that's the answer
  0 candidates → fall back to top-N from active_hypotheses by specialist
                confidence, run the bracket on those.

Each match runs through tournament_judge.judge_match with:
- streetview + reference images for both countries
- per-country specialist evidence summary

The final match additionally asks for coordinates. country_result is rendered
in the v10 format with a "Tournament:" provenance block listing every match.

Seeding score per candidate:
  +2 strongly_support, +1 support, 0 neutral, -1 contradicts, -2 strongly_contradicts
summed across all 5 specialists' country_evaluate outputs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from vlm_council.agents.tournament_judge import (
    _format_specialist_evidence,
    filter_visible_refs,
    identify_visible_features,
    judge_match,
    judge_match_symmetric,
    verify_ref_match,
)
from vlm_council.config import load_config
from vlm_council.rag.keyed_lookup import Reference
from vlm_council.rag_toolbox import (
    RAGToolbox,
    parse_driving_side,
    parse_road_lines,
)
from vlm_council.state import RAGFinding, TournamentMatch, VLMCouncilState


def _load_bollard_db() -> dict[str, dict]:
    """Load bollard_country_summary.json, returns empty dict on failure."""
    candidate = Path(__file__).parent.parent / "data" / "bollards" / "bollard_country_summary.json"
    try:
        with open(candidate) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


_BOLLARD_DB: dict[str, dict] = _load_bollard_db()


_CONFIDENCE_WEIGHTS = {
    "strongly_support": 2,
    "support": 1,
    "neutral": 0,
    "contradicts": -1,
    "strongly_contradicts": -2,
}


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def _country_to_hyp_id(country: str) -> str:
    return "country_" + country.lower().replace(" ", "_")


def _seed_scores(state: VLMCouncilState, candidates: list[str]) -> dict[str, int]:
    """Sum specialist confidence weights per country from hypothesis_evaluations."""
    scores: dict[str, int] = {c: 0 for c in candidates}
    target_ids = {_country_to_hyp_id(c): c for c in candidates}
    for e in state.get("hypothesis_evaluations", []) or []:
        hid = e.get("hypothesis_id", "")
        country = target_ids.get(hid)
        if not country:
            continue
        weight = _CONFIDENCE_WEIGHTS.get(e.get("confidence", "neutral"), 0)
        scores[country] += weight
    return scores


def _seed_pool(state: VLMCouncilState, finalists: int) -> list[str]:
    """Build the seeded participant list (highest score first).

    Pool source priority:
      1. candidate_pool (set by country_hypotheses).
      2. fallback: top-N from active_hypotheses by specialist score.
    """
    pool = list(state.get("candidate_pool") or [])
    if not pool:
        hypotheses = state.get("active_hypotheses", []) or []
        pool = [h.get("value", "") for h in hypotheses if h.get("value")]

    if not pool:
        return []

    scores = _seed_scores(state, pool)
    original_index = {c: i for i, c in enumerate(pool)}
    pool.sort(key=lambda c: (-scores.get(c, 0), original_index.get(c, 0)))

    if len(pool) > finalists:
        pool = pool[:finalists]
    return pool


def _toolbox_or_none() -> RAGToolbox | None:
    cfg = load_config()
    if not cfg.rag_data_dir:
        _log("tournament: VLM_DATA_DIR not set, running without RAG references.")
        return None
    try:
        return RAGToolbox(cfg.rag_data_dir)
    except Exception as e:  # noqa: BLE001
        _log(f"tournament: failed to init RAGToolbox ({type(e).__name__}: {e}); skipping refs.")
        return None


def _regulatory_evidence(state: VLMCouncilState) -> list[str]:
    out: list[str] = []
    for key in ("regulatory_assessment", "regulatory_country_assessment"):
        a = state.get(key) or {}
        ev = a.get("evidence") if isinstance(a, dict) else None
        if isinstance(ev, list):
            out.extend(str(e) for e in ev)
    return out


def _build_bracket(seeded: list[str]) -> list[tuple[str, str | None, str | None, str]]:
    """Produce the schedule of matches.

    Each entry: (round_label, country_a_or_None_for_carryover, country_b, match_kind)
      match_kind ∈ {"first", "carryover_b", "final"}
        - "first" → both countries known up front
        - "carryover_b" → country_a is the previous winner; country_b is fixed
        - "final" → both fixed (semis already produced winners) but emitted with is_final

    For sizes other than the documented ones, we just chain pairwise from the
    seeded order until one survivor remains.
    """
    n = len(seeded)
    if n <= 1:
        return []

    if n == 2:
        return [("final", seeded[0], seeded[1], "final")]

    if n == 3:
        # A vs B then winner vs C
        return [
            ("semi", seeded[0], seeded[1], "first"),
            ("final", None, seeded[2], "final_carry"),  # carry winner forward
        ]

    if n == 4:
        # 1v4, 2v3, then final
        return [
            ("semi-1", seeded[0], seeded[3], "first"),
            ("semi-2", seeded[1], seeded[2], "first"),
            ("final", None, None, "final_pair"),  # both come from prior winners
        ]

    # Generic fallback for N > 4: pairwise chain (seed0 vs seed1, winner vs seed2, ...)
    sched: list[tuple[str, str | None, str | None, str]] = [
        ("round-1", seeded[0], seeded[1], "first")
    ]
    for i in range(2, n):
        kind = "final" if i == n - 1 else "carry"
        sched.append((f"round-{i}", None, seeded[i], "final_carry" if kind == "final" else "carryover_b"))
    return sched


async def tournament_node(state: VLMCouncilState) -> dict[str, Any]:
    """Run the dynamic pairwise tournament and emit the final country_result."""
    cfg = load_config()
    finalists = cfg.tournament_finalists

    seeded = _seed_pool(state, finalists)
    image_b64 = state.get("image_b64", "")
    image_mime = state.get("image_mime", "image/jpeg")

    # No survivors at all: emit Unknown but still surface what we tried.
    if not seeded:
        _log("tournament: no candidates at all, emitting Unknown.")
        return {
            "country_result": (
                "Country: Unknown\n"
                "Coordinates: 0.0, 0.0\n"
                "Reasoning: No candidates survived pre-filters and no fallback "
                "hypotheses available."
            ),
            # No coordinate is available in this degenerate case, so don't
            # emit a fake (0, 0) to distance metrics.
            "coordinates": None,
            "current_phase": "tournament",
        }

    # Single survivor: skip the bracket.
    if len(seeded) == 1:
        winner = seeded[0]
        _log(f"tournament: single survivor → '{winner}', no match needed.")
        warnings = state.get("road_filter_warnings") or []
        warning_block = ("\n\nRoad Check: " + " | ".join(warnings)) if warnings else ""
        return {
            "country_result": (
                f"Country: {winner}\n"
                f"Coordinates: 0.0, 0.0\n"
                f"Reasoning: Only one candidate survived the pre-filters; selected by elimination."
                f"\n\nTournament: walkover (1 candidate)."
                f"{warning_block}"
            ),
            # Walkover: no tournament final ran, so we have no learned
            # coordinate. Downstream should treat this as "no prediction"
            # rather than a real (0, 0) location.
            "coordinates": None,
            "current_phase": "tournament",
            "tournament_log": [],
        }

    # Setup RAG context shared by every match
    toolbox = _toolbox_or_none()
    evidence = _regulatory_evidence(state)
    driving_side = parse_driving_side(evidence)
    if driving_side == "UNCLEAR":
        driving_side_str: str | None = None
    else:
        driving_side_str = driving_side
    road_line = parse_road_lines(evidence)
    warnings = list(state.get("road_filter_warnings") or [])

    # Feature identification, one LLM call per image to detect which artifact
    # categories are visible. Drives category-targeted fetch_references.
    detected_categories: list[str] = []
    bollard_properties: dict = {}
    if toolbox is not None:
        available_cats = toolbox.lookup.available_categories_multi(seeded)
        if available_cats:
            _log(f"tournament: running feature_identifier ({len(available_cats)} cats available)")
            # Pass only the candidate countries' DB entries, the prior pipeline
            # already narrowed the search space to these countries, so any
            # bollard property outside their union can't possibly match a real
            # ref anyway. This both shrinks the prompt vocab and ensures every
            # detected term is matchable at fetch time.
            db_props_for_candidates = {c: _BOLLARD_DB[c] for c in seeded if c in _BOLLARD_DB}
            feat = await identify_visible_features(
                image_b64=image_b64,
                image_mime=image_mime,
                available_categories=available_cats,
                db_bollard_props=db_props_for_candidates or None,
            )
            detected_categories = feat.get("categories", [])
            bollard_properties = feat.get("bollard_properties", {})
            _log(
                f"tournament: detected features={detected_categories}, "
                f"bollard_props={bollard_properties}"
            )

    schedule = _build_bracket(seeded)
    _log(
        f"tournament: bracket size={len(seeded)} → {len(schedule)} match(es); "
        f"seeded={seeded}; driving_side={driving_side}, road_line={road_line!r}"
    )

    matches_out: list[TournamentMatch] = []
    findings_out: list[RAGFinding] = []

    # Per-country cache of verified refs, avoids re-running filter+verify
    # for carryover winners that reappear in later rounds.
    verified_refs_cache: dict[str, list[Reference]] = {}

    # Refs actually passed to judge_match_symmetric, keyed by (country, ref_path).
    # Only populated at match-call time, not on fetch/verify.
    refs_used_in_match: set[tuple[str, str]] = set()

    last_winner: str | None = None
    final_winner: str | None = None
    final_coords: str = "0.0, 0.0"

    for round_label, slot_a, slot_b, kind in schedule:
        # Resolve participants
        if kind == "first":
            country_a, country_b = slot_a, slot_b
            is_final = False
        elif kind == "carryover_b":
            country_a, country_b = last_winner, slot_b
            is_final = False
        elif kind == "final_carry":
            country_a, country_b = last_winner, slot_b
            is_final = True
        elif kind == "final":
            country_a, country_b = slot_a, slot_b
            is_final = True
        elif kind == "final_pair":
            # 4-bracket final: both winners come from the two prior matches
            if len(matches_out) < 2:
                _log("tournament: final_pair scheduled but <2 prior matches, skipping.")
                continue
            country_a = matches_out[0]["winner"]
            country_b = matches_out[1]["winner"]
            is_final = True
        else:
            _log(f"tournament: unknown match kind '{kind}', skipping.")
            continue

        if not country_a or not country_b or country_a == country_b:
            # Defensive: one side missing → auto-advance
            chosen = country_a or country_b
            if not chosen:
                _log(f"tournament[{round_label}]: both sides missing, abort match.")
                continue
            _log(f"tournament[{round_label}]: defaulting to '{chosen}' (no opponent).")
            last_winner = chosen
            if is_final:
                final_winner = chosen
                final_coords = "0.0, 0.0"
            continue

        # Fetch refs for both countries, category-targeted, then filter
        # against the streetview, then verify each match individually.
        # Both countries cached per-image so carryover winners don't repeat work.
        refs_a: list[Reference] = []
        refs_b: list[Reference] = []
        if toolbox is not None:
            import asyncio as _asyncio

            async def _get_verified_refs(country: str) -> list[Reference]:
                if country in verified_refs_cache:
                    cached = verified_refs_cache[country]
                    _log(
                        f"tournament[{round_label}]: {country}, using {len(cached)} cached verified refs"
                    )
                    return cached

                if detected_categories:
                    raw_refs = toolbox.fetch_references(
                        [country],
                        categories=detected_categories,
                        bollard_materials=bollard_properties.get("materials") or None,
                        bollard_colors=bollard_properties.get("colors") or None,
                    )
                else:
                    raw_refs = toolbox.fetch_references(
                        [country],
                        bollard_materials=bollard_properties.get("materials") or None,
                        bollard_colors=bollard_properties.get("colors") or None,
                    )

                if not raw_refs:
                    verified_refs_cache[country] = []
                    return []

                filtered = await filter_visible_refs(
                    image_b64=image_b64,
                    image_mime=image_mime,
                    country=country,
                    refs=raw_refs,
                )
                _log(
                    f"tournament[{round_label}]: filter {country}: "
                    f"{len(raw_refs)} → {len(filtered)} matched"
                )

                if filtered:
                    verify_tasks = [
                        verify_ref_match(image_b64=image_b64, image_mime=image_mime, ref=r)
                        for r in filtered
                    ]
                    confirm_flags = await _asyncio.gather(*verify_tasks)
                    verified = [r for r, ok in zip(filtered, confirm_flags) if ok]
                    _log(
                        f"tournament[{round_label}]: verify {country}: "
                        f"{len(filtered)} → {len(verified)} confirmed"
                    )
                else:
                    verified = []

                verified_refs_cache[country] = verified
                return verified

            refs_a, refs_b = await _asyncio.gather(
                _get_verified_refs(country_a),
                _get_verified_refs(country_b),
            )

        spec_block = _format_specialist_evidence(state, country_a, country_b)

        # Pool-rank for symmetric tie-break: lower index = higher seed.
        # `seeded` is the ranked list at the start of the bracket; later
        # winners (carryover) inherit their original seed position.
        try:
            pool_rank_a = seeded.index(country_a)
        except ValueError:
            pool_rank_a = len(seeded)  # not in pool → lowest seed
        try:
            pool_rank_b = seeded.index(country_b)
        except ValueError:
            pool_rank_b = len(seeded)

        _log(
            f"tournament[{round_label}]: {country_a} vs {country_b} "
            f"(refs A={len(refs_a)}, B={len(refs_b)}, final={is_final}, "
            f"pool_rank A={pool_rank_a} B={pool_rank_b})"
        )

        # Record refs that are actually handed to the judge for this match.
        for ref in refs_a:
            refs_used_in_match.add((country_a, ref.image_path))
        for ref in refs_b:
            refs_used_in_match.add((country_b, ref.image_path))

        t0 = time.time()
        try:
            result = await judge_match_symmetric(
                image_b64=image_b64,
                image_mime=image_mime,
                country_a=country_a,
                country_b=country_b,
                refs_a=refs_a,
                refs_b=refs_b,
                driving_side=driving_side_str,
                road_line=road_line,
                warnings=warnings,
                specialist_block=spec_block,
                is_final=is_final,
                pool_rank_a=pool_rank_a,
                pool_rank_b=pool_rank_b,
            )
        except Exception as e:  # noqa: BLE001
            # Judge failed entirely, fall back to higher pool-rank seed,
            # NOT to country_a (that was the bias source we just fixed).
            fallback_winner = country_a if pool_rank_a <= pool_rank_b else country_b
            _log(
                f"tournament[{round_label}]: ERROR ({type(e).__name__}: {e}), "
                f"falling back to higher seed '{fallback_winner}'."
            )
            result = {
                "winner": fallback_winner,
                "reasoning": f"Judge error: {type(e).__name__}",
                "coordinates": "0.0, 0.0",
                "agreement": "judge_error",
            }

        elapsed = time.time() - t0
        winner = result.get("winner") or (
            country_a if pool_rank_a <= pool_rank_b else country_b
        )
        reasoning = result.get("reasoning", "")
        agreement = result.get("agreement", "unknown")
        _log(
            f"tournament[{round_label}]: winner='{winner}' "
            f"(agreement={agreement}) in {elapsed:.1f}s"
        )

        match: TournamentMatch = {
            "round_label": round_label,
            "country_a": country_a,
            "country_b": country_b,
            "pool_rank_a": pool_rank_a,
            "pool_rank_b": pool_rank_b,
            "winner": winner,
            "reasoning": reasoning,
            "agreement": agreement,
        }
        matches_out.append(match)

        loser = country_b if winner == country_a else country_a
        findings_out.append(RAGFinding(
            kind="tournament_match",
            country=loser,
            detail=reasoning or f"lost {round_label} to {winner}",
            opponent=winner,
            winner=winner,
        ))

        last_winner = winner
        if is_final:
            final_winner = winner
            final_coords = result.get("coordinates", "0.0, 0.0") or "0.0, 0.0"

    if final_winner is None:
        # Shouldn't happen, defensive fallback
        final_winner = last_winner or seeded[0]
        _log(f"tournament: no final match recorded, using last_winner='{final_winner}'.")

    # Render country_result with v10-style Tournament + Road Check provenance
    tourn_lines = [
        f"  {m['round_label']}: {m['country_a']} vs {m['country_b']} -> {m['winner']}"
        for m in matches_out
    ]
    tournament_block = "Tournament:\n" + "\n".join(tourn_lines) if tourn_lines else "Tournament: (no matches run)"
    warning_block = ("\nRoad Check: " + " | ".join(warnings)) if warnings else ""

    final_reasoning = ""
    if matches_out:
        final_reasoning = matches_out[-1].get("reasoning", "")

    country_result = (
        f"Country: {final_winner}\n"
        f"Coordinates: {final_coords}\n"
        f"Reasoning: {final_reasoning or 'Tournament winner (see provenance below).'}"
        f"\n\n{tournament_block}"
        f"{warning_block}"
    )

    # Convert coordinates string ("lat, lng") to structured dict for downstream
    # metrics.  If the string is unparseable or the walkover-fallback (0, 0)
    # was written, keep it as None so distance metrics don't see (0, 0)
    # outliers.
    from vlm_council.coordinates import parse_coordinates
    coords_struct = parse_coordinates(f"Coordinates: {final_coords}")
    # If the walkover / no-candidates fallback (0.0, 0.0) leaked through,
    # treat it as "no coordinate available" rather than a real prediction.
    if coords_struct is not None and coords_struct.get("lat") == 0.0 and coords_struct.get("lng") == 0.0:
        coords_struct = None

    # Only refs that were actually passed to a judge call, not merely fetched/verified.
    rag_refs_seen: list[list[str]] = [[country, path] for country, path in refs_used_in_match]

    return {
        "country_result": country_result,
        "coordinates": coords_struct,
        "final_reasoning": final_reasoning,
        "tournament_log": matches_out,
        "rag_findings": findings_out,
        "rag_refs_seen": (state.get("rag_refs_seen") or []) + rag_refs_seen,
        "current_phase": "tournament",
    }
