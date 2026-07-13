"""Load VLM Council result.json files + ground truth into RunRecord objects.

result.json structure (from vlm_council/batch.py):
  image_path, model, judge_model, timing, assessments, progressive_narrowing,
  country_assessments, hypothesis_evaluations, country_result, coordinates,
  final_reasoning.

progressive_narrowing sub-keys:
  region_consensus, confirmed_region, proposed_regions, region_candidates,
  region_decision_reasoning, path.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlm_council.evaluate import (
    _extract_coordinates,
    _extract_country,
    _haversine_km,
    _normalize_country,
)


# CANONICAL COUNTRY MATCHING — single source of truth for the whole submission.
# Keep countries_match() behaviourally identical to /tmp/canonical_matching.py.
# The alias table maps alpha-2 code -> set of accepted normalized names.
_COUNTRY_ALIASES = {
    "tr": {"turkey", "turkiye", "türkiye"},
    "us": {"usa", "united states", "united states of america", "america", "u.s.", "u.s.a."},
    "gb": {"uk", "united kingdom", "great britain", "england", "scotland", "wales", "britain"},
    "ru": {"russia", "russian federation"},
    "kr": {"south korea", "republic of korea", "korea, republic of", "korea"},
    "kp": {"north korea", "dprk"},
    "de": {"germany", "deutschland"},
    "cz": {"czech republic", "czechia"},
    "nl": {"netherlands", "the netherlands", "holland"},
    "ae": {"united arab emirates", "uae"},
    "ci": {"ivory coast", "cote d'ivoire", "côte d'ivoire"},
    "ba": {"bosnia", "bosnia and herzegovina"},
    "tt": {"trinidad", "trinidad and tobago"},
    "tl": {"east timor", "timor-leste", "timor leste"},
    "sz": {"swaziland", "eswatini"},
    "mm": {"burma", "myanmar"},
    "ir": {"iran"},
    "sy": {"syria"},
    "la": {"laos"},
    "bo": {"bolivia"},
    "ve": {"venezuela"},
    "vn": {"vietnam"},
    "md": {"moldova"},
    "tz": {"tanzania"},
}


def _normalize_country_canonical(name: str) -> str:
    """Lowercase + strip surrounding whitespace/punctuation (canonical)."""
    import string
    if not name:
        return ""
    return name.strip().lower().rstrip(string.punctuation + " ").strip()


def countries_match(predicted: str, actual_code: str) -> bool:
    """True iff `predicted` refers to ISO alpha-2 `actual_code`.

    1. direct code equality
    2. explicit version-independent alias table
    3. pycountry name set (name/official/common/alpha2/alpha3)
    4. pycountry fuzzy search
    """
    if not predicted or not actual_code:
        return False
    pred = _normalize_country_canonical(predicted)
    code = actual_code.strip().lower()
    if not pred:
        return False
    if pred == code:
        return True
    if pred in _COUNTRY_ALIASES.get(code, set()):
        return True
    try:
        import pycountry
    except Exception:
        return False
    target = pycountry.countries.get(alpha_2=code.upper())
    if target is None:
        return False
    candidates = set()
    for attr in ("name", "official_name", "common_name", "alpha_2", "alpha_3"):
        val = getattr(target, attr, None)
        if val:
            candidates.add(_normalize_country_canonical(val))
    candidates.discard("")
    if pred in candidates:
        return True
    try:
        results = pycountry.countries.search_fuzzy(pred)
        if results and results[0].alpha_2.lower() == code:
            return True
    except LookupError:
        pass
    return False


@dataclass
class RunRecord:
    """Normalized view of a single image's run + ground truth."""

    image_id: str
    image_path: str

    # Ground truth
    truth_country_code: str
    truth_country_name: str
    truth_lat: float
    truth_lng: float

    # Prediction
    pred_country: str
    pred_lat: float | None
    pred_lng: float | None
    final_reasoning: str

    # Topology
    path: str                   # "A" | "B"
    region_consensus: bool
    confirmed_region: str
    proposed_regions: list[str]
    region_candidates: dict     # {region: {country: agent_count}}
    region_decision_reasoning: str

    # Per-agent assessments
    assessments: dict[str, dict]           # initial, agent_name → AgentAssessment
    country_assessments: dict[str, dict]   # Path B only

    # Hypothesis evaluations
    hypothesis_evaluations: list[dict]     # list of HypothesisEvaluation dicts
    active_hypotheses: list[dict]          # final country hypotheses presented

    # Timing
    total_seconds: float

    # Raw, kept for the LLM judge
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_correct(self) -> bool:
        return countries_match(self.pred_country, self.truth_country_code)

    @property
    def haversine_km(self) -> float | None:
        if self.pred_lat is None or self.pred_lng is None:
            return None
        return _haversine_km(self.pred_lat, self.pred_lng, self.truth_lat, self.truth_lng)

    @property
    def lat_error(self) -> float | None:
        if self.pred_lat is None:
            return None
        return self.pred_lat - self.truth_lat

    @property
    def lng_error(self) -> float | None:
        if self.pred_lng is None:
            return None
        diff = self.pred_lng - self.truth_lng
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    @property
    def truth_in_hypothesis_pool(self) -> bool:
        """Was the truth country among the country hypotheses evaluated?"""
        for h in self.active_hypotheses:
            if countries_match(h.get("value", ""), self.truth_country_code):
                return True
        # Fallback: check hypothesis_evaluations hypothesis_ids
        for e in self.hypothesis_evaluations:
            hid = e.get("hypothesis_id", "")
            if hid.startswith("country_"):
                country = hid[len("country_"):].replace("_", " ")
                if countries_match(country, self.truth_country_code):
                    return True
        return False


def _country_name_for_code(code: str) -> str:
    code = code.strip().lower()
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=code.upper())
        if c is not None:
            name = getattr(c, "common_name", None) or c.name
            return _normalize_country(name)
    except Exception:
        pass
    return code


def load_ground_truth(csv_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            if not filename:
                continue
            image_id = Path(filename).stem
            try:
                lat = float(row["lat"])
                lng = float(row["lng"])
            except (ValueError, KeyError):
                continue
            code = row["country_code"].strip().lower()
            out[image_id] = {
                "country_code": code,
                "country_name": _country_name_for_code(code),
                "lat": lat,
                "lng": lng,
            }
    return out


def _load_one(result_dir: Path) -> dict | None:
    f = result_dir / "result.json"
    if not f.exists():
        return None
    with open(f) as fp:
        return json.load(fp)


def load_run(results_dir: Path, ground_truth_csv: Path) -> list[RunRecord]:
    truth = load_ground_truth(ground_truth_csv)
    records: list[RunRecord] = []

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue
        image_id = result_dir.name
        gt = truth.get(image_id)
        if gt is None:
            continue
        raw = _load_one(result_dir)
        if raw is None:
            continue
        if raw.get("error"):
            continue

        country_result = raw.get("country_result", "") or ""
        pred_country = _extract_country(country_result)
        coords = _extract_coordinates(country_result)
        if coords is None:
            # New format: {"lat": float, "lng": float} or None.
            # Legacy format: "lat, lng" as a string.
            raw_coords = raw.get("coordinates")
            if isinstance(raw_coords, dict):
                lat = raw_coords.get("lat")
                lng = raw_coords.get("lng")
                if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                    coords = (float(lat), float(lng))
            elif isinstance(raw_coords, str) and raw_coords:
                try:
                    lat_s, lng_s = raw_coords.split(",")
                    coords = (float(lat_s), float(lng_s))
                except Exception:
                    coords = None

        pn = raw.get("progressive_narrowing", {}) or {}

        # Reconstruct active_hypotheses: best-effort from hypothesis_evaluations
        # (the state field isn't persisted, so we derive unique hypotheses from evals)
        seen_hyps: dict[str, dict] = {}
        for e in raw.get("hypothesis_evaluations", []):
            hid = e.get("hypothesis_id", "")
            if hid and hid not in seen_hyps:
                value = hid
                if hid.startswith("country_"):
                    value = hid[len("country_"):].replace("_", " ")
                elif hid.startswith("region_"):
                    value = hid[len("region_"):].replace("_", " ")
                seen_hyps[hid] = {"hypothesis_id": hid, "value": value}
        # Keep only country-level hypotheses for truth_in_hypothesis_pool
        active_hypotheses = [
            h for h in seen_hyps.values()
            if h["hypothesis_id"].startswith("country_")
        ]

        records.append(
            RunRecord(
                image_id=image_id,
                image_path=raw.get("image_path", ""),
                truth_country_code=gt["country_code"],
                truth_country_name=gt["country_name"],
                truth_lat=gt["lat"],
                truth_lng=gt["lng"],
                pred_country=_normalize_country(pred_country),
                pred_lat=coords[0] if coords else None,
                pred_lng=coords[1] if coords else None,
                final_reasoning=raw.get("final_reasoning", "") or "",
                path=pn.get("path", ""),
                region_consensus=bool(pn.get("region_consensus", False)),
                confirmed_region=pn.get("confirmed_region", "") or "",
                proposed_regions=list(pn.get("proposed_regions", []) or []),
                region_candidates=dict(pn.get("region_candidates", {}) or {}),
                region_decision_reasoning=pn.get("region_decision_reasoning", "") or "",
                assessments=dict(raw.get("assessments", {}) or {}),
                country_assessments=dict(raw.get("country_assessments", {}) or {}),
                hypothesis_evaluations=list(raw.get("hypothesis_evaluations", []) or []),
                active_hypotheses=active_hypotheses,
                total_seconds=float((raw.get("timing") or {}).get("total_seconds", 0.0)),
                raw=raw,
            )
        )

    return records


AGENT_NAMES: tuple[str, ...] = ("linguistic", "landscape", "botanics", "regulatory", "meta")


def top1_country(assessment: dict | None) -> str | None:
    if not assessment:
        return None
    candidates = assessment.get("candidates") or []
    if not candidates:
        return None
    country = candidates[0].get("country", "")
    return _normalize_country(country) if country else None


def topk_countries(assessment: dict | None, k: int = 3) -> list[str]:
    if not assessment:
        return []
    out = []
    for c in (assessment.get("candidates") or [])[:k]:
        name = c.get("country", "")
        if name:
            out.append(_normalize_country(name))
    return out
