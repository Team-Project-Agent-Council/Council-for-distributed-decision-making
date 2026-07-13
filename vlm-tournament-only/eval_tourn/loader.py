"""Load v12 result.json files + ground truth into RunRecord objects.

Single source of truth for the evaluation suite. Reuses normalization helpers
from ``vlm_council.evaluate`` so country-name matching stays consistent.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import string

from vlm_council.evaluate import (
    _extract_coordinates,
    _extract_country,
    _haversine_km,
    _normalize_country,
)

# Country centroids (ISO alpha-2 lowercase → [lat, lng]).
# Used as coordinate fallback when the model predicted a country but gave no
# valid coordinates (or emitted the (0, 0) null-island sentinel).
_CENTROIDS_PATH = Path(__file__).parent / "country_centroids.json"
_CENTROIDS: dict[str, tuple[float, float]] = {}
if _CENTROIDS_PATH.exists():
    _raw = json.loads(_CENTROIDS_PATH.read_text())
    _CENTROIDS = {k: (float(v[0]), float(v[1])) for k, v in _raw.items()}


def _centroid_for_country(country_name: str) -> tuple[float, float] | None:
    """Return (lat, lng) centroid for a predicted country name, or None."""
    if not country_name or not _CENTROIDS:
        return None
    norm = _normalize_country(country_name).strip().lower()
    if norm in _CENTROIDS:
        return _CENTROIDS[norm]
    # Try pycountry lookup to get the ISO alpha-2 code
    try:
        import pycountry
        hits = pycountry.countries.search_fuzzy(norm)
        if hits:
            code = hits[0].alpha_2.lower()
            if code in _CENTROIDS:
                return _CENTROIDS[code]
    except Exception:
        pass
    return None


# Country matching — CANONICAL implementation (single source of truth).
# Behaviourally identical to /tmp/canonical_matching.py: the alias table maps
# alpha-2 code -> set of accepted normalized names, and `countries_match`
# follows the 4-step logic (direct code eq -> alias table -> pycountry name set
# -> fuzzy). Version-independent (does not depend on the installed pycountry
# release naming, e.g. TR "Turkey" vs "Türkiye").
_COUNTRY_ALIASES: dict[str, set[str]] = {
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


def _canon_normalize_country(name: str) -> str:
    """Lowercase + strip surrounding whitespace/punctuation (canonical)."""
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
    pred = _canon_normalize_country(predicted)
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
            candidates.add(_canon_normalize_country(val))
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

    image_id: str                       # e.g. "9NNxopqtafH8pTbN_3"
    image_path: str                     # path the council saw

    # Ground truth
    truth_country_code: str             # ISO alpha-2, lowercase, e.g. "kg"
    truth_country_name: str             # canonical name, lowercase
    truth_lat: float
    truth_lng: float

    # Prediction
    pred_country: str                   # raw model output, lowercased
    pred_lat: float | None
    pred_lng: float | None
    final_reasoning: str

    # Topology
    path: str                           # "A" | "B" | ""
    region_consensus: bool
    confirmed_region: str
    candidate_pool: list[str]

    # Per-agent
    assessments: dict[str, dict]                # initial, agent_name → AgentAssessment
    country_assessments: dict[str, dict]        # Path B only, same shape

    # Multi-round signals
    hypothesis_evaluations: list[dict]
    tournament_log: list[dict]
    rag_findings: list[dict]
    road_filter_warnings: list[str]

    # Timing
    total_seconds: float

    # Raw, kept for the LLM judge so we don't lose anything
    raw: dict[str, Any] = field(default_factory=dict)

    # Convenience
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
        # wrap to [-180, 180]
        diff = self.pred_lng - self.truth_lng
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff


# Ground truth loader

def load_ground_truth(csv_path: Path) -> dict[str, dict]:
    """Return ``{image_id_without_extension: {country_code, country_name, lat, lng}}``.

    The CSV columns are: ``filename, sample_id, round, lat, lng, country_code, ...``
    The image_id we key on matches the directory names under ``results_v12_pn/``,
    which is ``<sample_id>_<round>`` (the basename of ``filename`` without extension).
    """
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


def _country_name_for_code(code: str) -> str:
    """Best-effort ISO code → canonical name. Falls back to the code itself."""
    code = code.strip().lower()
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=code.upper())
        if c is not None:
            # Prefer the short common name when the SDK exposes it
            name = getattr(c, "common_name", None) or c.name
            return _normalize_country(name)
    except Exception:
        pass
    return code


# Result loader

def _load_one(result_dir: Path) -> dict | None:
    f = result_dir / "result.json"
    if not f.exists():
        return None
    with open(f) as fp:
        return json.load(fp)


def load_run(results_dir: Path, ground_truth_csv: Path) -> list[RunRecord]:
    """Load every ``result.json`` under ``results_dir`` and pair with ground truth.

    Skips images that have no ground-truth row.
    """
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

        country_result = raw.get("country_result", "") or ""
        pred_country = _extract_country(country_result)
        coords = _extract_coordinates(country_result)
        if coords is None:
            # Fall back to the dedicated coordinates field
            coord_str = raw.get("coordinates", "") or ""
            try:
                lat_s, lng_s = coord_str.split(",")
                coords = (float(lat_s), float(lng_s))
            except Exception:
                coords = None
        # (0, 0) is the null-island sentinel emitted when the model returned no
        # parseable coordinates. Replace with the predicted country's centroid so
        # distance stats still include this run but aren't skewed by a bogus point.
        if (coords is None or (coords[0] == 0.0 and coords[1] == 0.0)) and pred_country:
            coords = _centroid_for_country(pred_country)

        pn = raw.get("progressive_narrowing", {}) or {}

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
                candidate_pool=list(raw.get("candidate_pool", []) or []),
                assessments=dict(raw.get("assessments", {}) or {}),
                country_assessments=dict(raw.get("country_assessments", {}) or {}),
                hypothesis_evaluations=list(raw.get("hypothesis_evaluations", []) or []),
                tournament_log=list(raw.get("tournament_log", []) or []),
                rag_findings=list(raw.get("rag_findings", []) or []),
                road_filter_warnings=list(raw.get("road_filter_warnings", []) or []),
                total_seconds=float((raw.get("timing") or {}).get("total_seconds", 0.0)),
                raw=raw,
            )
        )

    return records


# Helpers used across modules

AGENT_NAMES: tuple[str, ...] = ("linguistic", "landscape", "botanics", "regulatory", "meta")


def top1_country(assessment: dict | None) -> str | None:
    """Return the normalized top-1 country from an AgentAssessment, or None."""
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
