"""Load Global Context Re-guess result.json files + ground truth into RunRecord objects.

Single source of truth for the eval_reguess evaluation suite.

CSV columns (georc_locations.csv):
    filename, sample_id, round, lat, lng, country_code, heading, pitch, panoId

The image_id matches directory names under results_global_context_re_guess_*/,
which is <sample_id>_<round> (the basename of filename without extension).
"""

from __future__ import annotations

import csv
import json
import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


AGENT_NAMES: list[str] = ["linguistic", "landscape", "botanics", "regulatory", "meta"]


# Version-independent country aliases (alpha-2 -> extra accepted names).
# Guards against pycountry release changes (e.g. TR renamed "Turkey" -> "Türkiye")
# and common model spellings. Identical across all approach loaders.
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


# ── Country normalization ────────────────────────────────────────────────────


def _normalize_country(name: str) -> str:
    """Lowercase and strip punctuation for comparison."""
    if not name:
        return ""
    name = name.strip().lower()
    # Remove trailing punctuation
    name = name.rstrip(string.punctuation + " ")
    return name


def _country_name_for_code(code: str) -> str:
    """Best-effort ISO alpha-2 code → canonical name. Falls back to the code."""
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


def countries_match(predicted: str, actual_code: str) -> bool:
    """Return True if predicted country string matches the given ISO alpha-2 code.

    Tries direct string comparison first, then pycountry name lookup, then fuzzy.
    Case-insensitive; strips punctuation.
    """
    if not predicted or not actual_code:
        return False
    pred = _normalize_country(predicted)
    code = actual_code.strip().lower()
    if not pred:
        return False
    # Direct code match
    if pred == code:
        return True
    # Version-independent alias match
    if pred in _COUNTRY_ALIASES.get(code, set()):
        return True
    try:
        import pycountry
    except Exception:
        return False

    target = pycountry.countries.get(alpha_2=code.upper())
    if target is None:
        return False

    candidates: set[str] = set()
    for attr in ("name", "official_name", "common_name", "alpha_2", "alpha_3"):
        val = getattr(target, attr, None)
        if val:
            candidates.add(_normalize_country(val))
    candidates.discard("")

    if pred in candidates:
        return True

    # Fuzzy search on predicted string
    try:
        results = pycountry.countries.search_fuzzy(pred)
        if results and results[0].alpha_2.lower() == code:
            return True
    except LookupError:
        pass
    return False


# ── Haversine ────────────────────────────────────────────────────────────────


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── Coordinate / country parsing ─────────────────────────────────────────────


def _parse_pred_country(country_result: str) -> str:
    """Extract country name from the 'Country: X' pattern in country_result string."""
    if not country_result:
        return ""
    # Scan lines for "Country: X"
    for line in country_result.splitlines():
        m = re.match(r"^\s*Country:\s*(.+)", line, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # Strip trailing punctuation/notes in parentheses
            name = re.sub(r"\s*\(.*\).*$", "", name).strip()
            return _normalize_country(name)
    # Fallback: first non-empty line
    for line in country_result.splitlines():
        line = line.strip()
        if line:
            return _normalize_country(line[:80])
    return ""


def _parse_coordinates(raw: Any) -> tuple[float, float] | None:
    """Parse coordinates from the coordinates field (dict or string)."""
    if isinstance(raw, dict):
        try:
            lat = float(raw.get("lat") or raw.get("latitude") or 0)
            lon = float(raw.get("lon") or raw.get("lng") or raw.get("longitude") or 0)
            if lat != 0 or lon != 0:
                return (lat, lon)
        except (TypeError, ValueError):
            pass
        return None
    if isinstance(raw, str) and raw.strip():
        # Try "lat, lon" format
        parts = raw.strip().split(",")
        if len(parts) == 2:
            try:
                return (float(parts[0].strip()), float(parts[1].strip()))
            except ValueError:
                pass
    return None


def _extract_coords_from_result_text(country_result: str) -> tuple[float, float] | None:
    """Fallback: extract 'Coordinates: lat, lon' from country_result text."""
    if not country_result:
        return None
    for line in country_result.splitlines():
        m = re.match(r"^\s*Coordinates?:\s*([0-9.\-]+)\s*,\s*([0-9.\-]+)", line, re.IGNORECASE)
        if m:
            try:
                return (float(m.group(1)), float(m.group(2)))
            except ValueError:
                pass
    return None


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class RunRecord:
    """Normalized view of a single image's re-guess run + ground truth."""

    image_id: str                  # e.g. "1NJsXTxIF9GGMDxC_1"
    image_path: str                # path the council saw

    # Ground truth
    truth_country_code: str        # ISO alpha-2, lowercase, e.g. "kg"
    truth_country_name: str        # canonical name, lowercase
    truth_lat: float
    truth_lon: float

    # Prediction
    pred_country: str              # normalized model output
    pred_lat: float | None
    pred_lon: float | None
    final_reasoning: str

    # Re-guess specific
    r1_assessments: dict[str, Any]   # raw round_1_assessments
    r2_assessments: dict[str, Any]   # raw round_2_assessments

    # Timing
    total_seconds: float | None

    # Raw (kept for judge module)
    raw: dict[str, Any] = field(default_factory=dict)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_correct(self) -> bool:
        return countries_match(self.pred_country, self.truth_country_code)

    @property
    def haversine_km(self) -> float | None:
        if self.pred_lat is None or self.pred_lon is None:
            return None
        return _haversine_km(self.pred_lat, self.pred_lon, self.truth_lat, self.truth_lon)

    @property
    def lat_error(self) -> float | None:
        if self.pred_lat is None:
            return None
        return self.pred_lat - self.truth_lat

    @property
    def lon_error(self) -> float | None:
        if self.pred_lon is None:
            return None
        diff = self.pred_lon - self.truth_lon
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff


# ── Ground truth loader ───────────────────────────────────────────────────────


def load_ground_truth(csv_path: Path) -> dict[str, dict]:
    """Return ``{image_id: {country_code, country_name, lat, lon}}``.

    CSV columns: filename, sample_id, round, lat, lng, country_code, ...
    image_id = <sample_id>_<round> (stem of filename).
    """
    out: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # Detect lon column: "lng" or "lon"
        lon_col = "lng" if "lng" in fieldnames else "lon"
        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            image_id = Path(filename).stem
            try:
                lat = float(row["lat"])
                lon = float(row[lon_col])
            except (ValueError, KeyError):
                continue
            code = (row.get("country_code") or "").strip().lower()
            if not code:
                continue
            out[image_id] = {
                "country_code": code,
                "country_name": _country_name_for_code(code),
                "lat": lat,
                "lon": lon,
            }
    return out


# ── Result loader ─────────────────────────────────────────────────────────────


def _load_one(result_dir: Path) -> dict | None:
    f = result_dir / "result.json"
    if not f.exists():
        return None
    with open(f) as fp:
        return json.load(fp)


def load_run(results_dir: Path | str, gt_csv: Path | str) -> list[RunRecord]:
    """Load every result.json under results_dir and pair with ground truth.

    Skips images that have no ground-truth row or no result.json.
    Accepts either a single directory or a list of directories.
    """
    results_dir = Path(results_dir)
    gt_csv = Path(gt_csv)
    truth = load_ground_truth(gt_csv)
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

        country_result = raw.get("country_result") or ""
        pred_country = _parse_pred_country(country_result)

        # Parse coordinates: prefer the coordinates field, fall back to text
        coords_raw = raw.get("coordinates")
        coords = _parse_coordinates(coords_raw)
        if coords is None:
            coords = _extract_coords_from_result_text(country_result)

        timing = raw.get("timing") or {}
        total_seconds: float | None
        try:
            total_seconds = float(timing.get("total_seconds") or 0) or None
        except (TypeError, ValueError):
            total_seconds = None

        records.append(
            RunRecord(
                image_id=image_id,
                image_path=raw.get("image_path") or "",
                truth_country_code=gt["country_code"],
                truth_country_name=gt["country_name"],
                truth_lat=gt["lat"],
                truth_lon=gt["lon"],
                pred_country=pred_country,
                pred_lat=coords[0] if coords else None,
                pred_lon=coords[1] if coords else None,
                final_reasoning=raw.get("final_reasoning") or "",
                r1_assessments=dict(raw.get("round_1_assessments") or {}),
                r2_assessments=dict(raw.get("round_2_assessments") or {}),
                total_seconds=total_seconds,
                raw=raw,
            )
        )

    return records


# ── Assessment helpers ────────────────────────────────────────────────────────


def top1_country(assessment_dict: dict | None) -> str | None:
    """Return the normalized top-1 country from an assessment dict, or None."""
    if not assessment_dict:
        return None
    candidates = assessment_dict.get("candidates") or []
    if not candidates:
        return None
    country = candidates[0].get("country") or ""
    return _normalize_country(country) if country else None


def topk_countries(assessment_dict: dict | None, k: int) -> list[str]:
    """Return normalized top-k countries from an assessment dict."""
    if not assessment_dict:
        return []
    out = []
    for c in (assessment_dict.get("candidates") or [])[:k]:
        name = c.get("country") or ""
        if name:
            out.append(_normalize_country(name))
    return out
