"""Load Hub-and-Spoke result.json files + ground truth into RunRecord objects.

Single source of truth for the Hub-and-Spoke evaluation suite.
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


# ---------------------------------------------------------------------------
# Country normalisation helpers (self-contained, no vlm_council import)
# ---------------------------------------------------------------------------

# Canonical alias table: maps ISO alpha-2 code -> set of accepted normalized names.
# Version-independent (does not depend on installed pycountry release naming).
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


def _normalize_country(name: str) -> str:
    """Lowercase + strip surrounding whitespace/punctuation."""
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
    pred = _normalize_country(predicted)
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
            candidates.add(_normalize_country(val))
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


def _country_name_for_code(code: str) -> str:
    """Best-effort ISO code -> canonical name."""
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


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _extract_country(country_result: str) -> str:
    """Parse 'Country: X' from the country_result field."""
    if not country_result:
        return ""
    for line in country_result.splitlines():
        m = re.match(r"(?i)country\s*:\s*(.+)", line.strip())
        if m:
            return m.group(1).strip().rstrip(".,;")
    return ""


def _extract_coordinates(country_result: str) -> tuple[float, float] | None:
    """Parse 'Coordinates: lat, lon' from country_result."""
    if not country_result:
        return None
    for line in country_result.splitlines():
        m = re.match(r"(?i)coordinates?\s*:\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)", line.strip())
        if m:
            try:
                return float(m.group(1)), float(m.group(2))
            except ValueError:
                pass
    return None


def _parse_response_candidates(agent_response: str) -> list[dict]:
    """Extract candidates list from an agent_response JSON string."""
    if not agent_response:
        return []
    text = agent_response.strip()
    # Strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data.get("candidates") or []
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Try to extract embedded JSON object
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start: i + 1])
                        if isinstance(data, dict):
                            return data.get("candidates") or []
                    except Exception:
                        pass
                    break
    return []


# ---------------------------------------------------------------------------
# RunRecord
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """Normalised view of a single image's Hub-and-Spoke run + ground truth."""

    image_id: str
    image_path: str

    # Ground truth
    truth_country_code: str
    truth_country_name: str
    truth_lat: float
    truth_lon: float

    # Prediction
    pred_country: str
    pred_lat: float | None
    pred_lon: float | None
    final_reasoning: str

    # Hub-and-Spoke specific
    assessments: dict[str, dict]      # initial agent assessments keyed by agent name
    discussion_log: list[dict]        # list of {round_number, judge_question, target_agent, agent_response}
    discussion_rounds: int            # number of hub rounds that happened (0-3)

    # Timing
    total_seconds: float | None

    # Raw for judge
    raw: dict[str, Any] = field(default_factory=dict)

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


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path: Path) -> dict[str, dict]:
    """Return {image_id: {country_code, country_name, lat, lon}}.

    CSV columns: filename, sample_id, round, lat, lng, country_code, ...
    image_id = stem of filename (e.g. '1NJsXTxIF9GGMDxC_1')
    """
    out: dict[str, dict] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip()
            if not filename:
                continue
            image_id = Path(filename).stem
            try:
                lat = float(row["lat"])
                lon = float(row["lng"])
            except (ValueError, KeyError):
                continue
            code = row.get("country_code", "").strip().lower()
            out[image_id] = {
                "country_code": code,
                "country_name": _country_name_for_code(code),
                "lat": lat,
                "lon": lon,
            }
    return out


# ---------------------------------------------------------------------------
# Per-image loader
# ---------------------------------------------------------------------------

def _load_one(result_dir: Path) -> dict | None:
    f = result_dir / "result.json"
    if not f.exists():
        return None
    with open(f) as fp:
        return json.load(fp)


def _parse_coords_field(raw_coord) -> tuple[float, float] | None:
    """Parse the 'coordinates' field which may be a dict, string, or empty."""
    if isinstance(raw_coord, dict):
        try:
            return float(raw_coord["lat"]), float(raw_coord["lon"])
        except (KeyError, ValueError, TypeError):
            return None
    if isinstance(raw_coord, str) and raw_coord.strip():
        try:
            parts = raw_coord.split(",")
            if len(parts) == 2:
                return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    return None


def load_run(results_dir: Path, gt_csv: Path) -> list[RunRecord]:
    """Load every result.json under results_dir and pair with ground truth."""
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

        country_result = raw.get("country_result", "") or ""
        pred_country_raw = _extract_country(country_result)
        coords = _extract_coordinates(country_result)
        if coords is None:
            coords = _parse_coords_field(raw.get("coordinates"))

        timing = raw.get("timing") or {}
        total_seconds: float | None = None
        if timing.get("total_seconds") is not None:
            try:
                total_seconds = float(timing["total_seconds"])
            except (ValueError, TypeError):
                pass

        records.append(
            RunRecord(
                image_id=image_id,
                image_path=raw.get("image_path", ""),
                truth_country_code=gt["country_code"],
                truth_country_name=gt["country_name"],
                truth_lat=gt["lat"],
                truth_lon=gt["lon"],
                pred_country=_normalize_country(pred_country_raw),
                pred_lat=coords[0] if coords else None,
                pred_lon=coords[1] if coords else None,
                final_reasoning=raw.get("final_reasoning", "") or "",
                assessments=dict(raw.get("assessments", {}) or {}),
                discussion_log=list(raw.get("discussion_log", []) or []),
                discussion_rounds=int(raw.get("discussion_rounds", 0) or 0),
                total_seconds=total_seconds,
                raw=raw,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Helpers used across modules
# ---------------------------------------------------------------------------

def top1_country(assessment_dict: dict | None) -> str | None:
    """Return the normalised top-1 country from an agent assessment."""
    if not assessment_dict:
        return None
    candidates = assessment_dict.get("candidates") or []
    if not candidates:
        return None
    country = candidates[0].get("country", "")
    return _normalize_country(country) if country else None


def topk_countries(assessment_dict: dict | None, k: int = 3) -> list[str]:
    """Return the first k normalised country names from an agent assessment."""
    if not assessment_dict:
        return []
    out = []
    for c in (assessment_dict.get("candidates") or [])[:k]:
        name = c.get("country", "")
        if name:
            out.append(_normalize_country(name))
    return out


def parse_discussion_for_agent(record: RunRecord, agent_name: str) -> list[dict]:
    """Extract all discussion_log entries where target_agent == agent_name."""
    return [
        entry for entry in record.discussion_log
        if entry.get("target_agent", "") == agent_name
    ]
