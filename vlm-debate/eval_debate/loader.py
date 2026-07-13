"""Data loader for the Debate VLM Council approach.

GT CSV columns: filename, sample_id, round, lat, lng, country_code
Results dir: per-image subdirs each containing result.json
"""

from __future__ import annotations

import csv
import json
import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

AGENT_NAMES = ["linguistic", "landscape", "botanics", "regulatory", "meta"]


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Country normalisation
# ---------------------------------------------------------------------------

# CANONICAL country matching — single source of truth for the whole submission.
# The alias table maps alpha-2 code -> set of accepted normalized names.
# Kept behaviourally identical across all approaches' eval loaders and
# vlm_council/evaluate.py so every approach matches countries the same way.
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


def _extract_country(country_result: str) -> str:
    if not country_result:
        return ""
    for line in country_result.splitlines():
        m = re.match(r"Country:\s*(.+)", line.strip(), re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".")
    return country_result.splitlines()[0].strip()


def _parse_coords(coordinates) -> tuple[float | None, float | None]:
    if not coordinates:
        return None, None
    if isinstance(coordinates, dict):
        lat = coordinates.get("lat") or coordinates.get("latitude")
        lon = coordinates.get("lon") or coordinates.get("lng") or coordinates.get("longitude")
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            return None, None
    if isinstance(coordinates, str):
        parts = coordinates.split(",")
        if len(parts) == 2:
            try:
                return float(parts[0].strip()), float(parts[1].strip())
            except ValueError:
                return None, None
    return None, None


def top1_country(assessment: dict | None) -> str | None:
    if not assessment:
        return None
    cands = assessment.get("candidates", [])
    if cands and isinstance(cands[0], dict):
        return cands[0].get("country")
    return None


def topk_countries(assessment: dict | None, k: int = 3) -> list[str]:
    if not assessment:
        return []
    cands = assessment.get("candidates", [])
    return [c.get("country", "") for c in cands[:k] if isinstance(c, dict) and c.get("country")]


# ---------------------------------------------------------------------------
# RunRecord
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    image_id: str
    truth_country_name: str
    truth_country_code: str
    truth_lat: float
    truth_lon: float
    pred_country: str
    pred_lat: float | None
    pred_lon: float | None
    is_correct: bool
    haversine_km: float | None
    total_seconds: float | None
    r1_assessments: dict = field(default_factory=dict)
    debate: dict = field(default_factory=dict)
    final_reasoning: str = ""
    image_path: str = ""

    @property
    def debate_happened(self) -> bool:
        return any(
            p.get("exchanges")
            for p in self.debate.get("pairings", [])
        )

    @property
    def total_debate_rounds(self) -> int:
        return int(self.debate.get("total_rounds", 0) or 0)

    @property
    def termination_reason(self) -> str:
        decisions = self.debate.get("moderator_decisions", [])
        for d in reversed(decisions):
            r = d.get("termination_reason", "")
            if r:
                return r
        return ""

    @property
    def any_revision(self) -> bool:
        return any(
            ex.get("revised", False)
            for p in self.debate.get("pairings", [])
            for ex in p.get("exchanges", [])
        )


# ---------------------------------------------------------------------------
# load_run
# ---------------------------------------------------------------------------

def load_run(results_dir: Path, gt_csv: Path) -> list[RunRecord]:
    results_dir = Path(results_dir)
    gt_csv = Path(gt_csv)

    # Load ground truth, columns: filename, sample_id, round, lat, lng, country_code
    gt: dict[str, dict] = {}
    with open(gt_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # image_id comes from the directory name which is "<sample_id>_<round>"
            sample_id = row.get("sample_id", "")
            rnd = row.get("round", "")
            img_id = f"{sample_id}_{rnd}"
            gt[img_id] = row

    records: list[RunRecord] = []
    for img_dir in sorted(results_dir.iterdir()):
        if not img_dir.is_dir():
            continue
        result_file = img_dir / "result.json"
        if not result_file.exists():
            continue
        try:
            data = json.loads(result_file.read_text())
        except Exception:
            continue
        if data.get("error"):
            continue

        image_id = img_dir.name
        gt_row = gt.get(image_id)
        if gt_row is None:
            continue

        try:
            truth_lat = float(gt_row.get("lat", 0))
            truth_lon = float(gt_row.get("lng", 0))
        except (ValueError, TypeError):
            truth_lat, truth_lon = 0.0, 0.0

        truth_code = (gt_row.get("country_code") or "").upper()
        # Get country name from pycountry
        truth_name = truth_code
        try:
            import pycountry
            hit = pycountry.countries.get(alpha_2=truth_code)
            if hit:
                truth_name = hit.name
        except ImportError:
            pass

        pred_country = _extract_country(data.get("country_result", ""))
        pred_lat, pred_lon = _parse_coords(data.get("coordinates"))

        correct = countries_match(pred_country, truth_code) or countries_match(pred_country, truth_name)

        hav = None
        if pred_lat is not None and pred_lon is not None:
            hav = haversine_km(truth_lat, truth_lon, pred_lat, pred_lon)

        timing = data.get("timing") or {}
        total_sec = None
        if isinstance(timing, dict):
            total_sec = timing.get("total_seconds")
            if total_sec is None:
                total_sec = timing.get("total")

        records.append(RunRecord(
            image_id=image_id,
            truth_country_name=truth_name,
            truth_country_code=truth_code,
            truth_lat=truth_lat,
            truth_lon=truth_lon,
            pred_country=pred_country,
            pred_lat=pred_lat,
            pred_lon=pred_lon,
            is_correct=correct,
            haversine_km=hav,
            total_seconds=float(total_sec) if total_sec is not None else None,
            r1_assessments=data.get("round_1_assessments", {}),
            debate=data.get("debate", {}),
            final_reasoning=data.get("final_reasoning", ""),
            image_path=data.get("image_path", "") or "",
        ))

    return records
