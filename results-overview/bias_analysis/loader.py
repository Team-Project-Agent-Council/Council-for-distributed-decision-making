"""Self-contained loader for cross-approach bias analysis.

Reads per-image ``result.json`` files (council dict-coords AND baseline
string-coords schemas) and pairs them with ground truth into flat ``Record``
objects. Deliberately independent of any single approach's eval package so the
aggregate view stays decoupled.
"""

from __future__ import annotations

import csv
import json
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path

_CENTROIDS_PATH = Path(__file__).parent / "country_centroids.json"
_CENTROIDS: dict[str, tuple[float, float]] = {}
if _CENTROIDS_PATH.exists():
    _raw = json.loads(_CENTROIDS_PATH.read_text())
    _CENTROIDS = {k: (float(v[0]), float(v[1])) for k, v in _raw.items()}


def _normalize_country(name: str) -> str:
    if not name:
        return ""
    return name.strip().lower().rstrip(string.punctuation + " ").strip()


def _centroid_for_country(country_name: str) -> tuple[float, float] | None:
    if not country_name or not _CENTROIDS:
        return None
    norm = _normalize_country(country_name)
    if norm in _CENTROIDS:
        return _CENTROIDS[norm]
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


@dataclass
class Record:
    approach: str
    image_id: str
    truth_lat: float
    truth_lng: float
    truth_country_code: str
    pred_lat: float | None
    pred_lng: float | None
    pred_country: str

    @property
    def has_coords(self) -> bool:
        return self.pred_lat is not None and self.pred_lng is not None

    @property
    def haversine_km(self) -> float | None:
        if not self.has_coords:
            return None
        r = 6371.0
        p1, p2 = math.radians(self.truth_lat), math.radians(self.pred_lat)
        dphi = math.radians(self.pred_lat - self.truth_lat)
        dlmb = math.radians(self.pred_lng - self.truth_lng)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

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


def load_ground_truth(csv_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            try:
                lat = float(row["lat"])
                lng = float(row["lng"])
            except (ValueError, KeyError):
                continue
            out[Path(filename).stem] = {
                "lat": lat,
                "lng": lng,
                "country_code": (row.get("country_code") or "").strip().lower(),
            }
    return out


_COORD_LINE = re.compile(r"Coordinates?\s*:\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", re.I)
_COUNTRY_LINE = re.compile(r"Country\s*:\s*([^\n]+)", re.I)


def _parse_coords(raw: dict) -> tuple[float, float] | None:
    """Handle dict {"lat","lng"}, string "lat,lng", then country_result text."""
    coords = raw.get("coordinates")
    if isinstance(coords, dict):
        lat, lng = coords.get("lat"), coords.get("lng")
        if lat is not None and lng is not None:
            return float(lat), float(lng)
    if isinstance(coords, str) and coords.strip():
        try:
            lat_s, lng_s = coords.split(",")
            return float(lat_s), float(lng_s)
        except (ValueError, TypeError):
            pass
    m = _COORD_LINE.search(raw.get("country_result", "") or "")
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _parse_country(raw: dict) -> str:
    m = _COUNTRY_LINE.search(raw.get("country_result", "") or "")
    if m:
        # strip trailing period / parenthetical
        return _normalize_country(m.group(1).split("(")[0])
    parsed = raw.get("parsed")
    if isinstance(parsed, dict) and parsed.get("Country"):
        return _normalize_country(str(parsed["Country"]))
    return ""


def load_run(approach: str, results_dir: Path, gt_csv: Path) -> list[Record]:
    truth = load_ground_truth(gt_csv)
    records: list[Record] = []
    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue
        gt = truth.get(result_dir.name)
        if gt is None:
            continue
        rf = result_dir / "result.json"
        if not rf.exists():
            continue
        try:
            raw = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        pred_country = _parse_country(raw)
        coords = _parse_coords(raw)
        # (0,0) is the null-island sentinel — treat as missing.
        if coords is not None and coords[0] == 0.0 and coords[1] == 0.0:
            coords = None
        if coords is None and pred_country:
            coords = _centroid_for_country(pred_country)

        records.append(Record(
            approach=approach,
            image_id=result_dir.name,
            truth_lat=gt["lat"],
            truth_lng=gt["lng"],
            truth_country_code=gt["country_code"],
            pred_lat=coords[0] if coords else None,
            pred_lng=coords[1] if coords else None,
            pred_country=pred_country,
        ))
    return records
