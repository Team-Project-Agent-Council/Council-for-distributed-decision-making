"""Pure metric functions - no I/O, no side effects."""

from __future__ import annotations

import math
import re


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two GPS points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def geoguessr_score(distance_km: float) -> int:
    """Standard GeoGuessr exponential decay: 5000 * exp(-d / 2000)."""
    if math.isnan(distance_km):
        return 0
    return round(5000 * math.exp(-distance_km / 2000))


def country_match(predicted: str, ground_truth: str) -> bool:
    """Case-insensitive match after normalising via pycountry when possible."""
    predicted = predicted.strip()
    ground_truth = ground_truth.strip()

    if predicted.lower() == ground_truth.lower():
        return True

    try:
        import pycountry

        def _normalise(name: str) -> str | None:
            c = pycountry.countries.get(name=name)
            if c:
                return c.name
            # fuzzy search as fallback
            results = pycountry.countries.search_fuzzy(name)
            return results[0].name if results else None

        norm_pred = _normalise(predicted)
        norm_gt = _normalise(ground_truth)
        if norm_pred and norm_gt:
            return norm_pred.lower() == norm_gt.lower()
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(
    r"(?:Coordinates?|GPS)[:\s]+(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_COUNTRY_RE = re.compile(r"Country[:\s]+([^\n]+)", re.IGNORECASE)


def parse_country_result(text: str) -> tuple[str, float, float]:
    """Parse judge output into (country, lat, lon).

    Expected format::

        Country: <name>
        Coordinates: <lat>, <lon>
        Reasoning: ...

    Returns ``("", float("nan"), float("nan"))`` on parse failure.
    """
    country = ""
    lat = float("nan")
    lon = float("nan")

    if not text:
        return country, lat, lon

    m = _COUNTRY_RE.search(text)
    if m:
        country = m.group(1).strip().rstrip(".")

    m = _COORD_RE.search(text)
    if m:
        try:
            lat, lon = float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    return country, lat, lon
