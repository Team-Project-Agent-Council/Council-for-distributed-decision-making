"""Pure metric functions — no I/O, no side effects."""

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


_COMMON_NAME_TO_ALPHA2 = {
    "turkey": "tr",
    "ivory coast": "ci",
    "cote d'ivoire": "ci",
    "congo": "cd",
    "east timor": "tl",
    "swaziland": "sz",
    "burma": "mm",
    "macedonia": "mk",
    "palestine": "ps",
}


def country_match(predicted: str, ground_truth: str) -> bool:
    """Case-insensitive match after normalising via pycountry when possible."""
    predicted = predicted.strip()
    ground_truth = ground_truth.strip()

    if predicted.lower() == ground_truth.lower():
        return True

    try:
        import pycountry

        def _normalise(name: str) -> str | None:
            low = name.lower()
            if low in _COMMON_NAME_TO_ALPHA2:
                return _COMMON_NAME_TO_ALPHA2[low]
            c = pycountry.countries.get(name=name)
            if c:
                return c.alpha_2.lower()
            c = pycountry.countries.get(alpha_2=name.upper())
            if c:
                return c.alpha_2.lower()
            c = pycountry.countries.get(alpha_3=name.upper())
            if c:
                return c.alpha_2.lower()
            try:
                results = pycountry.countries.search_fuzzy(name)
                return results[0].alpha_2.lower() if results else None
            except LookupError:
                return None

        norm_pred = _normalise(predicted)
        norm_gt = _normalise(ground_truth)
        if norm_pred and norm_gt:
            return norm_pred == norm_gt
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(
    r"(?:\*{0,2}Coordinates?\*{0,2}|GPS)[:\s]*\*{0,2}\s*(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_COUNTRY_RE = re.compile(r"\*{0,2}Country\*{0,2}[:\s]+\*{0,2}([^*\n]+)", re.IGNORECASE)


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

    coords_m = re.search(r"(?i)^Coordinates:\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)$", text, re.MULTILINE)
    if coords_m:
        try:
            lat = float(coords_m.group(1))
            lon = float(coords_m.group(2))
        except ValueError:
            pass

    return country, lat, lon


def snap_coords_to_country(country: str, lat: float, lon: float) -> tuple[float, float]:
    """Return (lat, lon) snapped to country centroid if coords are (0,0) or implausible.

    Implausible means the reverse-geocoded country doesn't match the claimed one.
    Uses Nominatim for both reverse-geocoding the check and forward-geocoding the centroid.
    """
    from geopy.geocoders import Nominatim
    import math as _math

    geolocator = Nominatim(user_agent="llm_council_snap", timeout=5)

    # Treat (0, 0) as clearly wrong — skip straight to centroid
    if _math.isnan(lat) or _math.isnan(lon) or (abs(lat) < 0.01 and abs(lon) < 0.01):
        return _get_centroid(country, geolocator)

    # Reverse-geocode to check if coords actually land in the claimed country
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
        if location and location.raw.get("address"):
            geocoded_cc = location.raw["address"].get("country_code", "").lower()
            try:
                import pycountry
                claimed_cc = pycountry.countries.search_fuzzy(country)[0].alpha_2.lower()
            except Exception:
                claimed_cc = ""
            if claimed_cc and geocoded_cc and geocoded_cc != claimed_cc:
                return _get_centroid(country, geolocator)
    except Exception:
        pass

    return lat, lon


def _get_centroid(country: str, geolocator=None) -> tuple[float, float]:
    try:
        if geolocator is None:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="llm_council_centroid", timeout=5)
        location = geolocator.geocode(country, exactly_one=True)
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass
    return float("nan"), float("nan")
