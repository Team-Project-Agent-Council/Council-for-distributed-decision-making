"""Canonical region taxonomy for the VLM Council.

The 12 regions used in every prompt and the country→region reverse lookup are
loaded once from ``data/regions.json`` at import time. This is the only source
of truth, agents that talk about regions must import ``CANONICAL_REGIONS`` and
use ``normalize_region`` / ``country_to_region`` instead of repeating the names.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path


CANONICAL_REGIONS: list[str] = [
    "Europe",
    "East Asia",
    "Southeast Asia",
    "South Asia",
    "Central Asia",
    "Middle East",
    "North Africa",
    "Sub-Saharan Africa",
    "North America",
    "Central America & Caribbean",
    "South America",
    "Oceania",
]

REGIONS_PROMPT_LIST: str = ", ".join(CANONICAL_REGIONS)


_ALIASES: dict[str, str] = {
    # Spelling / punctuation drift
    "south east asia": "Southeast Asia",
    "south-east asia": "Southeast Asia",
    "se asia": "Southeast Asia",
    "ne asia": "East Asia",
    "far east": "East Asia",
    "central america and caribbean": "Central America & Caribbean",
    "central america": "Central America & Caribbean",
    "caribbean": "Central America & Caribbean",
    "latin america": "South America",
    "north africa & middle east": "Middle East",
    "mena": "Middle East",
    "sub saharan africa": "Sub-Saharan Africa",
    "subsaharan africa": "Sub-Saharan Africa",
    "africa": "Sub-Saharan Africa",
    # Sub-region collapses
    "western europe": "Europe",
    "eastern europe": "Europe",
    "southern europe": "Europe",
    "northern europe": "Europe",
    "scandinavia": "Europe",
    "balkans": "Europe",
    "south eastern europe": "Europe",
    "south-eastern europe": "Europe",
    "southeastern europe": "Europe",
    # Common LLM slippage
    "australia": "Oceania",
    "australasia": "Oceania",
    "pacific": "Oceania",
    "polynesia": "Oceania",
    "melanesia": "Oceania",
    "micronesia": "Oceania",
    "russia": "Europe",
    "russian federation": "Europe",
    "central asia / russia": "Central Asia",
}


def _normalize_key(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


_CANONICAL_LOOKUP = {_normalize_key(r): r for r in CANONICAL_REGIONS}


def normalize_region(name: str | None) -> str | None:
    """Map any reasonable region string back to one of CANONICAL_REGIONS.

    Returns None if the input cannot be resolved.
    """
    if not name:
        return None
    key = _normalize_key(name)
    if key in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[key]
    if key in _ALIASES:
        return _ALIASES[key]
    # Substring fallback: "western europe" → contains "europe"
    for canon_key, canon in _CANONICAL_LOOKUP.items():
        if canon_key in key or key in canon_key:
            return canon
    return None


# country → region reverse lookup, populated at import from regions.json

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "regions.json"


def _load_country_index() -> tuple[dict[str, str], dict[str, list[str]]]:
    if not _DATA_PATH.exists():
        return {}, {}
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    # Validate: every region in regions.json must be canonical
    for region in raw.keys():
        if region not in CANONICAL_REGIONS:
            raise ValueError(
                f"regions.json contains non-canonical region '{region}'. "
                f"Valid: {CANONICAL_REGIONS}"
            )
    by_country: dict[str, str] = {}
    by_region: dict[str, list[str]] = {r: list(cs) for r, cs in raw.items()}
    for region, countries in raw.items():
        for country in countries:
            by_country[_normalize_key(country)] = region
    return by_country, by_region


_COUNTRY_TO_REGION, _REGION_TO_COUNTRIES = _load_country_index()


# Country-name aliases the LLM may produce that differ from the canonical name
# in regions.json. Keys must already be _normalize_key'd.
_COUNTRY_ALIASES: dict[str, str] = {
    "turkiye": "Turkey",
    "czechia": "Czech Republic",
    "uk": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "america": "United States",
    "united states of america": "United States",
    "south korea": "South Korea",
    "republic of korea": "South Korea",
    "north korea": "North Korea",
    "dprk": "North Korea",
    "uae": "United Arab Emirates",
    "vietnam": "Vietnam",
    "viet nam": "Vietnam",
    "myanmar (burma)": "Myanmar",
    "burma": "Myanmar",
    "ivory coast": None,  # Côte d'Ivoire, not in our data; flag as None
    "cote d'ivoire": None,
    "swaziland": "Eswatini",
    "macedonia": "North Macedonia",
    "republic of macedonia": "North Macedonia",
    "russian federation": "Russia",
    "bosnia": "Bosnia and Herzegovina",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
}


def country_to_region(country: str | None) -> str | None:
    """Look up the canonical region for a country name (case/diacritic-insensitive)."""
    if not country:
        return None
    key = _normalize_key(country)
    if key in _COUNTRY_TO_REGION:
        return _COUNTRY_TO_REGION[key]
    aliased = _COUNTRY_ALIASES.get(key)
    if aliased:
        return _COUNTRY_TO_REGION.get(_normalize_key(aliased))
    return None


def _lookup_canonical(key: str) -> str | None:
    """Return the original-cased country name in regions.json matching `key`, or None."""
    if key not in _COUNTRY_TO_REGION:
        return None
    for _region, countries in _REGION_TO_COUNTRIES.items():
        for c in countries:
            if _normalize_key(c) == key:
                return c
    return None


def canonical_country_name(country: str | None) -> str | None:
    """Resolve a country name to its canonical form as used in regions.json.

    Returns None if the input cannot be matched (and is not a known alias to
    something inside the dataset). Single-pass lookup: aliases are resolved at
    most once, so self-referential alias entries cannot loop.
    """
    if not country:
        return None
    key = _normalize_key(country)
    canon = _lookup_canonical(key)
    if canon:
        return canon
    aliased = _COUNTRY_ALIASES.get(key)
    if not aliased:
        return None
    alias_key = _normalize_key(aliased)
    if alias_key == key:
        return None
    return _lookup_canonical(alias_key)


def countries_in_region(region: str) -> list[str]:
    """Return all known countries in a canonical region. Empty list if region unknown."""
    canon = normalize_region(region)
    if not canon:
        return []
    return list(_REGION_TO_COUNTRIES.get(canon, []))
