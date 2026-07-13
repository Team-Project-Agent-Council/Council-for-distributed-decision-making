"""Resolve free-form country names to plonkit slugs.

Uses a lazy-loaded mapping from plonkit_meta.json with a multi-step
resolution chain: exact slug → title → ISO alpha-2 → pycountry fuzzy.
"""
from __future__ import annotations

import json
from pathlib import Path

from geoguessr_rag.config import PLONKIT_META_PATH

_SLUG_MAP: dict[str, str] | None = None


def _build_slug_map() -> dict[str, str]:
    """Build a lookup: lowercased key → slug."""
    with open(PLONKIT_META_PATH, encoding="utf-8") as f:
        data = json.load(f)

    mapping: dict[str, str] = {}
    for slug, entry in data.items():
        mapping[slug.lower()] = slug
        pub = entry.get("public", {})
        title = pub.get("title", "")
        if title:
            mapping[title.lower()] = slug
        code = pub.get("code", "")
        if code:
            mapping[code.lower()] = slug

    return mapping


def _get_slug_map() -> dict[str, str]:
    global _SLUG_MAP
    if _SLUG_MAP is None:
        _SLUG_MAP = _build_slug_map()
    return _SLUG_MAP


def resolve_to_slug(country_name: str) -> str | None:
    """Resolve a free-form country name to its plonkit slug.

    Resolution chain:
    1. Direct slug match (lowercased)
    2. Title match (lowercased)
    3. ISO alpha-2 code match
    4. pycountry fuzzy search → slug
    """
    if not country_name:
        return None

    slug_map = _get_slug_map()
    key = country_name.strip().lower()

    if key in slug_map:
        return slug_map[key]

    # Try with common transformations
    hyphenated = key.replace(" ", "-")
    if hyphenated in slug_map:
        return slug_map[hyphenated]

    # pycountry fuzzy match → get the standard name → check slug map
    try:
        import pycountry
        results = pycountry.countries.search_fuzzy(country_name)
        if results:
            c = results[0]
            for candidate in [c.name.lower(), c.alpha_2.lower(), getattr(c, "common_name", "").lower()]:
                if candidate and candidate in slug_map:
                    return slug_map[candidate]
            # Try hyphenated name
            hyphenated_name = c.name.lower().replace(" ", "-")
            if hyphenated_name in slug_map:
                return slug_map[hyphenated_name]
            common = getattr(c, "common_name", "")
            if common:
                hyphenated_common = common.lower().replace(" ", "-")
                if hyphenated_common in slug_map:
                    return slug_map[hyphenated_common]
    except Exception:
        pass

    return None
