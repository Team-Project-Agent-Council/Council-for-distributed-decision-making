"""Keyed lookup for GeoGuessr reference data.

Provides deterministic retrieval of reference images by category and country.
No embedding model needed, pure dictionary lookup with property filtering.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Reference:
    country: str
    category: str
    image_path: str
    properties: list[str] = field(default_factory=list)


CATEGORY_PRIORITY = [
    "bollards",
    "license_plates",
    "signs_stop",
    "signs_yield",
    "signs_chevrons",
    "signs_pedestrian",
    "utility_poles_full",
    "traffic_lights",
    "post_boxes",
    "sidewalks",
    "signs_speed",
    "signs_bus_stop",
    "signs_directions",
    "signs_railway_crossing",
    "signs_back",
    "signs_posts",
    "signs_road_numbering",
    "signs_animal_warning",
    "signs_street_names",
]

_CATEGORY_ALIASES = {
    "utility_poles": "utility_poles_full",
    "utility poles": "utility_poles_full",
    "poles": "utility_poles_full",
}

def _resolve_category(cat: str) -> str:
    """Resolve a category name to the actual directory name."""
    c = cat.strip().lower()
    return _CATEGORY_ALIASES.get(c, c)

BOLLARD_MATERIALS = {"Metal", "Concrete", "Plastic", "Wood", "Rock"}
BOLLARD_COLORS = {"Red", "White", "Black", "Blue", "Yellow", "Grey", "Orange", "Green", "Brown", "Pink"}

# Bidirectional aliases: maps alternate names to canonical names AND vice versa.
# At index time, data is stored under its normalized name from the file.
# At query time, we try both the raw normalized input AND the alias-resolved form.
_COUNTRY_ALIASES = {
    "uk": "united kingdom",
    "usa": "united states",
    "us": "united states",
    "south korea": "south korea",
    "north korea": "north korea",
    "czechia": "czech republic",
    "czech": "czech republic",
    "ivory coast": "cote d'ivoire",
    "trinidad": "trinidad and tobago",
    "uae": "united arab emirates",
    "dr congo": "democratic republic of the congo",
    "congo": "republic of the congo",
    "east timor": "timor-leste",
    "burma": "myanmar",
    "swaziland": "eswatini",
    "turkiye": "turkey",
    "turkeye": "turkey",
    "bosnia": "bosnia and herzegovina",
    "holland": "netherlands",
    "the netherlands": "netherlands",
}


def _normalize(name: str) -> str:
    s = name.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


class KeyedLookup:
    """Deterministic reference lookup by category and country."""

    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._bollard_db: dict[str, list[dict]] = {}
        self._manifests: dict[str, list[dict]] = {}
        self._road_lines: dict[str, list[str]] = {}
        self._country_index: dict[str, dict[str, list]] = {}
        self._load_all()

    def _load_all(self):
        self._load_bollards()
        self._load_road_lines()
        for cat in CATEGORY_PRIORITY:
            if cat in ("bollards", "road_lines"):
                continue
            self._load_manifest(cat)

    def _load_bollards(self):
        db_path = self._base / "bollards" / "bollard_database.json"
        if not db_path.exists():
            return
        self._bollard_db = json.loads(db_path.read_text())
        idx = {}
        for country, bollards in self._bollard_db.items():
            key = _normalize(country)
            idx[key] = bollards
        self._country_index["bollards"] = idx

    def _load_road_lines(self):
        path = self._base / "road_lines" / "road_lines_by_country.json"
        if not path.exists():
            return
        raw = json.loads(path.read_text())
        idx = {}
        for country, patterns in raw.items():
            idx[_normalize(country)] = patterns
        self._road_lines = raw
        self._country_index["road_lines"] = idx

    def _load_manifest(self, category: str):
        cat_dir = self._base / category
        manifest_path = cat_dir / "manifest.json"
        if not manifest_path.exists():
            return
        entries = json.loads(manifest_path.read_text())
        self._manifests[category] = entries
        idx: dict[str, list] = {}
        for entry in entries:
            key = _normalize(entry["country"])
            if key not in idx:
                idx[key] = []
            idx[key].append(entry)
        self._country_index[category] = idx

    def _resolve_country(self, country: str, category: str | None = None) -> str:
        """Resolve a country name to the key used in the index.

        Strategy:
        1. Normalize the input
        2. If it exists directly in the category index, use it as-is
        3. Try the alias mapping
        4. Fall back to the normalized input
        """
        norm = _normalize(country)

        if category:
            idx = self._country_index.get(category, {})
            if norm in idx:
                return norm

        alias = _COUNTRY_ALIASES.get(norm)
        if alias:
            resolved = _normalize(alias)
            if category:
                idx = self._country_index.get(category, {})
                if resolved in idx:
                    return resolved
            else:
                return resolved

        return norm

    def _resolve_country_all_keys(self, country: str) -> list[str]:
        """Return all possible index keys for a country name.

        Used by available_categories to check across all categories where
        different categories may store the same country under different keys.
        """
        norm = _normalize(country)
        keys = [norm]
        alias = _COUNTRY_ALIASES.get(norm)
        if alias:
            keys.append(_normalize(alias))
        return keys

    def available_categories(self, country_a: str, country_b: str) -> list[str]:
        """Return categories that have data for at least one of the two countries."""
        keys_a = self._resolve_country_all_keys(country_a)
        keys_b = self._resolve_country_all_keys(country_b)
        available = []
        for cat in CATEGORY_PRIORITY:
            idx = self._country_index.get(cat, {})
            found = any(k in idx for k in keys_a) or any(k in idx for k in keys_b)
            if found:
                display = cat.replace("_full", "")
                available.append(display)
        if "road_lines" not in available:
            rl_idx = self._country_index.get("road_lines", {})
            found = any(k in rl_idx for k in keys_a) or any(k in rl_idx for k in keys_b)
            if found:
                available.append("road_lines")
        return available

    def available_categories_multi(self, countries: list[str]) -> list[str]:
        """Return categories that have data for ANY of the given countries."""
        all_keys: list[str] = []
        for c in countries:
            all_keys.extend(self._resolve_country_all_keys(c))
        available = []
        for cat in CATEGORY_PRIORITY:
            idx = self._country_index.get(cat, {})
            if any(k in idx for k in all_keys):
                display = cat.replace("_full", "")
                available.append(display)
        if "road_lines" not in available:
            rl_idx = self._country_index.get("road_lines", {})
            if any(k in rl_idx for k in all_keys):
                available.append("road_lines")
        return available

    def _get_entries(self, category: str, country: str) -> list:
        """Get index entries for a country in a category, trying all key variants."""
        idx = self._country_index.get(category, {})
        key = self._resolve_country(country, category)
        entries = idx.get(key, [])
        if entries:
            return entries
        # Fallback: try all keys
        for k in self._resolve_country_all_keys(country):
            entries = idx.get(k, [])
            if entries:
                return entries
        return []

    def lookup_bollards(
        self, country: str, materials: list[str] | None = None, colors: list[str] | None = None, max_refs: int = 50
    ) -> list[Reference]:
        """Lookup bollards for a country, optionally filtered by properties."""
        bollards = self._get_entries("bollards", country)
        if not bollards:
            return []

        if materials or colors:
            scored = []
            for b in bollards:
                props = set(b.get("properties", []))
                score = 0
                if materials:
                    score += len(props & set(materials))
                if colors:
                    score += len(props & set(colors))
                if score > 0:
                    scored.append((score, b))
            scored.sort(key=lambda x: -x[0])
            bollards = [b for _, b in scored[:max_refs]]
        else:
            bollards = bollards[:max_refs]

        results = []
        for b in bollards:
            img_path = self._base / "bollards" / b["image"]
            if img_path.exists():
                results.append(Reference(
                    country=country,
                    category="bollards",
                    image_path=str(img_path),
                    properties=b.get("properties", []),
                ))
        return results[:max_refs]

    def lookup_category(self, category: str, country: str, max_refs: int = 50) -> list[Reference]:
        """Generic lookup for any non-bollard category."""
        entries = self._get_entries(category, country)
        if not entries:
            return []

        results = []
        for entry in entries[:max_refs]:
            img_rel = entry.get("image", "")
            img_path = self._base / category / img_rel
            if img_path.exists():
                results.append(Reference(
                    country=country,
                    category=category,
                    image_path=str(img_path),
                    properties=entry.get("properties", []),
                ))
        return results[:max_refs]

    def lookup_road_lines(self, country: str) -> str | None:
        """Return road line patterns as numbered list."""
        entries = self._get_entries("road_lines", country)
        if not entries:
            return None
        if len(entries) == 1:
            return entries[0]
        return "; ".join(f"({i+1}) {e}" for i, e in enumerate(entries))

    def road_line_patterns(self, country: str) -> list[str]:
        """Return raw road-line pattern strings for a country, e.g.
        ['Outside: Yellow | Inside: White', 'Outside: Red | Inside: White'].

        Empty list if no entry exists.
        """
        entries = self._get_entries("road_lines", country)
        return [str(e) for e in entries if isinstance(e, str)]

    def fetch_references(
        self,
        country: str,
        categories: list[str],
        bollard_materials: list[str] | None = None,
        bollard_colors: list[str] | None = None,
        max_total: int = 50,
    ) -> tuple[list[Reference], str | None]:
        """Fetch references for a country across multiple categories.

        Returns (image_references, road_lines_text).
        Respects priority order and max_total cap for images.
        """
        refs: list[Reference] = []
        road_text = None

        resolved = [_resolve_category(c) for c in categories]
        ordered = [c for c in CATEGORY_PRIORITY if c in resolved]
        if "road_lines" in resolved:
            road_text = self.lookup_road_lines(country)

        for cat in ordered:
            if len(refs) >= max_total:
                break
            remaining = max_total - len(refs)

            if cat == "bollards":
                new = self.lookup_bollards(country, bollard_materials, bollard_colors, max_refs=remaining)
            else:
                new = self.lookup_category(cat, country, max_refs=remaining)
            refs.extend(new)

        return refs[:max_total], road_text
