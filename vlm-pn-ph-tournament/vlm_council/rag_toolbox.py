"""RAG Toolbox: deterministic grounding operations used by the rag_ground graph node.

Wraps the vendored KeyedLookup and the driving_side.json table. All operations are
pure dictionary lookups, no LLM round-trips. Each operation can return a recovery
warning when its strict version would eliminate ALL candidates (mirrors v10's
_try_road_marking_recovery from the parent repo's vlm_council pipeline).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vlm_council.rag.keyed_lookup import (
    CATEGORY_PRIORITY,
    KeyedLookup,
    Reference,
)


_DRIVING_SIDE_FILENAME = "driving_side.json"


def _normalize(name: str) -> str:
    s = name.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


@dataclass
class DrivingSideResult:
    kept: list[str]
    eliminated: list[tuple[str, str]]
    warning: str | None
    payload: str


@dataclass
class RoadCheckResult:
    by_country: dict[str, tuple[str, str]]  # country -> (verdict, pattern_text)
    warning: str | None
    payload: str


# Regex patterns to extract driving-side and road-line color claims from
# regulatory agent's evidence list (free-text strings).
_LEFT_PATTERNS = [
    r"\bleft[- ]hand\s+(?:traffic|drive)\b",
    r"\bdriving?\s+on\s+the\s+left\b",
    r"\bdriv\w*\s+on\s+left\b",
    r"\bleft[- ]side\s+driv",
    r"\bdriving?\s+side[: ]+\s*left\b",
]
_RIGHT_PATTERNS = [
    r"\bright[- ]hand\s+(?:traffic|drive)\b",
    r"\bdriving?\s+on\s+the\s+right\b",
    r"\bdriv\w*\s+on\s+right\b",
    r"\bright[- ]side\s+driv",
    r"\bdriving?\s+side[: ]+\s*right\b",
]


def parse_driving_side(evidence_strings: list[str]) -> Literal["LEFT", "RIGHT", "UNCLEAR"]:
    """Best-effort extraction of LEFT/RIGHT/UNCLEAR from regulatory agent's evidence."""
    blob = " ".join(str(e) for e in evidence_strings).lower()
    left_hits = sum(1 for p in _LEFT_PATTERNS if re.search(p, blob))
    right_hits = sum(1 for p in _RIGHT_PATTERNS if re.search(p, blob))
    if left_hits and not right_hits:
        return "LEFT"
    if right_hits and not left_hits:
        return "RIGHT"
    return "UNCLEAR"


_COLOR_WORDS = ("white", "yellow", "red", "blue", "green", "orange")
_VALID_OBSERVED_COLORS = {"white", "yellow", "red", "blue"}  # match table vocabulary


def parse_road_lines(evidence_strings: list[str]) -> str | None:
    """DEPRECATED, kept for back-compat with the regex-based prefilter.

    Looks for fragments mentioning road line or center/edge color statements.
    Returns the first matching short phrase, or None.

    The new pipeline uses the dedicated road_evidence_extractor instead and
    reads structured fields off ``state['road_evidence']``.
    """
    for ev in evidence_strings:
        s = str(ev).lower()
        if any(c in s for c in _COLOR_WORDS) and (
            "line" in s or "edge" in s or "center" in s or "centre" in s or "marking" in s
        ):
            return str(ev).strip()
    return None


_PATTERN_RE = re.compile(
    r"outside\s*:\s*(?P<outside>[a-z][a-z ]*?)\s*\|\s*inside\s*:\s*(?P<inside>[a-z][a-z ]*?)\s*$",
    re.IGNORECASE,
)


def parse_road_pattern(pattern_text: str) -> tuple[set[str], set[str]] | None:
    """Parse 'Outside: X | Inside: Y' into (outside_color_set, inside_color_set).

    Color sets allow multi-color entries like "White and yellow" → {white, yellow}.
    Returns None if the pattern cannot be parsed.
    """
    if not pattern_text:
        return None
    m = _PATTERN_RE.search(pattern_text.strip())
    if not m:
        return None

    def _colors(s: str) -> set[str]:
        s = s.lower().replace(" and ", " ").replace(",", " ")
        return {tok for tok in s.split() if tok in _COLOR_WORDS}

    out = _colors(m.group("outside"))
    inside = _colors(m.group("inside"))
    if not out or not inside:
        return None
    return out, inside


class RAGToolbox:
    """Deterministic grounding operations for the rag_ground node."""

    def __init__(self, data_dir: str | Path):
        self._base = Path(data_dir)
        if not self._base.exists():
            raise FileNotFoundError(f"RAG data dir does not exist: {self._base}")
        self._lookup = KeyedLookup(self._base)
        self._driving_left: set[str] = set()
        self._driving_right: set[str] = set()
        self._load_driving_side()

    def _load_driving_side(self) -> None:
        path = self._base / _DRIVING_SIDE_FILENAME
        if not path.exists():
            return
        raw = json.loads(path.read_text())
        self._driving_left = {_normalize(c) for c in raw.get("left", [])}
        self._driving_right = {_normalize(c) for c in raw.get("right", [])}

    @property
    def lookup(self) -> KeyedLookup:
        return self._lookup

    def driving_side_filter(
        self, candidates: list[str], observed: Literal["LEFT", "RIGHT", "UNCLEAR"]
    ) -> DrivingSideResult:
        """Eliminate candidates whose driving side contradicts the observed side.

        Recovery rule (lifted from v10): if the strict filter would eliminate every
        candidate, return them all with a warning instead. Same when observed is
        UNCLEAR or the table is empty (returns all candidates as kept).
        """
        if observed == "UNCLEAR" or not self._driving_left and not self._driving_right:
            payload = "Driving side: UNCLEAR, no eliminations applied."
            return DrivingSideResult(kept=list(candidates), eliminated=[], warning=None, payload=payload)

        kept: list[str] = []
        eliminated: list[tuple[str, str]] = []
        for c in candidates:
            norm = _normalize(c)
            if observed == "LEFT":
                if norm in self._driving_right:
                    eliminated.append((c, "right-hand-traffic country contradicts observed LEFT-hand traffic"))
                else:
                    kept.append(c)
            else:  # RIGHT
                if norm in self._driving_left:
                    eliminated.append((c, "left-hand-traffic country contradicts observed RIGHT-hand traffic"))
                else:
                    kept.append(c)

        if not kept and eliminated:
            warning = (
                f"Driving-side filter would eliminate ALL candidates given observed={observed}. "
                f"Recovery: keeping all candidates; the regulatory agent may have misread driving side."
            )
            payload = f"WARNING: {warning} Eliminated (now kept): {', '.join(c for c, _ in eliminated)}."
            return DrivingSideResult(kept=list(candidates), eliminated=[], warning=warning, payload=payload)

        if not eliminated:
            payload = f"Driving side {observed}: all candidates consistent."
        else:
            elim_str = ", ".join(f"{c} ({reason.split('contradicts')[0].strip()})" for c, reason in eliminated)
            payload = f"Driving side {observed}: eliminated {elim_str}."
        return DrivingSideResult(kept=kept, eliminated=eliminated, warning=None, payload=payload)

    def road_line_check(self, candidates: list[str], observed: str | None) -> RoadCheckResult:
        """DEPRECATED token-overlap check. Kept for back-compat with the regex-based path.

        New pipeline uses ``road_line_check_structured`` with extractor output.
        """
        if not observed:
            by_country = {c: ("UNKNOWN", "(no observation)") for c in candidates}
            return RoadCheckResult(
                by_country=by_country,
                warning=None,
                payload="Road check skipped: no road-line observation in regulatory evidence.",
            )

        obs_tokens = {t for t in re.findall(r"[a-z]+", observed.lower()) if len(t) >= 4}

        by_country: dict[str, tuple[str, str]] = {}
        for c in candidates:
            patterns_text = self._lookup.lookup_road_lines(c)
            if not patterns_text:
                by_country[c] = ("UNKNOWN", "(no table entry)")
                continue
            pat_tokens = {t for t in re.findall(r"[a-z]+", patterns_text.lower()) if len(t) >= 4}
            color_overlap = obs_tokens & pat_tokens & set(_COLOR_WORDS)
            if color_overlap:
                by_country[c] = ("MATCH", patterns_text)
            else:
                by_country[c] = ("MISMATCH", patterns_text)

        countries_with_data = [c for c, (v, _) in by_country.items() if v != "UNKNOWN"]
        all_mismatch = countries_with_data and all(by_country[c][0] == "MISMATCH" for c in countries_with_data)

        warning = None
        if all_mismatch:
            warning = (
                "Road-line check returns MISMATCH for all candidates with table data. "
                "Recovery: marking these UNKNOWN; the regulatory agent's color description may be wrong."
            )
            for c in countries_with_data:
                _, patterns = by_country[c]
                by_country[c] = ("UNKNOWN", patterns + " [recovery: was MISMATCH]")

        # Render payload
        lines = [f"Road observation: {observed}"]
        for c, (verdict, pattern) in by_country.items():
            lines.append(f"  {c}: {verdict}, data: {pattern}")
        if warning:
            lines.append(f"  WARNING: {warning}")
        payload = "\n".join(lines)

        return RoadCheckResult(by_country=by_country, warning=warning, payload=payload)

    def fetch_references(
        self,
        countries: list[str],
        categories: list[str] | None = None,
        max_per_country: int = 500,
        max_total: int = 500,
        skip: set[tuple[str, str]] | None = None,
        bollard_materials: list[str] | None = None,
        bollard_colors: list[str] | None = None,
    ) -> list[Reference]:
        """Fetch reference images for the given countries.

        Args:
            countries: candidate country names (any common form; KeyedLookup resolves aliases).
            categories: category names (defaults to CATEGORY_PRIORITY). road_lines is text-only,
                handled separately by road_line_check, so it's filtered out here.
            max_per_country: cap images returned per country.
            max_total: hard cap across all countries.
            skip: set of (country_normalized, category) pairs already shown in prior rounds.
            bollard_materials: if set, only bollard refs matching these materials are returned
                (sorted by match score descending).
            bollard_colors: if set, only bollard refs matching these colors are returned.
        """
        if categories is None:
            categories = list(CATEGORY_PRIORITY)
        # road_lines yields text via lookup_road_lines, not images; exclude here
        categories = [c for c in categories if c != "road_lines"]
        skip = skip or set()

        out: list[Reference] = []
        for country in countries:
            per_country: list[Reference] = []
            refs, _road_text = self._lookup.fetch_references(
                country,
                categories,
                bollard_materials=bollard_materials,
                bollard_colors=bollard_colors,
                max_total=max_per_country * 4,
            )
            for ref in refs:
                key = (_normalize(country), ref.category)
                if key in skip:
                    continue
                per_country.append(ref)
                if len(per_country) >= max_per_country:
                    break
            out.extend(per_country)
            if len(out) >= max_total:
                break
        return out[:max_total]

    def road_line_check_structured(
        self,
        candidates: list[str],
        outside_observed: str,
        inside_observed: str,
    ) -> RoadCheckResult:
        """Positional match: observed (outside, inside) vs table (Outside, Inside) per country.

        Verdicts:
          MATCH, at least one pattern variant for the country matches BOTH axes.
          MISMATCH, country has parseable patterns but none match both axes.
          UNKNOWN, no table entry, or unparseable patterns, or observed=unclear/none on either axis.

        Recovery rule: if every country with table data is MISMATCH, mark all UNKNOWN
        with a warning (the extractor's color reading is probably wrong).

        Multi-color entries like "White and yellow" match if observed is in the set.
        """
        # If either observed axis is non-actionable, skip, no eliminations.
        if outside_observed not in _VALID_OBSERVED_COLORS or inside_observed not in _VALID_OBSERVED_COLORS:
            by_country = {c: ("UNKNOWN", "(observation unclear)") for c in candidates}
            payload = (
                f"Road check skipped: observed outside='{outside_observed}', "
                f"inside='{inside_observed}' (need both to be a definite color)."
            )
            return RoadCheckResult(by_country=by_country, warning=None, payload=payload)

        by_country: dict[str, tuple[str, str]] = {}
        for c in candidates:
            patterns = self._lookup.road_line_patterns(c)
            if not patterns:
                by_country[c] = ("UNKNOWN", "(no table entry)")
                continue
            parsed_variants = [pv for pv in (parse_road_pattern(p) for p in patterns) if pv]
            if not parsed_variants:
                by_country[c] = ("UNKNOWN", f"(unparseable: {patterns})")
                continue

            matched = any(
                outside_observed in out_set and inside_observed in in_set
                for out_set, in_set in parsed_variants
            )
            patterns_text = " | ".join(patterns)
            if matched:
                by_country[c] = ("MATCH", patterns_text)
            else:
                by_country[c] = ("MISMATCH", patterns_text)

        countries_with_data = [c for c, (v, _) in by_country.items() if v != "UNKNOWN"]
        all_mismatch = bool(countries_with_data) and all(
            by_country[c][0] == "MISMATCH" for c in countries_with_data
        )

        warning = None
        if all_mismatch:
            warning = (
                f"Road-line check returns MISMATCH for ALL candidates given observed "
                f"outside={outside_observed}, inside={inside_observed}. "
                f"Recovery: marking these UNKNOWN; the road_evidence_extractor reading may be wrong."
            )
            for c in countries_with_data:
                _, patterns_text = by_country[c]
                by_country[c] = ("UNKNOWN", patterns_text + " [recovery: was MISMATCH]")

        lines = [
            f"Road observation (structured): outside={outside_observed}, inside={inside_observed}"
        ]
        for c, (verdict, patterns_text) in by_country.items():
            lines.append(f"  {c}: {verdict}, data: {patterns_text}")
        if warning:
            lines.append(f"  WARNING: {warning}")
        payload = "\n".join(lines)

        return RoadCheckResult(by_country=by_country, warning=warning, payload=payload)

    def render_refs_summary(self, refs: list[Reference]) -> str:
        """One-line per (country, category) summary for prompt text."""
        if not refs:
            return "No reference images available."
        buckets: dict[tuple[str, str], int] = {}
        for r in refs:
            buckets[(r.country, r.category)] = buckets.get((r.country, r.category), 0) + 1
        parts = [f"{country}/{category}: {n} img{'s' if n > 1 else ''}" for (country, category), n in buckets.items()]
        return " · ".join(parts)
