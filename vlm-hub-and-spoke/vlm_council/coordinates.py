"""Shared coordinate parsing.

The judge produces `country_result` as free text containing a line of the form:
    Coordinates: <lat>, <lng>

This module centralises the extraction so that `batch.py`, `run.py`, and
`evaluate.py` never drift on the regex.

Returned shape matches the demo fixture format used elsewhere in the project:
    {"lat": <float>, "lng": <float>}

An unparseable or out-of-range value returns None; callers decide whether to
substitute an empty string or preserve `None` in their output.
"""

from __future__ import annotations

import re
from typing import Optional

_COORD_RE = re.compile(r"Coordinates:\s*([-\d.]+)\s*,\s*([-\d.]+)")


def parse_coordinates(country_result: str) -> Optional[dict[str, float]]:
    """Extract {'lat': ..., 'lng': ...} from a `country_result` string.

    Returns None if the pattern is missing, unparseable, or the values fall
    outside valid geographic ranges.
    """
    if not country_result:
        return None
    match = _COORD_RE.search(country_result)
    if not match:
        return None
    try:
        lat = float(match.group(1))
        lng = float(match.group(2))
    except ValueError:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    return {"lat": lat, "lng": lng}
