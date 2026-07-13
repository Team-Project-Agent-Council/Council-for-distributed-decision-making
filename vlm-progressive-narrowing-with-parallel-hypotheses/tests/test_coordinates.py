"""Regression tests for `vlm_council.coordinates.parse_coordinates`.

Guards against silent regressions in the free-text coordinate parser that
turns the judge's `Country: X\\nCoordinates: <lat>, <lon>\\nReasoning: ...`
output into a structured `{"lat": float, "lng": float}` dict.

These tests also lock in the design decision that a judge failure must NOT
produce a `(0, 0)` fake coordinate: the caller has to actively pass the
placeholder text for the parser to return `(0, 0)`; a hard judge failure
now emits `coordinates=None` and an `error` field.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the project root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_council.coordinates import parse_coordinates  # noqa: E402


class ParseCoordinatesHappyPath(unittest.TestCase):
    """Well-formed judge outputs must round-trip into a valid dict."""

    def test_standard_output(self) -> None:
        text = (
            "Country: Uzbekistan\n"
            "Coordinates: 41.2, 71.5\n"
            "Reasoning: Fergana valley evidence."
        )
        self.assertEqual(parse_coordinates(text), {"lat": 41.2, "lng": 71.5})

    def test_negative_coordinates(self) -> None:
        text = "Coordinates: -34.6037, -58.3816"
        self.assertEqual(parse_coordinates(text), {"lat": -34.6037, "lng": -58.3816})

    def test_extreme_valid_ranges(self) -> None:
        for text, expected in [
            ("Coordinates: 90, 180", {"lat": 90.0, "lng": 180.0}),
            ("Coordinates: -90, -180", {"lat": -90.0, "lng": -180.0}),
            ("Coordinates: 0, 0", {"lat": 0.0, "lng": 0.0}),
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_coordinates(text), expected)


class ParseCoordinatesUnparseable(unittest.TestCase):
    """Malformed inputs must return None, never a partial or fabricated dict."""

    def test_empty_string(self) -> None:
        self.assertIsNone(parse_coordinates(""))

    def test_no_coordinates_line(self) -> None:
        self.assertIsNone(parse_coordinates("Country: Uzbekistan\nReasoning: text."))

    def test_non_numeric_values(self) -> None:
        self.assertIsNone(parse_coordinates("Coordinates: abc, def"))

    def test_out_of_range_lat(self) -> None:
        self.assertIsNone(parse_coordinates("Coordinates: 999, 0"))

    def test_out_of_range_lng(self) -> None:
        self.assertIsNone(parse_coordinates("Coordinates: 0, -500"))


class ParseCoordinatesEdgeCases(unittest.TestCase):
    """Weird-but-legal shapes the judge might emit."""

    def test_extra_whitespace_between_values(self) -> None:
        text = "Coordinates:   41.2 ,   71.5"
        self.assertEqual(parse_coordinates(text), {"lat": 41.2, "lng": 71.5})

    def test_only_reads_first_coordinates_line(self) -> None:
        # If the judge for some reason emits two Coordinates lines,
        # the FIRST one wins.  Locking in the current behaviour so any
        # future change is deliberate.
        text = (
            "Coordinates: 41.2, 71.5\n"
            "Coordinates: 99.9, 99.9\n"
        )
        self.assertEqual(parse_coordinates(text), {"lat": 41.2, "lng": 71.5})


if __name__ == "__main__":
    unittest.main()
