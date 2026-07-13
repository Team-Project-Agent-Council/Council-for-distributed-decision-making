"""Regression tests for the country-matching helpers in
`vlm_council.evaluate`.

`_countries_match` is the workhorse that decides whether a run counts as a
correct prediction. It uses a chain of normalisation steps + an alias table
+ ISO-code lookups, which is exactly the kind of code that breaks silently
when someone adds a new alias or changes the CSV format. These tests lock
down the current behaviour on realistic model outputs.

Test data uses ISO alpha-2 codes as ground truth (matching how
`_load_ground_truth` reads `georc_locations.csv`).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_council.evaluate import (  # noqa: E402
    _countries_match,
    _extract_country,
    _fold_diacritics,
    _is_neighbor,
    _normalize_country,
)


class CountriesMatchDirect(unittest.TestCase):
    """Direct name/code matches must always succeed."""

    def test_exact_lowercase_name(self) -> None:
        self.assertTrue(_countries_match("germany", "de"))

    def test_exact_capitalised_name(self) -> None:
        self.assertTrue(_countries_match("Germany", "de"))

    def test_iso_code_as_prediction(self) -> None:
        self.assertTrue(_countries_match("DE", "de"))

    def test_wrong_country(self) -> None:
        self.assertFalse(_countries_match("France", "de"))


class CountriesMatchAliases(unittest.TestCase):
    """Common LLM output variants must resolve to their canonical country."""

    def test_usa_variants(self) -> None:
        for prediction in ["USA", "U.S.", "U.S.A.", "United States"]:
            with self.subTest(prediction=prediction):
                self.assertTrue(_countries_match(prediction, "us"))

    def test_uk_variants(self) -> None:
        for prediction in ["UK", "United Kingdom", "Britain", "England"]:
            with self.subTest(prediction=prediction):
                self.assertTrue(_countries_match(prediction, "gb"))

    def test_korea_disambiguation(self) -> None:
        # "Korea" defaults to South Korea per the alias table
        self.assertTrue(_countries_match("Korea", "kr"))
        self.assertTrue(_countries_match("Republic of Korea", "kr"))
        self.assertTrue(_countries_match("DPRK", "kp"))
        # And they should NOT cross-match
        self.assertFalse(_countries_match("DPRK", "kr"))
        self.assertFalse(_countries_match("South Korea", "kp"))

    def test_ivory_coast(self) -> None:
        self.assertTrue(_countries_match("Cote d'Ivoire", "ci"))
        self.assertTrue(_countries_match("Ivory Coast", "ci"))

    def test_netherlands(self) -> None:
        self.assertTrue(_countries_match("The Netherlands", "nl"))
        self.assertTrue(_countries_match("Holland", "nl"))


class CountriesMatchOfficialLongForms(unittest.TestCase):
    """Ground-truth CSVs sometimes use long/official forms; matching must
    still work for both directions (prediction vs ground truth)."""

    def test_russian_federation(self) -> None:
        self.assertTrue(_countries_match("Russia", "ru"))
        # Long form as prediction
        self.assertTrue(_countries_match("Russian Federation", "ru"))

    def test_turkiye_with_diacritic(self) -> None:
        self.assertTrue(_countries_match("Türkiye", "tr"))
        self.assertTrue(_countries_match("Turkiye", "tr"))
        self.assertTrue(_countries_match("Turkey", "tr"))

    def test_curacao_with_diacritic(self) -> None:
        self.assertTrue(_countries_match("Curaçao", "cw"))
        self.assertTrue(_countries_match("Curacao", "cw"))

    def test_czechia_and_czech_republic(self) -> None:
        self.assertTrue(_countries_match("Czech Republic", "cz"))
        self.assertTrue(_countries_match("Czechia", "cz"))


class NormalizeCountry(unittest.TestCase):
    """Normalisation strips punctuation and lower-cases before alias lookup."""

    def test_trailing_period_stripped(self) -> None:
        self.assertEqual(_normalize_country("Germany."), "germany")

    def test_whitespace_and_case(self) -> None:
        self.assertEqual(_normalize_country("  GERMANY  "), "germany")


class ExtractCountry(unittest.TestCase):
    """`_extract_country` pulls the country name from the judge's free text."""

    def test_standard_format(self) -> None:
        text = "Country: Portugal\nCoordinates: 39.5, -8.0\nReasoning: ..."
        self.assertEqual(_extract_country(text), "Portugal")

    def test_no_prefix_falls_back_to_first_line(self) -> None:
        self.assertEqual(_extract_country("Portugal"), "Portugal")

    def test_empty(self) -> None:
        self.assertEqual(_extract_country(""), "")


class IsNeighbor(unittest.TestCase):
    """Neighbour lookup is used to score near-misses as partial credit.

    We rely on the packaged GEODATASOURCE CSV, which encodes real-world
    borders. Any change to that CSV or to the lookup semantics should be
    a conscious decision, so we lock down a handful of well-known pairs.
    """

    def test_direct_neighbours(self) -> None:
        # France borders Germany and Spain, not Portugal or Japan
        self.assertTrue(_is_neighbor("France", "de"))
        self.assertTrue(_is_neighbor("Spain", "fr"))

    def test_non_neighbours(self) -> None:
        self.assertFalse(_is_neighbor("Japan", "de"))
        self.assertFalse(_is_neighbor("Uruguay", "de"))

    def test_unknown_country_is_not_a_neighbor(self) -> None:
        self.assertFalse(_is_neighbor("Atlantis", "de"))


class FoldDiacritics(unittest.TestCase):
    """Diacritic + typographic-quote folding used by both name pipelines."""

    def test_umlaut(self) -> None:
        self.assertEqual(_fold_diacritics("Türkiye"), "Turkiye")

    def test_cedilla(self) -> None:
        self.assertEqual(_fold_diacritics("Curaçao"), "Curacao")

    def test_typographic_apostrophe(self) -> None:
        self.assertEqual(_fold_diacritics("cote d’ivoire"), "cote d'ivoire")

    def test_ascii_passthrough(self) -> None:
        self.assertEqual(_fold_diacritics("Germany"), "Germany")


if __name__ == "__main__":
    unittest.main()
