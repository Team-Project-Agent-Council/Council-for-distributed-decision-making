"""Standalone evaluator for the 500-image tournament run.

Computes:
- Country accuracy (exact ISO-2 match)
- Neighbouring-country accuracy (prediction shares a land border with GT)
- Haversine distance stats (mean, median, p90) between predicted and GT coords

Usage:
    python scripts/eval_500.py \
        --results results_tournament_500 \
        --gt georc_locations_500.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


# lowercase country name -> ISO-2
ISO = {
    "united states": "US", "usa": "US", "united states of america": "US", "america": "US",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB",
    "england": "GB", "scotland": "GB", "wales": "GB",
    "germany": "DE", "france": "FR", "italy": "IT", "spain": "ES", "portugal": "PT",
    "netherlands": "NL", "holland": "NL", "belgium": "BE", "switzerland": "CH",
    "austria": "AT", "poland": "PL", "czech republic": "CZ", "czechia": "CZ",
    "slovakia": "SK", "hungary": "HU", "romania": "RO", "bulgaria": "BG",
    "greece": "GR", "turkey": "TR", "turkiye": "TR",
    "russia": "RU", "russian federation": "RU", "ukraine": "UA", "belarus": "BY",
    "lithuania": "LT", "latvia": "LV", "estonia": "EE",
    "finland": "FI", "sweden": "SE", "norway": "NO", "denmark": "DK",
    "iceland": "IS", "ireland": "IE",
    "japan": "JP", "china": "CN", "south korea": "KR", "republic of korea": "KR",
    "north korea": "KP", "taiwan": "TW", "hong kong": "HK",
    "vietnam": "VN", "viet nam": "VN", "thailand": "TH", "cambodia": "KH", "laos": "LA",
    "malaysia": "MY", "singapore": "SG", "indonesia": "ID", "philippines": "PH",
    "myanmar": "MM", "burma": "MM", "brunei": "BN",
    "india": "IN", "pakistan": "PK", "bangladesh": "BD", "sri lanka": "LK",
    "nepal": "NP", "bhutan": "BT", "afghanistan": "AF",
    "kazakhstan": "KZ", "uzbekistan": "UZ", "kyrgyzstan": "KG", "tajikistan": "TJ",
    "turkmenistan": "TM", "mongolia": "MN",
    "iran": "IR", "iraq": "IQ", "saudi arabia": "SA", "uae": "AE",
    "united arab emirates": "AE", "qatar": "QA", "bahrain": "BH", "kuwait": "KW",
    "oman": "OM", "yemen": "YE", "israel": "IL", "palestine": "PS",
    "jordan": "JO", "lebanon": "LB", "syria": "SY",
    "egypt": "EG", "libya": "LY", "tunisia": "TN", "algeria": "DZ", "morocco": "MA",
    "western sahara": "EH", "sudan": "SD", "south sudan": "SS",
    "ethiopia": "ET", "eritrea": "ER", "djibouti": "DJ", "somalia": "SO",
    "kenya": "KE", "tanzania": "TZ", "uganda": "UG", "rwanda": "RW", "burundi": "BI",
    "democratic republic of the congo": "CD", "dr congo": "CD", "drc": "CD",
    "congo": "CG", "republic of the congo": "CG", "gabon": "GA", "cameroon": "CM",
    "central african republic": "CF", "chad": "TD", "niger": "NE", "nigeria": "NG",
    "benin": "BJ", "togo": "TG", "ghana": "GH",
    "ivory coast": "CI", "cote d ivoire": "CI",
    "cote d’ivoire": "CI", "côte d’ivoire": "CI", "côte d ivoire": "CI",
    "liberia": "LR", "sierra leone": "SL", "guinea": "GN", "guinea bissau": "GW",
    "senegal": "SN", "gambia": "GM", "mali": "ML", "mauritania": "MR",
    "burkina faso": "BF",
    "angola": "AO", "zambia": "ZM", "zimbabwe": "ZW", "botswana": "BW", "namibia": "NA",
    "south africa": "ZA", "lesotho": "LS", "swaziland": "SZ", "eswatini": "SZ",
    "mozambique": "MZ", "malawi": "MW", "madagascar": "MG", "mauritius": "MU",
    "seychelles": "SC", "comoros": "KM", "cape verde": "CV",
    "canada": "CA", "mexico": "MX", "guatemala": "GT", "belize": "BZ",
    "el salvador": "SV", "honduras": "HN", "nicaragua": "NI", "costa rica": "CR",
    "panama": "PA", "cuba": "CU", "haiti": "HT", "dominican republic": "DO",
    "jamaica": "JM", "trinidad and tobago": "TT", "barbados": "BB", "bahamas": "BS",
    "puerto rico": "PR", "aruba": "AW",
    "brazil": "BR", "argentina": "AR", "chile": "CL", "peru": "PE", "colombia": "CO",
    "venezuela": "VE", "ecuador": "EC", "bolivia": "BO", "paraguay": "PY",
    "uruguay": "UY", "guyana": "GY", "suriname": "SR", "french guiana": "GF",
    "australia": "AU", "new zealand": "NZ", "papua new guinea": "PG", "fiji": "FJ",
    "samoa": "WS", "tonga": "TO", "vanuatu": "VU", "solomon islands": "SB",
    "andorra": "AD", "monaco": "MC", "luxembourg": "LU", "liechtenstein": "LI",
    "san marino": "SM", "malta": "MT", "cyprus": "CY", "slovenia": "SI",
    "croatia": "HR", "serbia": "RS", "montenegro": "ME", "north macedonia": "MK",
    "macedonia": "MK", "albania": "AL", "bosnia and herzegovina": "BA",
    "kosovo": "XK", "moldova": "MD", "georgia": "GE", "armenia": "AM",
    "azerbaijan": "AZ",
    "hawaii": "US", "hawaii, usa": "US", "reunion": "RE", "réunion": "RE",
}


# ISO-2 -> set of ISO-2 land neighbours (Kosovo XK included).
# Compiled from CIA World Factbook / Wikipedia. Kept manually so we don't
# need geopandas at runtime.
NEIGHBOURS: dict[str, set[str]] = {
    "AF": {"IR", "PK", "TM", "UZ", "TJ", "CN"},
    "AL": {"ME", "XK", "MK", "GR"},
    "AM": {"GE", "AZ", "IR", "TR"},
    "AO": {"CD", "ZM", "NA", "CG"},
    "AR": {"CL", "BO", "PY", "BR", "UY"},
    "AT": {"DE", "CZ", "SK", "HU", "SI", "IT", "CH", "LI"},
    "AZ": {"AM", "GE", "IR", "RU", "TR"},
    "BA": {"HR", "RS", "ME"},
    "BD": {"IN", "MM"},
    "BE": {"NL", "DE", "LU", "FR"},
    "BF": {"ML", "NE", "BJ", "TG", "GH", "CI"},
    "BG": {"RO", "RS", "MK", "GR", "TR"},
    "BI": {"RW", "TZ", "CD"},
    "BJ": {"TG", "BF", "NE", "NG"},
    "BN": {"MY"},
    "BO": {"BR", "PY", "AR", "CL", "PE"},
    "BR": {"UY", "AR", "PY", "BO", "PE", "CO", "VE", "GY", "SR", "GF"},
    "BT": {"CN", "IN"},
    "BW": {"NA", "ZA", "ZM", "ZW"},
    "BY": {"LT", "LV", "PL", "RU", "UA"},
    "BZ": {"MX", "GT"},
    "CA": {"US"},
    "CD": {"CG", "CF", "SS", "UG", "RW", "BI", "TZ", "ZM", "AO"},
    "CF": {"CM", "TD", "SD", "SS", "CD", "CG"},
    "CG": {"GA", "CM", "CF", "CD", "AO"},
    "CH": {"DE", "FR", "IT", "AT", "LI"},
    "CI": {"LR", "GN", "ML", "BF", "GH"},
    "CM": {"NG", "TD", "CF", "CG", "GA", "GQ"},
    "CN": {"MN", "RU", "KP", "VN", "LA", "MM", "IN", "BT", "NP", "PK", "AF", "TJ", "KG", "KZ"},
    "CO": {"PA", "VE", "BR", "PE", "EC"},
    "CR": {"NI", "PA"},
    "CY": set(),
    "CZ": {"DE", "PL", "SK", "AT"},
    "DE": {"DK", "PL", "CZ", "AT", "CH", "FR", "LU", "BE", "NL"},
    "DJ": {"ER", "ET", "SO"},
    "DK": {"DE"},
    "DO": {"HT"},
    "DZ": {"TN", "LY", "NE", "ML", "MR", "EH", "MA"},
    "EC": {"CO", "PE"},
    "EE": {"LV", "RU"},
    "EG": {"LY", "SD", "IL", "PS"},
    "EH": {"MA", "DZ", "MR"},
    "ER": {"SD", "ET", "DJ"},
    "ES": {"PT", "FR", "AD", "MA"},
    "ET": {"ER", "DJ", "SO", "KE", "SS", "SD"},
    "FI": {"NO", "SE", "RU"},
    "FR": {"BE", "LU", "DE", "CH", "IT", "MC", "ES", "AD"},
    "GA": {"CG", "CM", "GQ"},
    "GB": {"IE"},
    "GE": {"RU", "TR", "AM", "AZ"},
    "GF": {"BR", "SR"},
    "GH": {"CI", "BF", "TG"},
    "GM": {"SN"},
    "GN": {"GW", "SN", "ML", "CI", "LR", "SL"},
    "GQ": {"GA", "CM"},
    "GR": {"AL", "MK", "BG", "TR"},
    "GT": {"MX", "BZ", "SV", "HN"},
    "GW": {"SN", "GN"},
    "GY": {"VE", "BR", "SR"},
    "HN": {"GT", "SV", "NI"},
    "HR": {"SI", "HU", "RS", "BA", "ME"},
    "HT": {"DO"},
    "HU": {"AT", "SK", "UA", "RO", "RS", "HR", "SI"},
    "ID": {"MY", "PG", "TL"},
    "IE": {"GB"},
    "IL": {"LB", "SY", "JO", "EG", "PS"},
    "IN": {"PK", "CN", "NP", "BT", "BD", "MM"},
    "IQ": {"TR", "IR", "KW", "SA", "JO", "SY"},
    "IR": {"AM", "AZ", "TM", "AF", "PK", "TR", "IQ"},
    "IS": set(),
    "IT": {"FR", "CH", "AT", "SI", "SM", "VA"},
    "JM": set(),
    "JO": {"SY", "IQ", "SA", "IL", "PS"},
    "JP": set(),
    "KE": {"SS", "ET", "SO", "UG", "TZ"},
    "KG": {"KZ", "CN", "TJ", "UZ"},
    "KH": {"TH", "LA", "VN"},
    "KM": set(),
    "KP": {"CN", "RU", "KR"},
    "KR": {"KP"},
    "KW": {"IQ", "SA"},
    "KZ": {"RU", "CN", "KG", "UZ", "TM"},
    "LA": {"MM", "CN", "VN", "KH", "TH"},
    "LB": {"SY", "IL"},
    "LI": {"AT", "CH"},
    "LK": set(),
    "LR": {"SL", "GN", "CI"},
    "LS": {"ZA"},
    "LT": {"LV", "BY", "PL", "RU"},
    "LU": {"BE", "DE", "FR"},
    "LV": {"EE", "RU", "BY", "LT"},
    "LY": {"TN", "DZ", "NE", "TD", "SD", "EG"},
    "MA": {"DZ", "EH", "ES"},
    "MC": {"FR"},
    "MD": {"RO", "UA"},
    "ME": {"HR", "BA", "RS", "XK", "AL"},
    "MG": set(),
    "MK": {"XK", "RS", "BG", "GR", "AL"},
    "ML": {"DZ", "NE", "BF", "CI", "GN", "SN", "MR"},
    "MM": {"BD", "IN", "CN", "LA", "TH"},
    "MN": {"RU", "CN"},
    "MR": {"EH", "DZ", "ML", "SN"},
    "MT": set(),
    "MU": set(),
    "MW": {"TZ", "MZ", "ZM"},
    "MX": {"US", "GT", "BZ"},
    "MY": {"TH", "ID", "BN"},
    "MZ": {"TZ", "MW", "ZM", "ZW", "ZA", "SZ"},
    "NA": {"AO", "ZM", "ZW", "BW", "ZA"},
    "NE": {"LY", "TD", "NG", "BJ", "BF", "ML", "DZ"},
    "NG": {"BJ", "NE", "TD", "CM"},
    "NI": {"HN", "CR"},
    "NL": {"BE", "DE"},
    "NO": {"SE", "FI", "RU"},
    "NP": {"CN", "IN"},
    "NZ": set(),
    "OM": {"SA", "YE", "AE"},
    "PA": {"CR", "CO"},
    "PE": {"EC", "CO", "BR", "BO", "CL"},
    "PG": {"ID"},
    "PH": set(),
    "PK": {"IR", "AF", "CN", "IN"},
    "PL": {"DE", "CZ", "SK", "UA", "BY", "LT", "RU"},
    "PS": {"IL", "EG", "JO"},
    "PR": set(),
    "PT": {"ES"},
    "PY": {"BO", "BR", "AR"},
    "QA": {"SA"},
    "RO": {"UA", "MD", "BG", "RS", "HU"},
    "RS": {"HU", "RO", "BG", "MK", "XK", "ME", "BA", "HR"},
    "RU": {"NO", "FI", "EE", "LV", "LT", "PL", "BY", "UA", "GE", "AZ", "KZ", "CN", "MN", "KP"},
    "RW": {"UG", "TZ", "BI", "CD"},
    "SA": {"JO", "IQ", "KW", "QA", "AE", "OM", "YE"},
    "SB": set(),
    "SC": set(),
    "SD": {"EG", "LY", "TD", "CF", "SS", "ET", "ER"},
    "SE": {"NO", "FI"},
    "SG": set(),
    "SI": {"IT", "AT", "HU", "HR"},
    "SK": {"CZ", "PL", "UA", "HU", "AT"},
    "SL": {"GN", "LR"},
    "SM": {"IT"},
    "SN": {"MR", "ML", "GN", "GW", "GM"},
    "SO": {"DJ", "ET", "KE"},
    "SR": {"GY", "BR", "GF"},
    "SS": {"SD", "ET", "KE", "UG", "CD", "CF"},
    "SV": {"GT", "HN"},
    "SY": {"TR", "IQ", "JO", "IL", "LB"},
    "SZ": {"ZA", "MZ"},
    "TD": {"LY", "SD", "CF", "CM", "NG", "NE"},
    "TG": {"GH", "BF", "BJ"},
    "TH": {"MM", "LA", "KH", "MY"},
    "TJ": {"AF", "UZ", "KG", "CN"},
    "TL": {"ID"},
    "TM": {"KZ", "UZ", "AF", "IR"},
    "TN": {"DZ", "LY"},
    "TO": set(),
    "TR": {"GR", "BG", "GE", "AM", "AZ", "IR", "IQ", "SY"},
    "TT": set(),
    "TW": set(),
    "TZ": {"KE", "UG", "RW", "BI", "CD", "ZM", "MW", "MZ"},
    "UA": {"BY", "RU", "MD", "RO", "HU", "SK", "PL"},
    "UG": {"SS", "KE", "TZ", "RW", "CD"},
    "US": {"CA", "MX"},
    "UY": {"BR", "AR"},
    "UZ": {"KZ", "KG", "TJ", "AF", "TM"},
    "VA": {"IT"},
    "VE": {"CO", "BR", "GY"},
    "VN": {"CN", "LA", "KH"},
    "VU": set(),
    "WS": set(),
    "XK": {"AL", "MK", "RS", "ME"},
    "YE": {"SA", "OM"},
    "ZA": {"NA", "BW", "ZW", "MZ", "SZ", "LS"},
    "ZM": {"CD", "TZ", "MW", "MZ", "ZW", "BW", "NA", "AO"},
    "ZW": {"ZM", "MZ", "BW", "ZA", "NA"},
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parse_coords(country_result: str) -> tuple[float, float] | None:
    m = re.search(r"Coordinates:\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)", country_result)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None


def parse_country(country_result: str) -> str | None:
    m = re.search(r"Country:\s*([^\n]+)", country_result)
    if not m:
        return None
    return m.group(1).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    args = ap.parse_args()

    gt: dict[str, dict] = {}
    with open(args.gt) as f:
        for row in csv.DictReader(f):
            stem = row["filename"].rsplit(".", 1)[0]
            gt[stem] = {
                "iso": row["country_code"].upper(),
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
            }

    total = 0
    correct = 0
    neighbour = 0
    unmapped: set[str] = set()
    distances: list[float] = []
    missing_coords = 0

    for d in args.results.iterdir():
        if not d.is_dir():
            continue
        stem = d.name
        if stem not in gt:
            continue
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        if data.get("error"):
            continue

        total += 1
        cr = data.get("country_result", "") or ""

        pred_name = parse_country(cr)
        pred_iso = ISO.get(pred_name) if pred_name else None
        if pred_name and pred_iso is None:
            unmapped.add(pred_name)

        gt_iso = gt[stem]["iso"]
        if pred_iso == gt_iso:
            correct += 1
        elif pred_iso and gt_iso in NEIGHBOURS.get(pred_iso, set()):
            neighbour += 1

        coords = parse_coords(cr)
        if coords is None:
            missing_coords += 1
        else:
            distances.append(haversine_km(coords[0], coords[1], gt[stem]["lat"], gt[stem]["lng"]))

    correct_or_neighbour = correct + neighbour

    def _pct(n: int) -> str:
        return f"{100 * n / total:.2f}%" if total else "n/a"

    print(f"Evaluated: {total}")
    print()
    print("=== Country accuracy ===")
    print(f"  Exact match:            {correct:4d}  ({_pct(correct)})")
    print(f"  Neighbouring country:   {neighbour:4d}  ({_pct(neighbour)})")
    print(f"  Correct or neighbour:   {correct_or_neighbour:4d}  ({_pct(correct_or_neighbour)})")
    print()
    print("=== Haversine distance (predicted vs GT coordinates) ===")
    if distances:
        distances.sort()
        mean_km = statistics.mean(distances)
        median_km = statistics.median(distances)
        p90 = distances[int(0.9 * (len(distances) - 1))]
        p95 = distances[int(0.95 * (len(distances) - 1))]
        print(f"  n:                {len(distances)}")
        print(f"  Mean:             {mean_km:8.1f} km")
        print(f"  Median:           {median_km:8.1f} km")
        print(f"  p90:              {p90:8.1f} km")
        print(f"  p95:              {p95:8.1f} km")
        print(f"  Min / Max:        {distances[0]:.1f} / {distances[-1]:.1f} km")
        print(f"  < 100 km:         {sum(1 for d in distances if d < 100):4d}  "
              f"({100 * sum(1 for d in distances if d < 100) / len(distances):.1f}%)")
        print(f"  < 500 km:         {sum(1 for d in distances if d < 500):4d}  "
              f"({100 * sum(1 for d in distances if d < 500) / len(distances):.1f}%)")
        print(f"  < 1000 km:        {sum(1 for d in distances if d < 1000):4d}  "
              f"({100 * sum(1 for d in distances if d < 1000) / len(distances):.1f}%)")
    if missing_coords:
        print(f"  (missing coordinates in {missing_coords} predictions)")

    if unmapped:
        print(f"\nUnmapped country names: {sorted(unmapped)}")


if __name__ == "__main__":
    main()
