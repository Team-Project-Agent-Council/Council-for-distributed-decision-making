"""Evaluate VLM Council results against ground truth.

Compares predicted country and coordinates against georc_locations.csv.
Outputs: country accuracy, neighbor accuracy, distance statistics, per-agent confidence, and per-image details.

Usage:
    python -m vlm_council.evaluate results/ Images/georc_locations.csv
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


def _load_country_codes_from_csv() -> dict[str, str]:
    """Load country_code -> country_name mapping from GEODATASOURCE-COUNTRY-BORDERS.CSV."""
    borders_path = Path(__file__).parent / "GEODATASOURCE-COUNTRY-BORDERS.CSV"
    codes: dict[str, str] = {}
    if not borders_path.exists():
        return codes
    with open(borders_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["country_code"].strip().lower()
            name = row["country_name"].strip().lower()
            if code and name:
                codes[code] = name
            # Also add border countries
            border_code = row["country_border_code"].strip().lower()
            border_name = row["country_border_name"].strip().lower()
            if border_code and border_name:
                codes[border_code] = border_name
    return codes


# Shorten official CSV names to common short forms used by models and ground truth
_CSV_NAME_OVERRIDES = {
    # Full official names (no parentheses)
    "united states of america": "united states",
    "united kingdom of great britain and northern ireland": "united kingdom",
    "russian federation": "russia",
    "syrian arab republic": "syria",
    "viet nam": "vietnam",
    "brunei darussalam": "brunei",
    "timor-leste": "east timor",
    "macao": "macau",
    "north macedonia": "north macedonia",
    "republic of north macedonia": "north macedonia",
    "republic of the congo": "republic of the congo",
    "côte d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    # Comma format (some CSVs use this)
    "korea, republic of": "south korea",
    "korea, democratic people's republic of": "north korea",
    "iran, islamic republic of": "iran",
    "lao people's democratic republic": "laos",
    "tanzania, united republic of": "tanzania",
    "venezuela, bolivarian republic of": "venezuela",
    "bolivia, plurinational state of": "bolivia",
    "congo, democratic republic of the": "congo",
    "congo, the democratic republic of the": "congo",
    "taiwan, province of china": "taiwan",
    "palestine, state of": "palestine",
    "moldova, republic of": "moldova",
    # Parentheses format (GEODATASOURCE CSV uses this)
    "bolivia (plurinational state of)": "bolivia",
    "congo (the democratic republic of the)": "congo",
    "iran (islamic republic of)": "iran",
    "korea (democratic people's republic of)": "north korea",
    "korea (the republic of)": "south korea",
    "lao people's democratic republic": "laos",
    "micronesia (federated states of)": "micronesia",
    "moldova (the republic of)": "moldova",
    "palestine, state of": "palestine",
    "taiwan (province of china)": "taiwan",
    "tanzania (the united republic of)": "tanzania",
    "venezuela (bolivarian republic of)": "venezuela",
    "gambia (the)": "gambia",
}


def _normalize_csv_name(name: str) -> str:
    """Normalize official CSV country names to common short forms."""
    return _CSV_NAME_OVERRIDES.get(name, name)


# Build country code mapping from CSV
_raw_codes = _load_country_codes_from_csv()
COUNTRY_CODE_TO_NAME = {code: _normalize_csv_name(name) for code, name in _raw_codes.items()}

# Common aliases the model might use
COUNTRY_ALIASES = {
    "usa": "united states", "u.s.": "united states", "u.s.a.": "united states",
    "uk": "united kingdom", "england": "united kingdom", "britain": "united kingdom",
    "south korea": "south korea", "republic of korea": "south korea", "korea": "south korea",
    "north korea": "north korea", "dprk": "north korea",
    "czech republic": "czech republic", "czechia": "czech republic",
    "ivory coast": "ivory coast", "cote d'ivoire": "ivory coast",
    "the netherlands": "netherlands", "holland": "netherlands",
    "uae": "united arab emirates",
    "drc": "congo", "democratic republic of the congo": "congo",
    "republic of congo": "republic of the congo",
    "bosnia": "bosnia and herzegovina",
    "trinidad": "trinidad and tobago",
    "timor-leste": "east timor", "timor leste": "east timor",
    "swaziland": "eswatini",
    "burma": "myanmar",
    "brunei darussalam": "brunei",
    "macau sar": "macau", "macao": "macau",
    "hong kong sar": "hong kong",
    "palestine": "palestine", "west bank": "palestine", "gaza": "palestine",
}


# --- Canonical country matching (single source of truth) -------------------
# Alpha-2 code -> set of accepted normalized names. Version-independent
# (does not depend on the installed pycountry naming). Used only by
# _countries_match below; the name->name COUNTRY_ALIASES above is kept for the
# reverse lookup / neighbor logic and must not be confused with this table.
_COUNTRY_ALIASES = {
    "tr": {"turkey", "turkiye", "türkiye"},
    "us": {"usa", "united states", "united states of america", "america", "u.s.", "u.s.a."},
    "gb": {"uk", "united kingdom", "great britain", "england", "scotland", "wales", "britain"},
    "ru": {"russia", "russian federation"},
    "kr": {"south korea", "republic of korea", "korea, republic of", "korea"},
    "kp": {"north korea", "dprk"},
    "de": {"germany", "deutschland"},
    "cz": {"czech republic", "czechia"},
    "nl": {"netherlands", "the netherlands", "holland"},
    "ae": {"united arab emirates", "uae"},
    "ci": {"ivory coast", "cote d'ivoire", "côte d'ivoire"},
    "ba": {"bosnia", "bosnia and herzegovina"},
    "tt": {"trinidad", "trinidad and tobago"},
    "tl": {"east timor", "timor-leste", "timor leste"},
    "sz": {"swaziland", "eswatini"},
    "mm": {"burma", "myanmar"},
    "ir": {"iran"},
    "sy": {"syria"},
    "la": {"laos"},
    "bo": {"bolivia"},
    "ve": {"venezuela"},
    "vn": {"vietnam"},
    "md": {"moldova"},
    "tz": {"tanzania"},
}


def _normalize_country_canonical(name: str) -> str:
    """Lowercase + strip surrounding whitespace/punctuation (canonical)."""
    import string
    if not name:
        return ""
    return name.strip().lower().rstrip(string.punctuation + " ").strip()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _normalize_country(name: str) -> str:
    """Normalize a country name for comparison."""
    name = name.lower().strip().rstrip(".")
    return COUNTRY_ALIASES.get(name, name)


# Build reverse lookup: country_name -> country_code
_NAME_TO_CODE = {v: k for k, v in COUNTRY_CODE_TO_NAME.items()}
for alias, canonical in COUNTRY_ALIASES.items():
    if canonical in _NAME_TO_CODE:
        _NAME_TO_CODE[alias] = _NAME_TO_CODE[canonical]


def _load_neighbors() -> dict[str, set[str]]:
    """Load country border data. Returns dict: country_code (lower) -> set of neighbor codes (lower)."""
    borders_path = Path(__file__).parent / "GEODATASOURCE-COUNTRY-BORDERS.CSV"
    neighbors: dict[str, set[str]] = {}
    if not borders_path.exists():
        return neighbors
    with open(borders_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["country_code"].strip().lower()
            border_code = row["country_border_code"].strip().lower()
            if not border_code:
                continue
            neighbors.setdefault(code, set()).add(border_code)
            neighbors.setdefault(border_code, set()).add(code)
    return neighbors


_NEIGHBORS = _load_neighbors()


def _is_neighbor(predicted: str, actual_code: str) -> bool:
    """Check if the predicted country is a neighbor of the actual country."""
    predicted_norm = _normalize_country(predicted)
    pred_code = _NAME_TO_CODE.get(predicted_norm, "")
    if not pred_code:
        return False
    actual_neighbors = _NEIGHBORS.get(actual_code, set())
    return pred_code in actual_neighbors


def _countries_match(predicted: str, actual_code: str) -> bool:
    """True iff `predicted` refers to ISO alpha-2 `actual_code`.

    Behaviourally identical to the canonical countries_match:
    1. direct code equality
    2. explicit version-independent alias table
    3. pycountry name set (name/official/common/alpha2/alpha3)
    4. pycountry fuzzy search
    """
    if not predicted or not actual_code:
        return False
    pred = _normalize_country_canonical(predicted)
    code = actual_code.strip().lower()
    if not pred:
        return False
    if pred == code:
        return True
    if pred in _COUNTRY_ALIASES.get(code, set()):
        return True
    try:
        import pycountry
    except Exception:
        return False
    target = pycountry.countries.get(alpha_2=code.upper())
    if target is None:
        return False
    candidates = set()
    for attr in ("name", "official_name", "common_name", "alpha_2", "alpha_3"):
        val = getattr(target, attr, None)
        if val:
            candidates.add(_normalize_country_canonical(val))
    candidates.discard("")
    if pred in candidates:
        return True
    try:
        results = pycountry.countries.search_fuzzy(pred)
        if results and results[0].alpha_2.lower() == code:
            return True
    except LookupError:
        pass
    return False


def _extract_country(country_result: str) -> str:
    """Extract country name from 'Country: X' format."""
    if not country_result:
        return ""
    match = re.search(r"Country:\s*(.+?)(?:\n|$)", country_result)
    if match:
        return match.group(1).strip()
    return country_result.strip().split("\n")[0].strip()


def _extract_coordinates(country_result: str) -> Optional[tuple[float, float]]:
    """Extract coordinates from 'Coordinates: lat, lon' format."""
    match = re.search(r"Coordinates:\s*([-\d.]+)\s*,\s*([-\d.]+)", country_result)
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except ValueError:
            pass
    return None


def _load_ground_truth(csv_path: Path) -> dict:
    """Load ground truth from georc_locations.csv.

    Returns dict: image_stem -> {country_code, country_name, lat, lng}
    """
    gt = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row["filename"].replace(".png", "").replace(".jpg", "")
            code = row["country_code"].lower()
            gt[stem] = {
                "country_code": code,
                "country_name": COUNTRY_CODE_TO_NAME.get(code, code),
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
            }
    return gt


def evaluate(results_dir: Path, gt_path: Path) -> None:
    """Evaluate results against ground truth."""
    gt = _load_ground_truth(gt_path)

    results = []
    for img_dir in sorted(results_dir.iterdir()):
        result_file = img_dir / "result.json"
        if not result_file.exists():
            continue
        try:
            with open(result_file) as f:
                data = json.load(f)
            if data.get("error"):
                results.append({"name": img_dir.name, "error": data["error"]})
            else:
                results.append({
                    "name": img_dir.name,
                    "country": _extract_country(data.get("country_result", "")),
                    "coordinates": _extract_coordinates(data.get("country_result", "")),
                    "timing": data.get("timing", {}).get("total_seconds"),
                    "assessments": data.get("assessments", {}),
                    "progressive_narrowing": data.get("progressive_narrowing", {}),
                    "hypothesis_evaluations": data.get("hypothesis_evaluations", []),
                    "model": data.get("model", "unknown"),
                })
        except (json.JSONDecodeError, OSError) as e:
            results.append({"name": img_dir.name, "error": str(e)})

    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    successful = total - errors

    print("=" * 60)
    print("VLM Council Evaluation")
    print("=" * 60)
    model = next((r["model"] for r in results if "error" not in r), "unknown")
    print(f"Model: {model}")
    print(f"Total images: {total}")
    print(f"Successful:   {successful}")
    print(f"Errors:       {errors}")
    print()

    if successful == 0:
        return

    # Timing statistics
    times = [r["timing"] for r in results if "error" not in r and r.get("timing")]
    if times:
        print(f"Timing (seconds per image):")
        print(f"  Mean:   {sum(times) / len(times):.1f}")
        print(f"  Median: {sorted(times)[len(times) // 2]:.1f}")
        print(f"  Min:    {min(times):.1f}")
        print(f"  Max:    {max(times):.1f}")
        print(f"  Total:  {sum(times):.0f}s ({sum(times) / 60:.1f} min)")
        print()

    # Progressive Narrowing statistics
    path_counts = Counter()
    hyp_eval_confidences = Counter()
    for r in results:
        if "error" in r:
            continue
        pn = r.get("progressive_narrowing", {})
        path_counts[pn.get("path", "?")] += 1
        for e in r.get("hypothesis_evaluations", []):
            hyp_eval_confidences[e.get("confidence", "?")] += 1

    if path_counts:
        print("Progressive Narrowing:")
        for path, count in path_counts.most_common():
            print(f"  Path {path}: {count} ({count * 100 // successful}%)")
        print()

    if hyp_eval_confidences:
        total_evals = sum(hyp_eval_confidences.values())
        print(f"Hypothesis evaluation confidence ({total_evals} total):")
        for conf, count in hyp_eval_confidences.most_common():
            print(f"  {conf:25s}: {count:5d} ({count * 100 // total_evals}%)")
        print()

    # Agent confidence from initial assessments (top candidate)
    agent_names = ["linguistic", "landscape", "botanics", "regulatory", "meta"]
    print("Agent confidence distribution (top candidate):")
    for agent in agent_names:
        confidences = Counter()
        for r in results:
            if "error" in r:
                continue
            assessments = r.get("assessments", {})
            candidates = assessments.get(agent, {}).get("candidates", [])
            if candidates and isinstance(candidates[0], dict):
                conf = candidates[0].get("confidence", "unknown")
            else:
                conf = "insufficient"
            confidences[conf] += 1
        conf_str = ", ".join(f"{k}: {v}" for k, v in sorted(confidences.items()))
        print(f"  {agent:12s}: {conf_str}")
    print()

    # Country accuracy
    correct = 0
    neighbor_hits = 0
    compared = 0
    distances = []
    wrong = []

    for r in results:
        if "error" in r:
            continue
        name = r["name"]
        if name not in gt:
            continue

        compared += 1
        truth = gt[name]
        predicted_raw = r.get("country", "")
        actual_code = truth["country_code"]
        actual_name = truth["country_name"]

        is_correct = _countries_match(predicted_raw, actual_code)
        if is_correct:
            correct += 1
        elif _is_neighbor(predicted_raw, actual_code):
            neighbor_hits += 1

        # Distance
        pred_coords = r.get("coordinates")
        dist = None
        if pred_coords:
            dist = _haversine_km(pred_coords[0], pred_coords[1], truth["lat"], truth["lng"])
            distances.append(dist)

        if not is_correct:
            wrong.append({
                "image": name,
                "predicted": predicted_raw if predicted_raw else "?",
                "actual": f"{actual_name} ({actual_code})",
                "distance_km": round(dist) if dist else None,
                "is_neighbor": _is_neighbor(predicted_raw, actual_code),
            })

    if compared > 0:
        accuracy = correct / compared * 100
        print(f"Country accuracy: {correct}/{compared} ({accuracy:.1f}%)")
        neighbor_acc = neighbor_hits / compared * 100
        combined = correct + neighbor_hits
        combined_acc = combined / compared * 100
        print(f"Neighbor hits:    {neighbor_hits}/{compared} ({neighbor_acc:.1f}%)")
        print(f"Correct Country or Neighbor: {combined}/{compared} ({combined_acc:.1f}%)")
        print()

    if distances:
        distances_sorted = sorted(distances)
        median_idx = len(distances_sorted) // 2
        within_150 = sum(1 for d in distances if d <= 150)
        within_750 = sum(1 for d in distances if d <= 750)
        within_2500 = sum(1 for d in distances if d <= 2500)
        print(f"Distance (km) — {len(distances)} images with coordinates:")
        print(f"  Mean:     {sum(distances) / len(distances):.0f} km")
        print(f"  Median:   {distances_sorted[median_idx]:.0f} km")
        print(f"  Min:      {min(distances):.0f} km")
        print(f"  Max:      {max(distances):.0f} km")
        print(f"  ≤150 km:  {within_150}/{len(distances)} ({within_150 / len(distances) * 100:.0f}%)")
        print(f"  ≤750 km:  {within_750}/{len(distances)} ({within_750 / len(distances) * 100:.0f}%)")
        print(f"  ≤2500 km: {within_2500}/{len(distances)} ({within_2500 / len(distances) * 100:.0f}%)")
        print()

    # Top predicted countries
    predicted_countries = Counter(r["country"] for r in results if "error" not in r and r.get("country"))
    print("Top predicted countries:")
    for country, count in predicted_countries.most_common(15):
        print(f"  {country}: {count}")
    print()

    # Wrong predictions
    if wrong:
        print(f"Wrong predictions ({len(wrong)}):")
        for w in wrong[:30]:
            dist_str = f" ({w['distance_km']} km)" if w["distance_km"] else ""
            neighbor_flag = " [NEIGHBOR]" if w.get("is_neighbor") else ""
            print(f"  {w['image']:30s} predicted={w['predicted']:20s} actual={w['actual']}{dist_str}{neighbor_flag}")
        if len(wrong) > 30:
            print(f"  ... and {len(wrong) - 30} more")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate VLM Council results")
    parser.add_argument("results_dir", help="Directory with result.json files")
    parser.add_argument("ground_truth", help="Path to georc_locations.csv")
    args = parser.parse_args()

    evaluate(
        results_dir=Path(args.results_dir),
        gt_path=Path(args.ground_truth),
    )


if __name__ == "__main__":
    main()