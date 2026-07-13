"""Load pre-computed vision results and ground-truth coordinates into SampleRecords."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import reverse_geocoder


@dataclass
class SampleRecord:
    location_id: str
    image_filename: str
    gt_lat: float
    gt_lng: float
    gt_country: str
    general_description: str
    crop_descriptions: list[str] = field(default_factory=list)
    result_dir: Path = field(default_factory=Path)


def _resolve_country(lat: float, lng: float) -> str:
    """Offline reverse-geocode a lat/lng to a country name."""
    results = reverse_geocoder.search([(lat, lng)], verbose=False)
    cc = results[0].get("cc", "")
    # reverse_geocoder returns ISO-3166-1 alpha-2 codes; map to full name
    try:
        import pycountry
        country = pycountry.countries.get(alpha_2=cc)
        return country.name if country else cc
    except Exception:
        return cc


def _descriptions_from_result(data: dict) -> tuple[str, list[str]]:
    """Extract general_description and crop_descriptions from a result.json dict."""
    general = data.get("scene_description", data.get("general_description", ""))
    crops: list[str] = []

    # Collect the main description first
    if data.get("general_description") and data["general_description"] != general:
        # result.json uses "scene_description" for the full text; "general_description"
        # may or may not be present separately - use whichever is non-empty
        general = data.get("scene_description") or data.get("general_description", "")

    # Gather focused crop descriptions from the details list
    for detail in data.get("details", []):
        focused = detail.get("focused_description")
        if focused:
            name = detail.get("name", "")
            crops.append(f"{name}: {focused}" if name else focused)

    return general, crops


def load_samples(
    results_dir: Path,
    mapping_csv: Path,
) -> list[SampleRecord]:
    """Return one SampleRecord per location that has both a result.json and a CSV row.

    Only locations whose result directory exists under ``results_dir`` are included.
    """
    # Build filename -> (lat, lng, country_code) lookup from the ground-truth CSV.
    # Supports two formats:
    #   - original:  image_filename, lat, lng
    #   - georc:     filename, lat, lng, country_code
    gt: dict[str, tuple[float, float, str]] = {}
    with mapping_csv.open() as fh:
        for row in csv.DictReader(fh):
            fname = row.get("image_filename") or row.get("filename", "")
            cc = row.get("country_code", "")
            gt[fname] = (float(row["lat"]), float(row["lng"]), cc)

    samples: list[SampleRecord] = []

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue
        result_json = result_dir / "result.json"
        if not result_json.exists():
            continue

        location_id = result_dir.name
        # Try .jpg first (original format), then .png (GeoRC format)
        image_filename = f"{location_id}.jpg"
        if image_filename not in gt:
            image_filename = f"{location_id}.png"
        if image_filename not in gt:
            continue

        lat, lng, cc = gt[image_filename]
        # Use country_code to resolve country name if available, else reverse-geocode
        if cc:
            try:
                import pycountry
                country_obj = pycountry.countries.get(alpha_2=cc.upper())
                gt_country = country_obj.name if country_obj else _resolve_country(lat, lng)
            except Exception:
                gt_country = _resolve_country(lat, lng)
        else:
            gt_country = _resolve_country(lat, lng)

        data = json.loads(result_json.read_text())
        general, crops = _descriptions_from_result(data)

        samples.append(
            SampleRecord(
                location_id=location_id,
                image_filename=image_filename,
                gt_lat=lat,
                gt_lng=lng,
                gt_country=gt_country,
                general_description=general,
                crop_descriptions=crops,
                result_dir=result_dir,
            )
        )

    return samples
