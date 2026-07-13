"""GeoRC test - runs the council on a single pre-computed vision result.

Usage
-----
    python georc_test.py 67gLC5CcGgkQIWEW_2

    python georc_test.py 67gLC5CcGgkQIWEW_2 --results-dir "results GeoRC"

    python georc_test.py 67gLC5CcGgkQIWEW_2 --verbose
"""

from __future__ import annotations

import os

if os.environ.get("CHROMA_SQLITE_PATCH"):
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import asyncio
import json
import math
from pathlib import Path

import click

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass


def _load_record(result_dir: Path):
    """Load a single result.json into a SampleRecord."""
    from evaluation.loader import SampleRecord

    result_json = result_dir / "result.json"
    if not result_json.exists():
        raise FileNotFoundError(f"No result.json found in {result_dir}")

    data = json.loads(result_json.read_text())

    general = data.get("scene_description") or data.get("general_description", "")

    crops = []
    for detail in data.get("details", []):
        focused = detail.get("focused_description")
        if focused:
            name = detail.get("name", "")
            crops.append(f"{name}: {focused}" if name else focused)

    location_id = result_dir.name

    return SampleRecord(
        location_id=location_id,
        image_filename=f"{location_id}.jpg",
        gt_lat=float("nan"),
        gt_lng=float("nan"),
        gt_country="",
        general_description=general,
        crop_descriptions=crops,
        result_dir=result_dir,
    )


async def _run(result_dir: Path, verbose: bool, output_name: str) -> None:
    from evaluation.runner import run_single
    from evaluation.metrics import parse_country_result
    import evaluation.runner as runner_module

    # Set the output filename for this run
    runner_module.RESULT_FILENAME = output_name

    record = _load_record(result_dir)

    click.echo(f"\nRunning council for: {record.location_id}")
    click.echo(f"  Crops found : {len(record.crop_descriptions)}")
    click.echo(f"  Description : {record.general_description[:100].replace(chr(10), ' ')!r}...")
    click.echo("")

    result = await run_single(record, run_name_prefix="georc-test", verbose=verbose)

    pred_country, pred_lat, pred_lon = parse_country_result(result.country_result)

    click.echo("\n" + "=" * 60)
    if result.error:
        click.echo(f"[ERROR] {result.error}")
    else:
        click.echo(f"Predicted country : {pred_country or '-'}")
        if not math.isnan(pred_lat):
            click.echo(f"Coordinates       : {pred_lat:.4f}, {pred_lon:.4f}")
        click.echo(f"\nFull judge output:\n{result.country_result}")

    output_path = result_dir / output_name
    click.echo(f"\nResult saved to: {output_path}")


@click.command()
@click.argument("location_id")
@click.option(
    "--results-dir",
    default="results GeoRC",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Parent directory containing the location folders.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-node progress while the council runs.",
)
@click.option(
    "--output-name",
    default="council_optimized_result.json",
    show_default=True,
    help="Filename for the result JSON (stored in the location dir).",
)
def main(location_id: str, results_dir: str, verbose: bool, output_name: str) -> None:
    result_dir = Path(results_dir) / location_id
    if not result_dir.exists():
        raise click.BadParameter(
            f"Directory not found: {result_dir}", param_hint="LOCATION_ID"
        )
    asyncio.run(_run(result_dir, verbose=verbose, output_name=output_name))


if __name__ == "__main__":
    main()
