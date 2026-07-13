"""Build and print the evaluation report; write evaluation_results.csv."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from rich.console import Console
from rich.table import Table

from evaluation.loader import SampleRecord
from evaluation.metrics import (
    country_match,
    geoguessr_score,
    haversine_km,
    parse_country_result,
)
from evaluation.runner import CouncilRunResult


def build_report(
    samples: list[SampleRecord],
    results: list[CouncilRunResult],
) -> list[dict]:
    """Zip samples and results into a list of per-row dicts."""
    by_id = {r.location_id: r for r in results}
    rows = []
    for s in samples:
        r = by_id.get(s.location_id)
        if r is None:
            continue

        pred_country, pred_lat, pred_lon = parse_country_result(r.country_result)
        error = r.error or ""

        if not math.isnan(pred_lat) and not math.isnan(pred_lon) and not error:
            dist_km = haversine_km(s.gt_lat, s.gt_lng, pred_lat, pred_lon)
        else:
            dist_km = float("nan")

        score = geoguessr_score(dist_km)
        match = country_match(pred_country, s.gt_country)

        rows.append(
            {
                "location_id": s.location_id,
                "gt_country": s.gt_country,
                "predicted_country": pred_country,
                "country_match": match,
                "dist_km": round(dist_km, 1) if not math.isnan(dist_km) else "",
                "geoguessr_score": score,
                "pred_lat": round(pred_lat, 4) if not math.isnan(pred_lat) else "",
                "pred_lon": round(pred_lon, 4) if not math.isnan(pred_lon) else "",
                "error": error,
            }
        )
    return rows


def print_report(rows: list[dict]) -> None:
    """Print a Rich table to stdout."""
    console = Console()

    table = Table(
        title="Council Evaluation Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Location ID", style="dim", no_wrap=True, max_width=42)
    table.add_column("GT Country")
    table.add_column("Predicted Country")
    table.add_column("Match", justify="center")
    table.add_column("Dist (km)", justify="right")
    table.add_column("Score", justify="right")

    valid_dists = [r["dist_km"] for r in rows if r["dist_km"] != ""]
    valid_scores = [r["geoguessr_score"] for r in rows]
    n_match = sum(1 for r in rows if r["country_match"])

    for r in rows:
        match_icon = "[green][ok][/green]" if r["country_match"] else "[red][x][/red]"
        err_suffix = f" [red][ERR][/red]" if r["error"] else ""
        table.add_row(
            r["location_id"] + err_suffix,
            r["gt_country"],
            r["predicted_country"] or "[dim]-[/dim]",
            match_icon,
            str(r["dist_km"]) if r["dist_km"] != "" else "[dim]-[/dim]",
            str(r["geoguessr_score"]),
        )

    console.print(table)

    n = len(rows)
    acc = n_match / n * 100 if n else 0
    mean_dist = sum(valid_dists) / len(valid_dists) if valid_dists else float("nan")
    median_dist = statistics.median(valid_dists) if valid_dists else float("nan")
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    median_score = statistics.median(valid_scores) if valid_scores else 0

    console.print(f"\n[bold]Summary ({n} images)[/bold]")
    console.print(f"  Country accuracy : [cyan]{acc:.1f}%[/cyan] ({n_match}/{n})")
    console.print(f"  Mean distance    : [cyan]{mean_dist:.0f} km[/cyan]" if not math.isnan(mean_dist) else "  Mean distance    : [dim]-[/dim]")
    console.print(f"  Median distance  : [cyan]{median_dist:.0f} km[/cyan]" if not math.isnan(median_dist) else "  Median distance  : [dim]-[/dim]")
    console.print(f"  Mean GeoGuessr   : [cyan]{mean_score:.0f} / 5000[/cyan]")
    console.print(f"  Median GeoGuessr : [cyan]{median_score:.0f} / 5000[/cyan]")


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write per-image metrics to a CSV file."""
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
