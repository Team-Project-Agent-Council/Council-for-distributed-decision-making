"""CLI orchestrator for the cross-approach geographic bias analysis.

Usage (from repo root):
    python -m bias_analysis.run
    python -m bias_analysis.run --out bias_analysis/output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_analysis import bias, sources
from bias_analysis.loader import load_run


def _strip_internal(metrics: dict) -> dict:
    """Recursively drop the raw ``_``-prefixed arrays before serialising."""
    if isinstance(metrics, dict):
        return {k: _strip_internal(v) for k, v in metrics.items() if not k.startswith("_")}
    return metrics


def _write_report(all_metrics: dict, out_dir: Path) -> None:
    combined = all_metrics["combined"]
    per = all_metrics["per_approach"]

    lines = ["# Cross-Approach Geographic Bias\n"]
    lines.append(
        f"Pooled over {len(per)} approaches, "
        f"{combined['n_total']} predictions "
        f"({combined['n_with_coords']} with coordinates).\n"
    )

    clustered = combined.get("clustered_bias_test")
    lines.append("## Overall verdict\n")
    if clustered:
        lines.append(
            f"Primary test: **one-sample t-test on per-image mean errors** "
            f"(n = {clustered['n_images']} distinct images, each averaged over "
            f"{clustered['n_approaches']} approaches). Averaging per image before "
            "testing avoids treating the same image's repeated predictions as "
            "independent, which would understate the standard error.\n"
        )
        lines.append(
            f"- North/South: **{clustered['north_bias_test']['interpretation']}** "
            f"(t={clustered['north_bias_test']['t_statistic']:.2f})"
        )
        lines.append(
            f"- East/West: **{clustered['east_bias_test']['interpretation']}** "
            f"(t={clustered['east_bias_test']['t_statistic']:.2f})"
        )
        lines.append(
            f"- Mean per-image signed error: "
            f"{clustered['lat_error_deg'].get('mean', 0):+.2f}° lat, "
            f"{clustered['lng_error_deg'].get('mean', 0):+.2f}° lng\n"
        )
        lines.append(
            "_Naive pooled test (all "
            f"{combined['n_with_coords']} rows, treats repeated images as "
            "independent — reported for reference only): "
            f"N/S {combined['north_bias_test']['interpretation']}, "
            f"E/W {combined['east_bias_test']['interpretation']}._\n"
        )
    else:
        lines.append(f"- North/South: **{combined['north_bias_test']['interpretation']}**")
        lines.append(f"- East/West: **{combined['east_bias_test']['interpretation']}**")

    lines.append(
        f"- Mean haversine: {combined['haversine_km'].get('mean', 0):.0f} km "
        f"(median {combined['haversine_km'].get('median', 0):.0f} km)\n"
    )

    lines.append("## Per-approach\n")
    lines.append("| Approach | n | acc | mean lat err | mean lng err | N-bias p | E-bias p |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for name, m in per.items():
        lines.append(
            f"| {name} | {m['n_total']} | {m['country_accuracy']:.1%} | "
            f"{m['lat_error_deg'].get('mean', 0):+.2f}° | "
            f"{m['lng_error_deg'].get('mean', 0):+.2f}° | "
            f"{m['north_bias_test']['p_value']:.3f} | "
            f"{m['east_bias_test']['p_value']:.3f} |"
        )
    lines.append("")
    lines.append("Positive lat error = prediction north of truth; "
                 "positive lng error = prediction east of truth.\n")
    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-approach geographic bias analysis")
    ap.add_argument("--out", default=None, help="output dir (default: bias_analysis/output)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "output"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    by_approach: dict[str, list] = {}
    for src in sources.SOURCES:
        if not src.results_dir.exists():
            print(f"[bias] SKIP {src.name}: {src.results_dir} not found")
            continue
        if not src.gt_csv.exists():
            print(f"[bias] SKIP {src.name}: {src.gt_csv} not found")
            continue
        records = load_run(src.name, src.results_dir, src.gt_csv)
        by_approach[src.name] = records
        n_coords = sum(1 for r in records if r.has_coords)
        print(f"[bias] {src.name}: {len(records)} records, {n_coords} with coords")

    if not by_approach:
        print("[bias] no approaches loaded — nothing to do")
        return

    all_metrics = bias.compute_all(by_approach)
    combined = all_metrics["combined"]
    per = all_metrics["per_approach"]

    # Plots
    bias.plot_bearing_rose(combined, plots_dir / "bearing_rose_combined.png",
                           title="Pooled error bearing rose (all approaches)")
    bias.plot_bearing_rose_by_approach(per, plots_dir / "bearing_rose_by_approach.png")
    bias.plot_error_distribution(combined, plots_dir / "error_distribution.png")
    bias.plot_quadrant_bars(per, combined, plots_dir / "quadrant_bars.png")
    bias.plot_error_map(combined, plots_dir / "error_map_combined.png")
    bias.plot_over_under_map(combined, plots_dir / "over_under_map_combined.png",
                             scope_label="pooled across all 7 approaches")
    # Per-approach over/under maps
    ou_dir = plots_dir / "over_under_by_approach"
    ou_dir.mkdir(exist_ok=True)
    for name, m in per.items():
        slug = name.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
        bias.plot_over_under_map(m, ou_dir / f"{slug}.png", scope_label=name)

    # JSON (strip raw arrays)
    serializable = _strip_internal(all_metrics)
    (out_dir / "bias_metrics.json").write_text(json.dumps(serializable, indent=2))

    _write_report(all_metrics, out_dir)

    print(f"\n[bias] wrote {out_dir / 'bias_metrics.json'}")
    print(f"[bias] combined n={combined['n_total']}, "
          f"with_coords={combined['n_with_coords']}, "
          f"accuracy={combined['country_accuracy']:.1%}, "
          f"mean haversine={combined['haversine_km'].get('mean', 0):.0f} km")
    clustered = combined.get("clustered_bias_test")
    if clustered:
        print(f"[bias] primary (per-image, n={clustered['n_images']}): "
              f"{clustered['north_bias_test']['interpretation']}")
        print(f"[bias] primary (per-image, n={clustered['n_images']}): "
              f"{clustered['east_bias_test']['interpretation']}")
    else:
        print(f"[bias] {combined['north_bias_test']['interpretation']}")
        print(f"[bias] {combined['east_bias_test']['interpretation']}")


if __name__ == "__main__":
    main()
