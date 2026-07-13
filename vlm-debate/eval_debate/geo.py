"""Geo-spatial bias metrics for the Debate approach."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_debate._style import STYLE, setup_plot_style
from eval_debate.loader import RunRecord, load_run, haversine_km


def _bearing(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _bias_test(errors: list[float]) -> dict:
    n = len(errors)
    if n < 2:
        return {"n": n, "mean": 0.0, "p_value": 1.0, "interpretation": "insufficient data"}
    mean_err = sum(errors) / n
    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_1samp(errors, 0.0)
    except ImportError:
        variance = sum((e - mean_err) ** 2 for e in errors) / (n - 1)
        se = math.sqrt(variance / n)
        t_stat = mean_err / se if se > 0 else 0.0
        # Approximate p-value via normal distribution
        p_val = 2 * (1 - _norm_cdf(abs(t_stat)))

    direction = "north" if mean_err > 0 else "south"
    sig = bool(p_val < 0.05)
    interp = (
        f"Significant {direction} bias (p={p_val:.4f}, mean={mean_err:+.2f}°)"
        if sig else
        f"No significant bias (p={p_val:.4f}, mean={mean_err:+.2f}°)"
    )
    return {"n": n, "mean": round(mean_err, 4), "p_value": round(float(p_val), 6),
            "significant": sig, "interpretation": interp}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _plot_error_dist(lat_errors, lon_errors, hav_errors, out_path: Path) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, label in zip(
        axes,
        [lat_errors, lon_errors, hav_errors],
        ["Latitude error (°)", "Longitude error (°)", "Haversine distance (km)"],
    ):
        ax.hist(data, bins=30, color=STYLE.primary, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
    fig.suptitle("Prediction error distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_bearing_rose(bearings: list[float], out_path: Path) -> None:
    setup_plot_style()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="polar")
    bins = 16
    counts, edges = np.histogram(np.radians(bearings), bins=bins, range=(0, 2 * math.pi))
    width = 2 * math.pi / bins
    ax.bar(edges[:-1], counts, width=width, color=STYLE.primary, alpha=0.8, edgecolor="white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Error bearing rose", pad=20)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_confusion(confusions: list[tuple[str, str]], out_path: Path) -> None:
    setup_plot_style()
    counter = Counter(confusions)
    top = counter.most_common(15)
    if not top:
        return
    labels = [f"{t}→{p}" for (t, p), _ in top]
    counts = [c for _, c in top]
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(labels))
    ax.barh(y, counts, color=STYLE.primary, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Top confusion pairs (truth → predicted)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(results_dir: Path, gt_csv: Path, out_dir: Path) -> dict:
    records = load_run(results_dir, gt_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    n = len(records)
    correct = sum(1 for r in records if r.is_correct)

    hav_vals = [r.haversine_km for r in records if r.haversine_km is not None]
    lat_errors = [r.pred_lat - r.truth_lat for r in records if r.pred_lat is not None]
    lon_errors = [r.pred_lon - r.truth_lon for r in records if r.pred_lon is not None]
    bearings = []
    for r in records:
        if r.pred_lat is not None and r.pred_lon is not None:
            bearings.append(_bearing(r.truth_lat, r.truth_lon, r.pred_lat, r.pred_lon))

    confusions = [
        (r.truth_country_name, r.pred_country)
        for r in records if not r.is_correct and r.pred_country
    ]
    counter = Counter(confusions)
    top_conf = [
        {"truth": t, "predicted": p, "count": c}
        for (t, p), c in counter.most_common(30)
    ]

    lo, hi = _wilson_ci(correct, n)
    summary = {
        "n_total": n,
        "country_accuracy": correct / n if n else 0.0,
        "ci_low": lo, "ci_high": hi,
        "haversine_km": {
            "mean": float(np.mean(hav_vals)) if hav_vals else None,
            "median": float(np.median(hav_vals)) if hav_vals else None,
            "p25": float(np.percentile(hav_vals, 25)) if hav_vals else None,
            "p75": float(np.percentile(hav_vals, 75)) if hav_vals else None,
        },
        "north_bias_test": _bias_test(lat_errors),
        "east_bias_test": _bias_test(lon_errors),
        "top_confusions": top_conf,
    }

    _plot_error_dist(lat_errors, lon_errors, hav_vals, plots_dir / "error_distribution.png")
    _plot_bearing_rose(bearings, plots_dir / "bearing_rose.png")
    _plot_confusion(confusions, plots_dir / "confusion_matrix.png")

    out_file = out_dir / "geo_metrics.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[geo] n={n}  accuracy={correct/n:.1%}  median_hav={summary['haversine_km']['median']:.0f} km")
    return summary
