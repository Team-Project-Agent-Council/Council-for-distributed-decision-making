"""Geo-spatial bias metrics.

Computes per-image and aggregate geographic statistics, with significance tests
for north/east bias and a country-pair confusion matrix.

Outputs:
  geo_metrics.json
  plots/error_distribution.png, three-panel: lat error, lng error, haversine
  plots/bearing_rose.png, polar histogram of error bearings
  plots/confusion_matrix.png, top-N most-confused country pairs
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from eval_pnt._style import STYLE, setup_plot_style
from eval_pnt.loader import RunRecord, load_run


def _bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Initial bearing from (lat1, lng1) to (lat2, lng2), in degrees [0, 360).

    0° = north, 90° = east, etc.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlng = math.radians(lng2 - lng1)
    x = math.sin(dlng) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlng)
    brg = math.degrees(math.atan2(x, y))
    return (brg + 360.0) % 360.0


def _quadrant(lat_err: float, lng_err: float) -> str:
    """Return 'NE', 'NW', 'SE', or 'SW' based on signed errors."""
    ns = "N" if lat_err >= 0 else "S"
    ew = "E" if lng_err >= 0 else "W"
    return ns + ew


def _t_test_one_sample(values: list[float]) -> tuple[float, float]:
    """Return (t_statistic, two-sided p-value) for H0: mean=0.

    Pure-python so we don't force a scipy dep at import time.
    """
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0:
        return float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0
    t = mean / (sd / math.sqrt(n))
    # Use scipy for the p-value if available; otherwise fall back to a normal
    # approximation (good enough for n > 30 which we'll have).
    try:
        from scipy import stats
        p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))
    except ImportError:
        # Two-sided p-value from the normal approx
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
    return float(t), p


def compute(records: list[RunRecord]) -> dict:
    """Compute all geo metrics over a list of records."""
    valid = [r for r in records if r.haversine_km is not None]

    lat_errors = [r.lat_error for r in valid if r.lat_error is not None]
    lng_errors = [r.lng_error for r in valid if r.lng_error is not None]
    haversines = [r.haversine_km for r in valid]

    bearings: list[float] = []
    quadrants: Counter = Counter()
    for r in valid:
        if r.pred_lat is None or r.pred_lng is None:
            continue
        bearings.append(_bearing_deg(r.truth_lat, r.truth_lng, r.pred_lat, r.pred_lng))
        quadrants[_quadrant(r.lat_error or 0.0, r.lng_error or 0.0)] += 1

    # Bias significance tests
    t_lat, p_lat = _t_test_one_sample(lat_errors)
    t_lng, p_lng = _t_test_one_sample(lng_errors)

    # Country-pair confusion (only when prediction is wrong)
    confusion: Counter = Counter()
    for r in records:
        if not r.is_correct and r.pred_country and r.truth_country_name:
            pair = (r.truth_country_name, r.pred_country)
            confusion[pair] += 1
    top_confusions = [
        {"truth": t, "predicted": p, "count": n}
        for (t, p), n in confusion.most_common(20)
    ]

    # Asymmetry: for each pair (A, B) check whether (B, A) also exists
    asymmetric_pairs = []
    seen = set()
    for (a, b), n_ab in confusion.items():
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        n_ba = confusion.get((b, a), 0)
        if n_ab + n_ba >= 2 and abs(n_ab - n_ba) >= 2:
            asymmetric_pairs.append({
                "country_a": a, "country_b": b,
                "a_predicted_as_b": n_ab,
                "b_predicted_as_a": n_ba,
                "asymmetry": int(n_ab - n_ba),
            })
    asymmetric_pairs.sort(key=lambda x: abs(x["asymmetry"]), reverse=True)

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        arr = np.asarray(xs, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    return {
        "n_total": len(records),
        "n_with_coords": len(valid),
        "country_accuracy": (
            sum(1 for r in records if r.is_correct) / len(records) if records else 0.0
        ),
        "haversine_km": _stats(haversines),
        "lat_error_deg": _stats(lat_errors),
        "lng_error_deg": _stats(lng_errors),
        "abs_lat_error_deg": _stats([abs(x) for x in lat_errors]),
        "abs_lng_error_deg": _stats([abs(x) for x in lng_errors]),
        "north_bias_test": {
            "t_statistic": t_lat, "p_value": p_lat,
            "interpretation": _interpret_bias(t_lat, p_lat, "north", "south"),
        },
        "east_bias_test": {
            "t_statistic": t_lng, "p_value": p_lng,
            "interpretation": _interpret_bias(t_lng, p_lng, "east", "west"),
        },
        "quadrants": dict(quadrants),
        "top_confusions": top_confusions,
        "asymmetric_confusion_pairs": asymmetric_pairs[:15],
        "_lat_errors": lat_errors,
        "_lng_errors": lng_errors,
        "_haversines": haversines,
        "_bearings": bearings,
    }


def _interpret_bias(t: float, p: float, pos: str, neg: str) -> str:
    if math.isnan(t):
        return "insufficient data"
    direction = pos if t > 0 else neg
    if p < 0.01:
        return f"strong {direction} bias (p={p:.4f})"
    if p < 0.05:
        return f"{direction} bias (p={p:.4f})"
    return f"no significant bias (p={p:.4f})"


def plot_error_distribution(metrics: dict, out_path: Path) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label, unit in [
        (axes[0], "_lat_errors", "Latitude error (pred − truth)", "deg"),
        (axes[1], "_lng_errors", "Longitude error (pred − truth)", "deg"),
        (axes[2], "_haversines", "Haversine distance", "km"),
    ]:
        data = metrics.get(key, [])
        if data:
            ax.hist(data, bins=30, edgecolor="black", alpha=0.75, color=STYLE.primary)
            mean = float(np.mean(data))
            ax.axvline(mean, color=STYLE.error, linestyle="--", label=f"mean={mean:.1f}")
            if key != "_haversines":
                ax.axvline(0, color="black", linestyle=":")
            ax.legend()
        ax.set_title(label)
        ax.set_xlabel(unit)
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bearing_rose(metrics: dict, out_path: Path) -> None:
    setup_plot_style()
    bearings = metrics.get("_bearings", [])
    if not bearings:
        return
    n_bins = 16
    bins = np.linspace(0, 360, n_bins + 1)
    counts, _ = np.histogram(bearings, bins=bins)
    theta = np.deg2rad(bins[:-1] + (360 / n_bins) / 2)
    width = 2 * np.pi / n_bins

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(theta, counts, width=width, edgecolor="black", alpha=0.85, color=STYLE.primary)
    ax.set_title("Error bearing rose (truth → prediction)", pad=20)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion(metrics: dict, out_path: Path, top_n: int = 15) -> None:
    setup_plot_style()
    pairs = metrics.get("top_confusions", [])[:top_n]
    if not pairs:
        return
    labels = [f"{p['truth']} → {p['predicted']}" for p in pairs]
    counts = [p["count"] for p in pairs]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(pairs))))
    ax.barh(range(len(pairs)), counts, color=STYLE.primary)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title(f"Top {len(pairs)} confusion pairs")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(results_dir: Path, gt_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    records = load_run(results_dir, gt_csv)
    metrics = compute(records)

    plot_error_distribution(metrics, plots_dir / "error_distribution.png")
    plot_bearing_rose(metrics, plots_dir / "bearing_rose.png")
    plot_confusion(metrics, plots_dir / "confusion_matrix.png")

    # Strip raw arrays before serializing (keep file small)
    serializable = {k: v for k, v in metrics.items() if not k.startswith("_")}
    out_file = out_dir / "geo_metrics.json"
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[geo] wrote {out_file}")
    print(f"[geo] n={metrics['n_total']}, accuracy={metrics['country_accuracy']:.1%}, "
          f"mean haversine={metrics['haversine_km'].get('mean', 0):.0f} km")
    print(f"[geo] {metrics['north_bias_test']['interpretation']}")
    print(f"[geo] {metrics['east_bias_test']['interpretation']}")
    return serializable
