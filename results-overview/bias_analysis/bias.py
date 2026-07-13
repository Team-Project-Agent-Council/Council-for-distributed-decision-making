"""Aggregation, significance tests, and plots for cross-approach geographic bias.

Algorithms (_bearing_deg, _quadrant, _t_test_one_sample, _interpret_bias,
_stats, bearing-rose / error-distribution plots) mirror the per-approach
implementation in ``vlm-pn-ph-tournament/eval_pnt/geo.py``; they are reproduced
here so this package stays standalone. The aggregation and by-approach /
world-map plots are new.
"""

from __future__ import annotations

import math
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from bias_analysis.loader import Record, _normalize_country

# --- Vector output (SVG + PDF alongside every PNG) -------------------------
# Patch Figure.savefig once so any .png write also emits sibling .svg/.pdf with
# the same basename, covering every plot in this module without touching call
# sites. Mirrors the pattern in eval_reguess/_style.py.
import os as _os

from matplotlib.figure import Figure as _Figure

_VECTOR_FORMATS = ("svg", "pdf")
_orig_savefig = _Figure.savefig


def _savefig_with_vectors(self, fname, *args, **kwargs):
    result = _orig_savefig(self, fname, *args, **kwargs)
    if isinstance(fname, (str, _os.PathLike)):
        path = _os.fspath(fname)
        if path.lower().endswith(".png"):
            base = path[:-4]
            vector_kwargs = dict(kwargs)
            vector_kwargs.pop("dpi", None)  # irrelevant for vector output
            for ext in _VECTOR_FORMATS:
                try:
                    _orig_savefig(self, f"{base}.{ext}", *args, **vector_kwargs)
                except Exception:
                    pass  # never let vector export break the primary PNG
    return result


if getattr(_Figure.savefig, "_emits_vectors", False) is False:
    _savefig_with_vectors._emits_vectors = True
    _Figure.savefig = _savefig_with_vectors

# Minimal inlined style (avoids importing any per-approach _style module).
_PRIMARY = "#4C72B0"
_ERROR = "#C44E52"


def setup_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })


# Country matching (trimmed alias table + pycountry fallback)

_COUNTRY_ALIASES = {
    "tr": {"turkey", "turkiye", "türkiye"},
    "us": {"usa", "united states", "united states of america", "america"},
    "gb": {"uk", "united kingdom", "great britain", "england", "britain"},
    "ru": {"russia", "russian federation"},
    "kr": {"south korea", "republic of korea", "korea"},
    "de": {"germany", "deutschland"},
    "cz": {"czech republic", "czechia"},
    "nl": {"netherlands", "the netherlands", "holland"},
    "ae": {"united arab emirates", "uae"},
    "ci": {"ivory coast", "cote d'ivoire", "côte d'ivoire"},
    "ba": {"bosnia", "bosnia and herzegovina"},
}


def countries_match(predicted: str, actual_code: str) -> bool:
    if not predicted or not actual_code:
        return False
    pred = _normalize_country(predicted)
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
            candidates.add(_normalize_country(val))
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


# Geometry / stats (mirrors eval_pnt/geo.py)

def _iso_a3_from_alpha2(code: str) -> str | None:
    if not code:
        return None
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=code.upper())
        return c.alpha_3 if c else None
    except Exception:
        return None


def _iso_a3_for_name(name: str) -> str | None:
    if not name:
        return None
    try:
        import pycountry
    except ImportError:
        return None
    n = name.strip()
    for getter in (
        lambda: pycountry.countries.get(name=n.title()),
        lambda: pycountry.countries.get(common_name=n.title()),
        lambda: pycountry.countries.get(official_name=n.title()),
    ):
        try:
            c = getter()
            if c is not None:
                return c.alpha_3
        except Exception:
            pass
    try:
        guess = pycountry.countries.search_fuzzy(n)
        if guess:
            return guess[0].alpha_3
    except Exception:
        return None
    return None


def _natural_earth_path():
    """Locate a Natural Earth admin-0 shapefile, or None.

    Checks $VLM_NATURAL_EARTH_PATH, then the repo-level
    data/natural_earth/ne_110m_admin_0_countries.shp shared by the approaches.
    """
    from pathlib import Path
    env = _os.environ.get("VLM_NATURAL_EARTH_PATH")
    if env and Path(env).exists():
        return Path(env)
    repo_root = Path(__file__).resolve().parent.parent
    cand = repo_root / "data" / "natural_earth" / "ne_110m_admin_0_countries.shp"
    return cand if cand.exists() else None


def _load_world():
    """Load the Natural Earth world GeoDataFrame, or None if unavailable."""
    ne = _natural_earth_path()
    if ne is None:
        return None
    try:
        import geopandas as gpd
        return gpd.read_file(str(ne))
    except Exception:
        return None


def _world_iso_col(world) -> str | None:
    for c in ("ISO_A3", "ADM0_A3", "iso_a3"):
        if c in world.columns:
            return c
    return None


def _bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlng = math.radians(lng2 - lng1)
    x = math.sin(dlng) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlng)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _quadrant(lat_err: float, lng_err: float) -> str:
    return ("N" if lat_err >= 0 else "S") + ("E" if lng_err >= 0 else "W")


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_sf2(t: float, df: float) -> float:
    """Exact two-sided survival p-value for Student's t (no scipy needed)."""
    if not math.isfinite(t):
        return 0.0
    x = df / (df + t * t)
    return float(_betai(df / 2.0, 0.5, x))


def _t_test_one_sample(values: list[float]) -> tuple[float, float]:
    """Two-sided one-sample t-test against H0: mean = 0.

    Uses scipy when available; otherwise an exact Student-t p-value via the
    regularised incomplete beta function (accurate at any df, unlike a normal
    approximation).
    """
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0:
        return (float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0)
    t = mean / (sd / math.sqrt(n))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except ImportError:
        p = _student_t_sf2(t, n - 1)
    return float(t), p


def _interpret_bias(t: float, p: float, pos: str, neg: str) -> str:
    if math.isnan(t):
        return "insufficient data"
    direction = pos if t > 0 else neg
    if p < 0.01:
        return f"strong {direction} bias (p={p:.4f})"
    if p < 0.05:
        return f"{direction} bias (p={p:.4f})"
    return f"no significant bias (p={p:.4f})"


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


# Aggregation

def compute(records: list[Record]) -> dict:
    """Compute bias metrics over a flat list of records (any mix of approaches)."""
    with_coords = [r for r in records if r.has_coords]

    lat_errors = [r.lat_error for r in with_coords]
    lng_errors = [r.lng_error for r in with_coords]
    haversines = [r.haversine_km for r in with_coords]

    bearings: list[float] = []
    quadrants: Counter = Counter()
    for r in with_coords:
        bearings.append(_bearing_deg(r.truth_lat, r.truth_lng, r.pred_lat, r.pred_lng))
        quadrants[_quadrant(r.lat_error, r.lng_error)] += 1

    t_lat, p_lat = _t_test_one_sample(lat_errors)
    t_lng, p_lng = _t_test_one_sample(lng_errors)

    n_correct = sum(1 for r in records if countries_match(r.pred_country, r.truth_country_code))

    return {
        "n_total": len(records),
        "n_with_coords": len(with_coords),
        "country_accuracy": (n_correct / len(records)) if records else 0.0,
        "haversine_km": _stats(haversines),
        "lat_error_deg": _stats(lat_errors),
        "lng_error_deg": _stats(lng_errors),
        "north_bias_test": {
            "t_statistic": t_lat, "p_value": p_lat,
            "interpretation": _interpret_bias(t_lat, p_lat, "north", "south"),
        },
        "east_bias_test": {
            "t_statistic": t_lng, "p_value": p_lng,
            "interpretation": _interpret_bias(t_lng, p_lng, "east", "west"),
        },
        "quadrants": dict(quadrants),
        "per_country": _per_country_confusion(records),
        "_lat_errors": lat_errors,
        "_lng_errors": lng_errors,
        "_haversines": haversines,
        "_bearings": bearings,
        "_truth_pts": [(r.truth_lat, r.truth_lng) for r in with_coords],
        "_pred_pts": [(r.pred_lat, r.pred_lng) for r in with_coords],
    }


def _per_country_confusion(records: list[Record]) -> dict:
    """Per-country TP / FP / FN aggregated over the given records.

    TP: truth = country and prediction correct.
    FN: truth = country but prediction wrong (a miss for that country).
    FP: predicted country but truth was something else (over-prediction).

    Everything is keyed by ISO alpha-3 so the truth side (TP/FN) and the
    prediction side (FP) for the *same* country net against each other, which
    is what the over/under map needs. A readable name is kept for the legend.
    """
    tp: Counter = Counter()
    fn: Counter = Counter()
    fp: Counter = Counter()
    name_for: dict[str, str] = {}

    for r in records:
        truth_iso = _iso_a3_from_alpha2(r.truth_country_code)
        correct = countries_match(r.pred_country, r.truth_country_code)
        if truth_iso:
            name_for.setdefault(truth_iso, (r.truth_country_code or "").lower())
            if correct:
                tp[truth_iso] += 1
            else:
                fn[truth_iso] += 1
        if r.pred_country and not correct:
            pred_iso = _iso_a3_for_name(r.pred_country)
            if pred_iso:
                name_for.setdefault(pred_iso, r.pred_country)
                fp[pred_iso] += 1

    out: dict[str, dict] = {}
    for iso in set(tp) | set(fn) | set(fp):
        out[iso] = {
            "tp": tp.get(iso, 0),
            "fn": fn.get(iso, 0),
            "fp": fp.get(iso, 0),
            "iso_a3": iso,
            "name": name_for.get(iso, iso),
        }
    return out


def _clustered_bias_tests(by_approach: dict[str, list[Record]]) -> dict:
    """Bias tests on per-image mean errors (one independent unit per image).

    The pooled record set repeats every image once per approach, so a t-test on
    all 3500 rows treats correlated measurements as independent and understates
    the standard error. Averaging each image's signed error across the
    approaches that predicted it yields one independent observation per image;
    the t-test on those means (n = number of distinct images) respects the
    repeated-measures structure and is the honest combined test.
    """
    lat_by_img: dict[str, list[float]] = {}
    lng_by_img: dict[str, list[float]] = {}
    for recs in by_approach.values():
        for r in recs:
            if not r.has_coords:
                continue
            lat_by_img.setdefault(r.image_id, []).append(r.lat_error)
            lng_by_img.setdefault(r.image_id, []).append(r.lng_error)

    lat_means = [float(np.mean(v)) for v in lat_by_img.values()]
    lng_means = [float(np.mean(v)) for v in lng_by_img.values()]

    t_lat, p_lat = _t_test_one_sample(lat_means)
    t_lng, p_lng = _t_test_one_sample(lng_means)
    return {
        "n_images": len(lat_means),
        "n_approaches": len(by_approach),
        "lat_error_deg": _stats(lat_means),
        "lng_error_deg": _stats(lng_means),
        "north_bias_test": {
            "t_statistic": t_lat, "p_value": p_lat,
            "interpretation": _interpret_bias(t_lat, p_lat, "north", "south"),
        },
        "east_bias_test": {
            "t_statistic": t_lng, "p_value": p_lng,
            "interpretation": _interpret_bias(t_lng, p_lng, "east", "west"),
        },
    }


def compute_all(by_approach: dict[str, list[Record]]) -> dict:
    all_records = [r for recs in by_approach.values() for r in recs]
    combined = compute(all_records)
    combined["clustered_bias_test"] = _clustered_bias_tests(by_approach)
    return {
        "combined": combined,
        "per_approach": {name: compute(recs) for name, recs in by_approach.items()},
    }


# Plots

def plot_bearing_rose(metrics: dict, out_path, title: str = "Error bearing rose (truth to prediction)") -> None:
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
    ax.bar(theta, counts, width=width, edgecolor="black", alpha=0.85, color=_PRIMARY)
    ax.set_title(title, pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_bearing_rose_by_approach(per_approach: dict, out_path) -> None:
    setup_plot_style()
    names = [n for n in per_approach if per_approach[n].get("_bearings")]
    if not names:
        return
    cols = 3
    rows = math.ceil(len(names) / cols)
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    n_bins = 16
    bins = np.linspace(0, 360, n_bins + 1)
    theta = np.deg2rad(bins[:-1] + (360 / n_bins) / 2)
    width = 2 * np.pi / n_bins
    for i, name in enumerate(names, start=1):
        bearings = per_approach[name]["_bearings"]
        counts, _ = np.histogram(bearings, bins=bins)
        ax = fig.add_subplot(rows, cols, i, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.bar(theta, counts, width=width, edgecolor="black", alpha=0.85, color=_PRIMARY)
        ax.set_title(name, fontsize=9, pad=12)
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["N", "", "E", "", "S", "", "W", ""])
        ax.set_yticklabels([])
    fig.suptitle("Error bearing rose per approach (truth to prediction)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_error_distribution(metrics: dict, out_path) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label, unit in [
        (axes[0], "_lat_errors", "Latitude error (pred - truth)", "deg"),
        (axes[1], "_lng_errors", "Longitude error (pred - truth)", "deg"),
        (axes[2], "_haversines", "Haversine distance", "km"),
    ]:
        data = metrics.get(key, [])
        if data:
            ax.hist(data, bins=40, edgecolor="black", alpha=0.75, color=_PRIMARY)
            mean = float(np.mean(data))
            ax.axvline(mean, color=_ERROR, linestyle="--", label=f"mean={mean:.1f}")
            if key != "_haversines":
                ax.axvline(0, color="black", linestyle=":")
            ax.legend()
        ax.set_title(label)
        ax.set_xlabel(unit)
        ax.set_ylabel("count")
    fig.suptitle("Pooled error distribution across all approaches", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_quadrant_bars(per_approach: dict, combined: dict, out_path) -> None:
    setup_plot_style()
    quads = ["NE", "NW", "SE", "SW"]
    names = list(per_approach.keys()) + ["COMBINED"]
    series = list(per_approach.values()) + [combined]

    x = np.arange(len(quads))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, m) in enumerate(zip(names, series)):
        q = m.get("quadrants", {})
        total = sum(q.values()) or 1
        shares = [q.get(k, 0) / total for k in quads]
        ax.bar(x + i * width, shares, width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(quads)
    ax.set_ylabel("share of predictions")
    ax.set_title("Error quadrant share (N/S x E/W) per approach")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_error_map(metrics: dict, out_path, max_arrows: int = 400) -> None:
    """Pooled truth->pred error vectors on a world map (mean vector highlighted)."""
    setup_plot_style()
    truth = metrics.get("_truth_pts", [])
    pred = metrics.get("_pred_pts", [])
    if not truth:
        return

    lat_err = metrics.get("_lat_errors", [])
    lng_err = metrics.get("_lng_errors", [])
    mean_lat = float(np.mean(lat_err)) if lat_err else 0.0
    mean_lng = float(np.mean(lng_err)) if lng_err else 0.0

    # Subsample arrows for legibility.
    idx = np.linspace(0, len(truth) - 1, min(max_arrows, len(truth))).astype(int)

    fig, ax = plt.subplots(figsize=(14, 7))
    world = _load_world()
    if world is not None:
        try:
            world.boundary.plot(ax=ax, color="#999999", linewidth=0.4)
        except Exception:
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
    else:
        # Plain lat/lng grid fallback.
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

    for i in idx:
        t_lat, t_lng = truth[i]
        p_lat, p_lng = pred[i]
        # skip antimeridian wrap-arounds for cleaner drawing
        if abs(p_lng - t_lng) > 180:
            continue
        ax.annotate(
            "", xy=(p_lng, p_lat), xytext=(t_lng, t_lat),
            arrowprops=dict(arrowstyle="->", color=_PRIMARY, alpha=0.25, lw=0.6),
        )
    ax.scatter([t[1] for t in [truth[i] for i in idx]],
               [t[0] for t in [truth[i] for i in idx]],
               s=4, color="black", alpha=0.4, zorder=3, label="ground truth")

    # Mean error vector (drawn from the world centroid for reference).
    ax.annotate(
        "", xy=(mean_lng * 6, mean_lat * 6), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=_ERROR, lw=3),
    )
    ax.text(0, 0, f"  mean err = ({mean_lat:+.1f} lat, {mean_lng:+.1f} lng)",
            color=_ERROR, fontsize=10, fontweight="bold")

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Pooled truth to prediction error vectors (mean error x6, red)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# Over / under-prediction choropleth (aggregated across approaches)

def _error_bias_rows(metrics: dict) -> list[dict]:
    """Rows with net-error volume per country, ISO-tagged, error>0 only.

    net = FP - FN. Positive = over-predicted (the pooled council names this
    country more often than it should); negative = under-predicted (misses).
    error_bias = sign(net) * log1p(|net|), matching the per-approach map.
    """
    rows = []
    for name, blk in (metrics.get("per_country") or {}).items():
        iso = blk.get("iso_a3")
        if not iso:
            continue
        fp = blk.get("fp", 0) or 0
        fn = blk.get("fn", 0) or 0
        if (fp + fn) <= 0:
            continue
        net = fp - fn
        rows.append({
            "name": name, "iso": iso, "fp": fp, "fn": fn, "tp": blk.get("tp", 0),
            "net": net,
            "error_bias": math.copysign(math.log1p(abs(net)), net) if net else 0.0,
        })
    return rows


def plot_over_under_map(metrics: dict, out_path, scope_label: str = "across all approaches") -> None:
    """Choropleth of over/under-prediction for the given records.

    ``scope_label`` describes the pool the metrics come from (e.g. "across all
    approaches" for the combined map, or a single approach's name). Red = net
    over-predicted (FP-dominant), blue = net under-predicted (FN-dominant),
    log-scaled magnitude. Falls back to a diverging bar chart when no Natural
    Earth shapefile / geopandas is available.
    """
    setup_plot_style()
    rows = _error_bias_rows(metrics)
    if not rows:
        return

    world = _load_world()
    if world is None:
        _plot_over_under_fallback(rows, out_path, scope_label=scope_label)
        return
    iso_col = _world_iso_col(world)
    if iso_col is None:
        _plot_over_under_fallback(rows, out_path, scope_label=scope_label)
        return

    from matplotlib.colors import TwoSlopeNorm
    max_abs = max((abs(r["error_bias"]) for r in rows), default=1.0) or 1.0
    vlim = max_abs * 1.05
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    cmap = plt.get_cmap("RdBu_r")
    iso_to_row = {r["iso"]: r for r in rows}

    try:
        world = world.to_crs("ESRI:54030")  # Robinson
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(14, 7.5))
    world.plot(ax=ax, color="#EEEEEE", edgecolor="#BBBBBB", linewidth=0.25)
    for _, geo_row in world.iterrows():
        r = iso_to_row.get(geo_row.get(iso_col))
        if not r:
            continue
        sub = world[world[iso_col] == geo_row.get(iso_col)]
        sub.plot(ax=ax, color=[cmap(norm(r["error_bias"]))],
                 edgecolor="#777777", linewidth=0.4)
    ax.set_axis_off()
    ax.set_title(
        f"Over/under-prediction — {scope_label}\n"
        "red = over-predicted (net false-positive)   blue = under-predicted (net false-negative)",
        fontsize=11,
    )

    import matplotlib.cm as cm
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.5, pad=0.02)
    cbar.set_label("sign(FP - FN) * log(1 + |FP - FN|)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_over_under_fallback(rows: list[dict], out_path, scope_label: str = "across all approaches") -> None:
    """Diverging horizontal bar of net error volume (no shapefile available)."""
    setup_plot_style()
    rows = sorted(rows, key=lambda r: r["net"])
    # Trim to the most extreme on each side to keep the figure readable.
    top = 25
    if len(rows) > 2 * top:
        rows = rows[:top] + rows[-top:]
    names = [(r["name"] or r["iso"]).title() for r in rows]
    nets = [r["net"] for r in rows]
    colors = ["#2166AC" if n < 0 else "#B2182B" for n in nets]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(rows))))
    ax.barh(range(len(rows)), nets, color=colors, edgecolor="#444444", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("net error volume (FP - FN):  <0 under-predicted,  >0 over-predicted")
    ax.set_title(f"Over/under-prediction per country — {scope_label}  (bar fallback, no shapefile)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
