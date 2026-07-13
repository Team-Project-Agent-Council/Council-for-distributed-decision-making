"""Geographic heatmap: per-country TPR + FP, static world maps.

Outputs:
  - ``plots/world_map_accuracy.png``, static, GeoPandas + matplotlib (Robinson)

Static plot requires a Natural Earth shapefile reachable at one of:
  1. ``$VLM_NATURAL_EARTH_PATH`` (env override)
  2. ``data/natural_earth/ne_110m_admin_0_countries.shp`` under the project root
If neither exists, falls back to a horizontal-bar plot showing TPR per country
(no map, but still a usable figure).
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from eval_pnt._style import STYLE, add_figure_caption, figsize_wide, setup_plot_style
from eval_pnt.loader import RunRecord, load_run


def _iso_a3_for_name(name: str) -> str | None:
    """Best-effort country-name → ISO alpha-3 via pycountry. Returns None on miss."""
    if not name:
        return None
    try:
        import pycountry
    except ImportError:
        return None
    n = name.strip()
    # Try direct lookups first
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
    except LookupError:
        return None
    except Exception:
        return None
    return None


def _iso_a3_from_alpha2(code: str) -> str | None:
    if not code:
        return None
    try:
        import pycountry
    except ImportError:
        return None
    try:
        c = pycountry.countries.get(alpha_2=code.upper())
        return c.alpha_3 if c else None
    except Exception:
        return None


def compute_heatmap(records: list[RunRecord]) -> dict:
    """Per-country TPR + FP across the run."""
    truth_total: dict[str, int] = defaultdict(int)
    truth_correct: dict[str, int] = defaultdict(int)
    pred_total: dict[str, int] = defaultdict(int)
    pred_false_pos: dict[str, int] = defaultdict(int)
    iso_for: dict[str, str | None] = {}
    images_correct: dict[str, list[str]] = defaultdict(list)
    images_missed: dict[str, list[str]] = defaultdict(list)
    images_fp: dict[str, list[str]] = defaultdict(list)

    n_total = len(records)

    for r in records:
        truth_name = r.truth_country_name or r.truth_country_code
        pred_name = r.pred_country
        truth_total[truth_name] += 1
        if truth_name not in iso_for:
            iso_for[truth_name] = (
                _iso_a3_from_alpha2(r.truth_country_code)
                or _iso_a3_for_name(truth_name)
            )
        if r.is_correct:
            truth_correct[truth_name] += 1
            images_correct[truth_name].append(r.image_id)
        else:
            images_missed[truth_name].append(r.image_id)

        if pred_name:
            pred_total[pred_name] += 1
            if pred_name not in iso_for:
                iso_for[pred_name] = _iso_a3_for_name(pred_name)
            if not r.is_correct:
                pred_false_pos[pred_name] += 1
                images_fp[pred_name].append(r.image_id)

    countries = set(truth_total) | set(pred_total)
    per_country: dict[str, dict] = {}
    for c in sorted(countries):
        n_truth = truth_total.get(c, 0)
        n_correct = truth_correct.get(c, 0)
        n_pred = pred_total.get(c, 0)
        n_fp = pred_false_pos.get(c, 0)
        tpr = (n_correct / n_truth) if n_truth else None
        ppv = ((n_pred - n_fp) / n_pred) if n_pred else None
        per_country[c] = {
            "n_truth": n_truth,
            "n_correct": n_correct,
            "n_predicted": n_pred,
            "n_false_positive": n_fp,
            "tpr": tpr,
            "ppv": ppv,
            "iso_a3": iso_for.get(c),
            "image_ids_correct": images_correct.get(c, [])[:10],
            "image_ids_missed": images_missed.get(c, [])[:10],
            "image_ids_false_positive": images_fp.get(c, [])[:10],
        }

    truth_countries = [c for c, n in truth_total.items() if n > 0]
    macro_avg_tpr = (
        sum(per_country[c]["tpr"] for c in truth_countries if per_country[c]["tpr"] is not None)
        / max(1, len(truth_countries))
    )

    return {
        "n_total_images": n_total,
        "n_countries_with_truth": len(truth_countries),
        "n_countries_predicted": sum(1 for c in pred_total if pred_total[c] > 0),
        "macro_avg_tpr": macro_avg_tpr,
        "per_country": per_country,
    }


# Static plot

def _natural_earth_path() -> Path | None:
    env = os.environ.get("VLM_NATURAL_EARTH_PATH")
    if env and Path(env).exists():
        return Path(env)
    _ne = Path("data") / "natural_earth" / "ne_110m_admin_0_countries.shp"
    candidates = [
        # approach-local data/ (legacy location)
        Path(__file__).resolve().parent.parent / _ne,
        # shared assets moved under results-overview/data/ (repo root is 3 up)
        Path(__file__).resolve().parent.parent.parent / "results-overview" / _ne,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _green_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "tpr_green", ["#FFFFFF", "#C7E9C0", "#74C476", STYLE.success, "#005A32"]
    )


def plot_world_map(metrics: dict, out_path: Path, n_caption: int | None = None) -> bool:
    """Try a real GeoPandas world map; fall back to a TPR bar plot if NE missing.

    Returns True if a map (not the fallback) was produced.
    """
    setup_plot_style()
    ne_path = _natural_earth_path()
    if ne_path is None:
        _plot_world_map_fallback(metrics, out_path, n_caption=n_caption)
        return False

    try:
        import geopandas as gpd
    except ImportError:
        _plot_world_map_fallback(metrics, out_path, n_caption=n_caption)
        return False

    try:
        world = gpd.read_file(str(ne_path))
    except Exception:
        _plot_world_map_fallback(metrics, out_path, n_caption=n_caption)
        return False

    iso_col = None
    for c in ("ISO_A3", "ADM0_A3", "iso_a3"):
        if c in world.columns:
            iso_col = c
            break
    if iso_col is None:
        _plot_world_map_fallback(metrics, out_path, n_caption=n_caption)
        return False

    per_country = metrics.get("per_country", {})
    iso_to_tpr: dict[str, float] = {}
    iso_to_fp: dict[str, int] = {}
    for c, blk in per_country.items():
        iso = blk.get("iso_a3")
        if not iso:
            continue
        if blk.get("tpr") is not None:
            iso_to_tpr[iso] = blk["tpr"]
        iso_to_fp[iso] = blk.get("n_false_positive", 0)

    world["_tpr"] = world[iso_col].map(iso_to_tpr)
    world["_fp"] = world[iso_col].map(iso_to_fp).fillna(0)

    # Reproject to Robinson if available; otherwise leave as-is.
    try:
        world = world.to_crs("ESRI:54030")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(13, 7))
    cmap = _green_cmap()
    world.plot(
        column="_tpr", cmap=cmap, vmin=0.0, vmax=1.0, ax=ax,
        edgecolor="#888888", linewidth=0.3,
        missing_kwds={"color": "#EEEEEE", "edgecolor": "#BBBBBB", "linewidth": 0.2},
        legend=True, legend_kwds={
            "label": "True-positive rate (correct ÷ truth = country)",
            "orientation": "horizontal", "shrink": 0.6, "pad": 0.02,
        },
    )
    # Outline FP-heavy countries in red; intensity ~ FP count
    max_fp = max(iso_to_fp.values()) if iso_to_fp else 0
    if max_fp > 0:
        for _, row in world.iterrows():
            fp = row.get("_fp", 0) or 0
            if fp <= 0:
                continue
            lw = 0.3 + min(2.5, fp / max_fp * 2.5)
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            x, y = geom.exterior.xy if geom.geom_type == "Polygon" else (None, None)
            ax.add_patch(plt.Polygon(
                list(zip(x, y)) if x is not None else [],
                fill=False, edgecolor=STYLE.error, linewidth=lw,
            )) if x is not None else None
    ax.set_axis_off()
    ax.set_title(
        "Geographic accuracy heatmap, green = TPR, red outline = false-positive count"
    )
    if n_caption is not None:
        add_figure_caption(
            ax,
            "Per-country true-positive rate (TPR) of the council's prediction. "
            "Countries shaded gray were never the truth. Red outlines on countries "
            "frequently predicted incorrectly.",
            n=n_caption,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_world_map_fallback(metrics: dict, out_path: Path, n_caption: int | None = None) -> None:
    """No shapefile available, show a horizontal bar of TPR per country with truth data."""
    setup_plot_style()
    per_country = metrics.get("per_country", {})
    rows = [
        (c, blk["tpr"], blk["n_truth"], blk["n_false_positive"])
        for c, blk in per_country.items()
        if blk.get("n_truth", 0) > 0 and blk.get("tpr") is not None
    ]
    rows.sort(key=lambda x: x[1])  # ascending so worst at the top
    if not rows:
        # Empty, still produce a placeholder so the report doesn't 404 on the img.
        fig, ax = plt.subplots(figsize=figsize_wide())
        ax.text(0.5, 0.5, "no per-country truth data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path)
        plt.close(fig)
        return

    countries = [r[0] for r in rows]
    tprs = [r[1] for r in rows]
    n_truth = [r[2] for r in rows]
    fps = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(rows))))
    cmap = _green_cmap()
    colors = [cmap(t) for t in tprs]
    bars = ax.barh(range(len(countries)), tprs, color=colors, edgecolor=STYLE.neutral)
    for i, (n, fp) in enumerate(zip(n_truth, fps)):
        ax.text(min(tprs[i] + 0.02, 1.02), i,
                f"n={n}, fp={fp}", va="center", fontsize=8, color="#333333")
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels([c.title() for c in countries])
    ax.set_xlabel("True-positive rate (correct ÷ truth = country)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-country TPR  (no Natural Earth shapefile, bar fallback)")
    if n_caption is not None:
        add_figure_caption(
            ax,
            "Per-country true-positive rate. Set $VLM_NATURAL_EARTH_PATH or place "
            "ne_110m_admin_0_countries.shp under data/natural_earth/ for a real map.",
            n=n_caption,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Points map: correct truth-locations + wrong pred-locations

def _load_world_for_basemap():
    ne_path = _natural_earth_path()
    if ne_path is None:
        return None
    try:
        import geopandas as gpd
        return gpd.read_file(str(ne_path))
    except Exception:
        return None


def _per_country_stats(metrics: dict) -> list[dict]:
    """Compute TP/FP/FN + precision/recall/F1 + n_truth/n_predicted per country."""
    out = []
    for c, blk in (metrics.get("per_country") or {}).items():
        iso = blk.get("iso_a3")
        if not iso:
            continue
        tp = blk.get("n_correct", 0) or 0
        fn = (blk.get("n_truth", 0) or 0) - tp
        fp = blk.get("n_false_positive", 0) or 0
        n_truth = tp + fn
        n_pred = tp + fp
        if not (n_truth or n_pred):
            continue
        precision = tp / n_pred if n_pred else None
        recall = tp / n_truth if n_truth else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else 0.0)
        out.append({
            "name": c, "iso": iso,
            "tp": tp, "fp": fp, "fn": fn,
            "n_truth": n_truth, "n_pred": n_pred,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return out


def _render_choropleth_with_legend(
    rows: list[dict],
    out_path: Path,
    *,
    color_value_key: str,
    color_vmin: float,
    color_vmax: float,
    color_vcenter: float,
    cmap_name: str,
    alpha_value_fn,            # row -> float in [0,1]; smaller = more washed-out
    sort_key,                  # row -> sortable; descending = best first
    legend_label_fn,           # row -> str (right side of swatch)
    title: str,
    caption: str,
    n_caption: int | None,
    countries_filter_fn=None,  # row -> bool; only those drawn
    grey_others: bool = True,
    alpha_floor: float = 0.35,  # bottom of (alpha_floor, 1.0) range; raise for more visibility
    alpha_span: float = 0.65,
    legend_mode: str = "best_worst",   # "best_worst" | "divergent_split"
    legend_split_predicate=None,        # row -> bool; True → left ("over"), False → right ("under")
    legend_left_label: str = "Best",
    legend_right_label: str = "Worst",
    legend_left_color: str | None = None,
    legend_right_color: str | None = None,
    legend_left_sort=None,
    legend_right_sort=None,
    edge_width_fn=None,        # row -> linewidth (default 0.5)
    alpha_curve: float = 0.5,  # exponent for alpha modulation (0.5 = sqrt, 1.0 = linear)
) -> bool:
    setup_plot_style()
    world = _load_world_for_basemap()
    if world is None:
        return False
    iso_col = None
    for c in ("ISO_A3", "ADM0_A3", "iso_a3"):
        if c in world.columns:
            iso_col = c
            break
    if iso_col is None:
        return False

    drawn = [r for r in rows if (countries_filter_fn is None or countries_filter_fn(r))]
    if not drawn:
        return False

    iso_to_row: dict[str, dict] = {r["iso"]: r for r in drawn}

    from matplotlib.colors import TwoSlopeNorm
    cmap = plt.get_cmap(cmap_name)
    if color_vmin < color_vcenter < color_vmax:
        norm = TwoSlopeNorm(vmin=color_vmin, vcenter=color_vcenter, vmax=color_vmax)
    else:
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=color_vmin, vmax=color_vmax)

    def _color_for(row: dict) -> tuple:
        v = row[color_value_key]
        rgba = cmap(norm(v))
        a = max(0.0, min(1.0, alpha_value_fn(row)))
        a = a ** alpha_curve
        return (rgba[0], rgba[1], rgba[2], alpha_floor + alpha_span * a)

    display_world = world
    try:
        display_world = world.to_crs("ESRI:54030")
    except Exception:
        pass

    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.4], wspace=0.05)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])

    if grey_others:
        display_world.plot(
            ax=ax_map, color="#EEEEEE",
            edgecolor="#BBBBBB", linewidth=0.25,
        )

    for _, row in display_world.iterrows():
        iso = row.get(iso_col)
        r = iso_to_row.get(iso)
        if not r:
            continue
        sub = display_world[display_world[iso_col] == iso]
        lw = edge_width_fn(r) if edge_width_fn is not None else 0.5
        sub.plot(
            ax=ax_map,
            color=[_color_for(r)],
            edgecolor="#777777", linewidth=lw,
        )
    ax_map.set_axis_off()
    ax_map.set_title(title, fontsize=11)

    # Two-column ranked legend
    if legend_mode == "divergent_split" and legend_split_predicate is not None:
        left_pool = [r for r in drawn if legend_split_predicate(r)]
        right_pool = [r for r in drawn if not legend_split_predicate(r)]
        left_items = sorted(left_pool, key=legend_left_sort or sort_key)
        right_items = sorted(right_pool, key=legend_right_sort or sort_key)
        n = len(left_items) + len(right_items)
        left_hdr_color = legend_left_color or STYLE.error
        right_hdr_color = legend_right_color or STYLE.success
    else:
        rows_sorted = sorted(drawn, key=sort_key)
        n = len(rows_sorted)
        half = (n + 1) // 2
        left_items = rows_sorted[:half]
        right_items = list(reversed(rows_sorted[half:]))
        left_hdr_color = legend_left_color or STYLE.success
        right_hdr_color = legend_right_color or STYLE.error

    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.set_axis_off()
    ax_legend.set_title(f"Per-country  (countries shown = {n})",
                        fontsize=10, loc="left")
    ax_legend.text(0.02, 0.965, legend_left_label, fontsize=9, fontweight="bold",
                   color=left_hdr_color)
    ax_legend.text(0.52, 0.965, legend_right_label, fontsize=9, fontweight="bold",
                   color=right_hdr_color)

    def _draw_col(items, x0):
        if not items:
            return
        top = 0.93
        bottom = 0.02
        step = (top - bottom) / max(1, len(items))
        font = max(6.0, min(9, step * 130))
        swatch_w = 0.05
        for i, r in enumerate(items):
            y = top - (i + 0.5) * step
            ax_legend.add_patch(plt.Rectangle(
                (x0, y - step * 0.35), swatch_w, step * 0.7,
                facecolor=_color_for(r), edgecolor="#444444", linewidth=0.4,
            ))
            label_name = (r["name"] or r["iso"]).title()
            if len(label_name) > 16:
                label_name = label_name[:15] + "…"
            ax_legend.text(
                x0 + swatch_w + 0.012, y,
                f"{label_name}  {legend_label_fn(r)}",
                fontsize=font, va="center",
            )

    _draw_col(left_items, 0.02)
    _draw_col(right_items, 0.52)

    if n_caption is not None:
        add_figure_caption(ax_map, caption, n=n_caption)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_world_map_f1(metrics: dict, out_path: Path,
                      n_caption: int | None = None) -> bool:
    """F1-score choropleth, divergent around the macro-F1 of the run.
    Alpha scales with √(TP+FP+FN), the per-country evidence."""
    rows = _per_country_stats(metrics)
    if not rows:
        return False
    macro_f1 = sum(r["f1"] for r in rows) / len(rows)
    macro_f1 = max(0.05, min(0.95, macro_f1))  # keep TwoSlopeNorm valid
    max_evidence = max((r["tp"] + r["fp"] + r["fn"]) for r in rows) or 1

    def _alpha(r):
        ev = r["tp"] + r["fp"] + r["fn"]
        return (ev / max_evidence) ** 0.5

    def _label(r):
        p = f"{r['precision']:.0%}" if r["precision"] is not None else ", "
        rec = f"{r['recall']:.0%}" if r["recall"] is not None else ", "
        return f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  F1={r['f1']:.2f}"

    return _render_choropleth_with_legend(
        rows, out_path,
        color_value_key="f1",
        color_vmin=0.0, color_vmax=1.0, color_vcenter=macro_f1,
        cmap_name="RdYlGn",
        alpha_value_fn=_alpha,
        sort_key=lambda r: (-r["f1"], -(r["tp"] + r["fp"] + r["fn"]), r["name"]),
        legend_label_fn=_label,
        title="",
        caption=(
            "Per-country F1 = 2·P·R / (P+R). Color is divergent on a red→yellow→green "
            f"scale centered at the run's macro-F1 ({macro_f1:.2f}); above-average "
            "countries are green, below-average red. Alpha scales with √(TP+FP+FN) so "
            "low-evidence countries appear washed-out. All countries with any signal "
            "(truth or prediction) are shown."
        ),
        n_caption=n_caption,
    )


def plot_world_map_error_bias(metrics: dict, out_path: Path,
                              n_caption: int | None = None) -> bool:
    """Signed error-volume choropleth: sign(FP − FN) · log(1 + |FP − FN|).
    Unlike a normalised (FP−FN)/(FP+FN) map, this keeps the *magnitude* of the
    net error: a country with FP=50, FN=0 reads far darker than one with
    FP=1, FN=0. Red = net over-predicted (FP-dominant), blue = net
    under-predicted (FN-dominant). Only countries with at least one error are
    shaded. Legend is split: red column = over-predicted, blue column =
    under-predicted (sorted by net-error volume within each side)."""
    rows = _per_country_stats(metrics)
    rows = [r for r in rows if (r["fp"] + r["fn"]) > 0]
    if not rows:
        return False
    for r in rows:
        net = r["fp"] - r["fn"]
        r["error_bias"] = math.copysign(math.log1p(abs(net)), net) if net else 0.0
    # Symmetric colour scale around 0, spanning the largest signed magnitude.
    max_abs = max((abs(r["error_bias"]) for r in rows), default=0.0) or 1.0
    vlim = max_abs * 1.05

    def _label(r):
        sign = "+" if (r["fp"] - r["fn"]) >= 0 else ""
        return f"FP={r['fp']} FN={r['fn']}  net={sign}{r['fp'] - r['fn']}"

    # Split: net > 0 → over-predicted (FP-dominant, red column);
    #        net < 0 → under-predicted (FN-dominant, blue column);
    #        net == 0 (FP=FN tie) → arbitrarily right side.
    def _is_over(r):
        return (r["fp"] - r["fn"]) > 0

    # Within each column, rank by net-error volume descending.
    def _left_sort(r):
        return (-(r["fp"] - r["fn"]), r["name"])

    def _right_sort(r):
        return (-(r["fn"] - r["fp"]), r["name"])

    return _render_choropleth_with_legend(
        rows, out_path,
        color_value_key="error_bias",
        color_vmin=-vlim, color_vmax=vlim, color_vcenter=0.0,
        cmap_name="RdBu_r",  # classic ColorBrewer divergent, softer than bwr
        alpha_value_fn=lambda r: 1.0,  # magnitude lives in colour, not alpha
        sort_key=lambda r: (-abs(r["error_bias"]), r["name"]),
        legend_label_fn=_label,
        title=("Error bias = sign(FP − FN) · log(1 + |FP − FN|);  "
               "red = over-predicted (net FP),  blue = under-predicted (net FN)"),
        caption=(
            "Per-country signed net-error volume, log-scaled. Red = the system "
            "makes more false-positive than false-negative errors for this "
            "country (net over-prediction); blue = net misses. Colour intensity "
            "grows with the size of the net error, so countries with many "
            "systematic errors dominate visually. TP ignored; only countries "
            "with at least one error are drawn, the rest appear grey."
        ),
        n_caption=n_caption,
        alpha_floor=1.0,
        alpha_span=0.0,
        alpha_curve=1.0,
        legend_mode="divergent_split",
        legend_split_predicate=_is_over,
        legend_left_label="Over-predicted (net FP)",
        legend_right_label="Under-predicted (net FN)",
        legend_left_color="#B2182B",   # deep red
        legend_right_color="#2166AC",  # deep blue
        legend_left_sort=_left_sort,
        legend_right_sort=_right_sort,
    )



# Interactive Folium map

def run(results_dir: Path, gt_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    records = load_run(results_dir, gt_csv)
    metrics = compute_heatmap(records)

    plot_world_map(metrics, plots_dir / "world_map_accuracy.png")
    plot_world_map_f1(metrics, plots_dir / "world_map_f1.png")
    plot_world_map_error_bias(metrics, plots_dir / "world_map_error_bias.png")

    out_file = out_dir / "heatmap_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"[heatmap] wrote {out_file}  countries_with_truth={metrics['n_countries_with_truth']}  "
        f"macro_avg_tpr={metrics['macro_avg_tpr']:.1%}"
    )
    return metrics
