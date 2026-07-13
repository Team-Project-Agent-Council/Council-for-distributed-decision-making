"""HTML report renderer for eval_reguess.

Reads geo/agents/judge JSONs and renders eval_reguess/templates/report.html.j2.
Gracefully returns None if jinja2 is not installed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eval_reguess.loader import AGENT_NAMES


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


_WORLD_MAP_SPECS = [
    ("world_map_accuracy.png", "Per-country true-positive rate (green) with false-positive outlines (red)."),
    ("world_map_f1.png", "Per-country F1, divergent around the run's macro-F1. Green = above average, red = below."),
    ("world_map_error_bias.png", "Per-country error bias (FP-FN)/(FP+FN). Red = over-predicted, blue = missed."),
]


def _world_maps_present(out_dir: Path) -> list[dict]:
    """Return [{file, caption}] for world-map PNGs that exist in plots/."""
    plots = out_dir / "plots"
    return [
        {"file": f"plots/{fname}", "caption": cap}
        for fname, cap in _WORLD_MAP_SPECS
        if (plots / fname).exists()
    ]


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _pct(x) -> str:
    f = _safe_float(x)
    if f is None:
        return "n/a"
    return f"{f:.1%}"


def _km(x) -> str:
    f = _safe_float(x)
    if f is None:
        return "n/a"
    return f"{f:,.0f} km"


def _fmt_f(x, decimals: int = 2) -> str:
    f = _safe_float(x)
    if f is None:
        return ", "
    return f"{f:.{decimals}f}"


def _stat_mean(block) -> str:
    if not isinstance(block, dict) or not block.get("n"):
        return ", "
    mean = _safe_float(block.get("mean"))
    if mean is None:
        return ", "
    return f"{mean:.2f}"


def render(out_dir: Path) -> Path | None:
    """Render report.html from JSONs under out_dir.

    Returns output path on success, None if jinja2 is not installed.
    """
    try:
        import jinja2
    except ImportError:
        print("[render_html] jinja2 not installed; skipping HTML report")
        return None

    geo = _load_json(out_dir / "geo_metrics.json")
    agents = _load_json(out_dir / "agent_metrics.json")
    judge = _load_json(out_dir / "judge_summary.json")
    heatmap = _load_json(out_dir / "heatmap_metrics.json")
    dynamics = _load_json(out_dir / "dynamics_metrics.json")
    world_maps = _world_maps_present(out_dir)
    template_dir = Path(__file__).resolve().parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=jinja2.select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.ChainableUndefined,
    )
    tmpl = env.get_template("report.html.j2")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_images": (geo or {}).get("n_total", 0),
        "geo": geo,
        "agents": agents,
        "judge": judge,
        "heatmap": heatmap,
        "dynamics": dynamics,
        "world_maps": world_maps,
        "agent_names": list(AGENT_NAMES),
        "pct": _pct,
        "km": _km,
        "fmt_f": _fmt_f,
        "stat_mean": _stat_mean,
    }

    html = tmpl.render(**payload)
    out_file = out_dir / "report.html"
    out_file.write_text(html)
    print(f"[render_html] wrote {out_file}")
    return out_file
