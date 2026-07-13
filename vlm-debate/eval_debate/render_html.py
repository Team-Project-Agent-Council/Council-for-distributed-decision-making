"""Render the Debate evaluation report as a self-contained HTML file."""

from __future__ import annotations

import json
from pathlib import Path


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


def render(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        print("[render_html] jinja2 not installed, skipping HTML render")
        return

    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=True)
    try:
        tmpl = env.get_template("report.html.j2")
    except Exception as e:
        print(f"[render_html] template error: {e}")
        return

    geo = _load_json(out_dir / "geo_metrics.json")
    agents = _load_json(out_dir / "agent_metrics.json")
    judge = _load_json(out_dir / "judge_summary.json")
    debate_stats = _load_json(out_dir / "debate_stats.json")
    heatmap = _load_json(out_dir / "heatmap_metrics.json")
    dynamics = _load_json(out_dir / "dynamics_metrics.json")
    world_maps = _world_maps_present(out_dir)

    # Read markdown report for raw text fallback
    md_path = out_dir / "report.md"
    md_text = md_path.read_text() if md_path.exists() else ""

    payload = {
        "geo": geo,
        "agents": agents,
        "judge": judge,
        "debate_stats": debate_stats,
        "heatmap": heatmap,
        "dynamics": dynamics,
        "world_maps": world_maps,
        "md_text": md_text,
        "plots_dir": "plots",
    }

    html = tmpl.render(**payload)
    out_file = out_dir / "report.html"
    out_file.write_text(html)
    print(f"[render_html] wrote {out_file}")
