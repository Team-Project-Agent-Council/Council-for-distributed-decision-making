"""HTML report renderer.

Reads the same JSON outputs as ``eval/report.py`` (geo, agents, funnel, judge,
heatmap, calibration) and renders a single-file
``report.html`` from the Jinja2 template at ``eval/templates/report.html.j2``.

Designed to be opened directly in a browser, no build step. JS is vanilla
(theme toggle, collapse, sortable tables, scroll-spy). The interactive Folium
map (when present) is embedded as an iframe.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eval_tourn.loader import AGENT_NAMES


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _safe_float(x) -> float | None:
    """Coerce to float, treating Jinja Undefined / None / non-numeric as None."""
    if x is None:
        return None
    try:
        # Trigger any Undefined fault here so it can be caught.
        return float(x)
    except Exception:
        return None


def _pct(x) -> str:
    f = _safe_float(x)
    if f is None:
        return "n/a"
    return f"{f:.1%}"


def _pct_ci(block) -> str:
    if not isinstance(block, dict):
        return "n/a"
    n = _safe_float(block.get("n", 0))
    if not n:
        return "n/a"
    rate = _safe_float(block.get("rate"))
    lo = _safe_float(block.get("ci_low"))
    hi = _safe_float(block.get("ci_high"))
    if rate is None or lo is None or hi is None:
        return "n/a"
    return f"{rate:.1%} [{lo:.1%}, {hi:.1%}]"


def _km(x) -> str:
    f = _safe_float(x)
    if f is None:
        return "n/a"
    return f"{f:,.0f} km"


def _stat_or_dash(per_agent_block, key: str) -> str:
    """Format mean ± stdev for an aggregated stats block, or ', ' if empty."""
    if not isinstance(per_agent_block, dict):
        return ", "
    st = per_agent_block.get(key) or {}
    if not isinstance(st, dict) or not st.get("n"):
        return ", "
    mean = _safe_float(st.get("mean"))
    stdev = _safe_float(st.get("stdev"))
    if mean is None:
        return ", "
    if stdev is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {stdev:.2f}"


def render(out_dir: Path) -> Path | None:
    """Build report.html from the JSONs already on disk under out_dir.

    Returns the output path on success, ``None`` if Jinja2 is not installed
    (caller should warn but continue, markdown report is still valid).
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
    calibration = _load_json(out_dir / "calibration_metrics.json")
    dynamics = _load_json(out_dir / "dynamics_metrics.json")


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
        "run_metadata": {
            "n_images": (geo or {}).get("n_total", 0),
        },
        "geo": geo,
        "agents": agents,
        "judge": judge,
        "heatmap": heatmap,
        "calibration": calibration,
        "dynamics": dynamics,
        "agent_names": list(AGENT_NAMES),
        "pct": _pct,
        "pct_ci": _pct_ci,
        "km": _km,
        "stat_or_dash": _stat_or_dash,
    }

    html = tmpl.render(**payload)
    out_file = out_dir / "report.html"
    out_file.write_text(html)
    print(f"[render_html] wrote {out_file}")
    return out_file
