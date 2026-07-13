"""Generate Plotly plots for every statistic from evaluate.py + analyze_rounds.py.

Reads the result.json files plus the ground-truth CSV, computes the same
statistics as the two text reports, and writes one PNG per statistic to
the output directory (default: debate_analysis_images/).

Usage:
    python -m vlm_council.plot_analysis results_debate_adversarial/ Images/georc_locations.csv
    python -m vlm_council.plot_analysis results_debate_adversarial/ Images/georc_locations.csv --out my_plots/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from vlm_council.evaluate import (
    _countries_match,
    _extract_coordinates,
    _extract_country,
    _haversine_km,
    _is_neighbor,
    _load_ground_truth,
    COUNTRY_CODE_TO_NAME,
)
from vlm_council.analyze_rounds import (
    AGENT_NAMES,
    _agent_correct,
    _classify_pairing,
    _last_position_in_pairing,
)


PIO_TEMPLATE = "plotly_white"
WIDTH = 1200
HEIGHT = 750
SCALE = 3  # ~3.6 megapixel, print-ready

# ── Paper style ──────────────────────────────────────────────────────────
# Okabe-Ito colour-blind-safe palette (Okabe & Ito 2008, "Color universal design")
PAPER_FONT_FAMILY = "Times New Roman, Georgia, serif"
PAPER_FONT_COLOR = "#222"
PAPER_FONT_SIZE = 14
PAPER_TITLE_SIZE = 19
PAPER_AXIS_TITLE_SIZE = 14
PAPER_TICK_SIZE = 12
PAPER_ANNO_SIZE = 11.5

# Semantic colours (Okabe-Ito derived)
COLOR_CORRECT  = "#009E73"   # bluish green
COLOR_WRONG    = "#D55E00"   # vermilion
COLOR_NEIGHBOR = "#E69F00"   # orange
COLOR_NEUTRAL  = "#999999"   # grey
COLOR_INFO     = "#0072B2"   # blue
COLOR_BAD      = "#CC79A7"   # reddish purple
COLOR_OK       = "#56B4E9"   # sky blue
COLOR_ACCENT   = "#F0E442"   # yellow

AGENT_COLORS = {
    "linguistic": "#0072B2",  # blue
    "landscape":  "#009E73",  # bluish green
    "botonics":   "#56B4E9",
    "botanics":   "#56B4E9",  # sky blue
    "regulatory": "#D55E00",  # vermilion
    "meta":       "#CC79A7",  # reddish purple
}


def set_paper_style(fig: go.Figure, *, subtitle: str | None = None) -> None:
    """Apply consistent paper style: serif font, sober gridlines, sample-size subtitle.

    Pass ``subtitle`` to inject a smaller second line under the main title
    (typically the sample size, e.g. ``"n = 500 images"``).
    """
    fig.update_layout(
        font=dict(family=PAPER_FONT_FAMILY, color=PAPER_FONT_COLOR, size=PAPER_FONT_SIZE),
        title=dict(
            x=0.5, xanchor="center",
            font=dict(family=PAPER_FONT_FAMILY, size=PAPER_TITLE_SIZE, color="#111"),
            pad=dict(b=14),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=110 if subtitle else 80, l=80, r=40, b=140),
    )
    fig.update_xaxes(
        title_font=dict(family=PAPER_FONT_FAMILY, size=PAPER_AXIS_TITLE_SIZE, color="#222"),
        tickfont=dict(family=PAPER_FONT_FAMILY, size=PAPER_TICK_SIZE, color="#333"),
        showgrid=False, zeroline=False, ticks="outside", ticklen=4,
        linecolor="#666", linewidth=1,
    )
    fig.update_yaxes(
        title_font=dict(family=PAPER_FONT_FAMILY, size=PAPER_AXIS_TITLE_SIZE, color="#222"),
        tickfont=dict(family=PAPER_FONT_FAMILY, size=PAPER_TICK_SIZE, color="#333"),
        showgrid=True, gridcolor="#E5E5E5", gridwidth=1,
        zeroline=False, ticks="outside", ticklen=4,
        linecolor="#666", linewidth=1,
    )
    if subtitle:
        fig.add_annotation(
            text=f"<i>{subtitle}</i>",
            xref="paper", yref="paper",
            x=0.5, y=1.04, xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(family=PAPER_FONT_FAMILY, size=12.5, color="#555"),
        )


# ── Data loading ─────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> list[dict]:
    """Load all result.json files. Skips errors."""
    results = []
    for img_dir in sorted(results_dir.iterdir()):
        result_file = img_dir / "result.json"
        if not result_file.exists():
            continue
        try:
            with open(result_file) as f:
                data = json.load(f)
            if data.get("error"):
                continue
            data["_name"] = img_dir.name
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


# ── Plot helpers ─────────────────────────────────────────────────────────

def save(fig: go.Figure, out_dir: Path, name: str) -> None:
    """Write a single plotly figure to PNG."""
    out_path = out_dir / f"{name}.png"
    fig.write_image(str(out_path), width=WIDTH, height=HEIGHT, scale=SCALE)
    print(f"  → {out_path.name}")


def annotate_below(fig: go.Figure, text: str, *, has_legend: bool = False) -> None:
    """Append a paper-style explanation block below the plot.

    When the figure also has a horizontal legend below the plot, pass
    ``has_legend=True`` so the annotation is pushed further down and the
    bottom margin is increased to make room for both.
    """
    y_anno = -0.30 if has_legend else -0.20
    bottom_margin = 240 if has_legend else 175
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0, y=y_anno, xanchor="left", yanchor="top",
        showarrow=False,
        align="left",
        font=dict(family=PAPER_FONT_FAMILY, size=PAPER_ANNO_SIZE, color="#444"),
    )
    fig.update_layout(margin=dict(b=bottom_margin))


# ── Plot 1: Country accuracy (correct/neighbor/wrong) ────────────────────

def plot_country_accuracy(results: list[dict], gt: dict, out_dir: Path) -> None:
    correct = neighbor = wrong = no_gt = 0
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            no_gt += 1
            continue
        country = _extract_country(r.get("country_result", ""))
        if _countries_match(country, truth["country_code"]):
            correct += 1
        elif _is_neighbor(country, truth["country_code"]):
            neighbor += 1
        else:
            wrong += 1

    total = correct + neighbor + wrong
    labels = ["Correct", "Neighbor", "Wrong"]
    values = [correct, neighbor, wrong]
    colors = [COLOR_CORRECT, COLOR_NEIGHBOR, COLOR_WRONG]
    pcts = [v / total * 100 for v in values]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_color="#222", marker_line_width=0.6,
        text=[f"<b>{v}</b><br>{p:.1f}%" for v, p in zip(values, pcts)],
        textposition="outside",
        textfont=dict(family=PAPER_FONT_FAMILY, size=14, color="#222"),
    )])
    fig.update_layout(
        title=dict(text="<b>Country-level prediction accuracy</b>"),
        yaxis=dict(title="Number of images", range=[0, max(values) * 1.20]),
        xaxis=dict(title=""),
        showlegend=False,
    )
    set_paper_style(fig, subtitle=f"Strict country match against ground truth (n = {total} images)")
    annotate_below(fig,
        "<b>Correct:</b> Predicted country code equals the ground-truth country code.&nbsp;&nbsp;"
        "<b>Neighbor:</b> Predicted country shares a land border with the GT country.<br>"
        "<b>Wrong:</b> Neither GT nor a neighbor. The Council reaches "
        f"<b>{correct/total*100:.1f}%</b> strict accuracy and "
        f"<b>{(correct+neighbor)/total*100:.1f}%</b> when neighbors are included.")
    save(fig, out_dir, "01_country_accuracy")


# ── Plot 2: Distance buckets ─────────────────────────────────────────────

def plot_distance_buckets(results: list[dict], gt: dict, out_dir: Path) -> None:
    distances = []
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        coords = _extract_coordinates(r.get("country_result", ""))
        if coords:
            d = _haversine_km(coords[0], coords[1], truth["lat"], truth["lng"])
            distances.append(d)

    if not distances:
        return

    n = len(distances)
    buckets = [
        ("≤ 150 km", sum(1 for d in distances if d <= 150),  COLOR_CORRECT),
        ("≤ 750 km", sum(1 for d in distances if d <= 750),  "#5DBB99"),
        ("≤ 2500 km", sum(1 for d in distances if d <= 2500), COLOR_NEIGHBOR),
        ("> 2500 km", sum(1 for d in distances if d > 2500),  COLOR_WRONG),
    ]
    labels = [b[0] for b in buckets]
    values = [b[1] for b in buckets]
    colors = [b[2] for b in buckets]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_color="#222", marker_line_width=0.6,
        text=[f"<b>{v}</b><br>{v / n * 100:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family=PAPER_FONT_FAMILY, size=14, color="#222"),
    )])
    mean_d = sum(distances) / n
    median_d = sorted(distances)[n // 2]
    fig.update_layout(
        title=dict(text="<b>Haversine distance to ground truth</b>"),
        yaxis=dict(title="Number of images", range=[0, max(values) * 1.20]),
        xaxis=dict(title="Great-circle distance between predicted and true coordinates"),
        showlegend=False,
    )
    set_paper_style(fig,
        subtitle=f"n = {n} images · mean = {mean_d:.0f} km · median = {median_d:.0f} km")
    annotate_below(fig,
        "Buckets are <b>cumulative</b>: the ≤150 km bin is contained in ≤750 km, which is contained in ≤2500 km. "
        "The complementary &gt;2500 km bin reports gross errors.<br>"
        "<b>≤150 km</b> is the GeoGuessr ‘perfect score’ threshold; <b>≤750 km</b> is the standard "
        "country-level threshold. Distances are computed via the Haversine formula.")
    save(fig, out_dir, "02_distance_buckets")


# ── Plot 3: Top predicted countries ──────────────────────────────────────

def plot_top_countries(results: list[dict], out_dir: Path) -> None:
    counter: Counter[str] = Counter()
    for r in results:
        country = _extract_country(r.get("country_result", "")).strip().rstrip(".")
        if country:
            counter[country] += 1
    top = counter.most_common(15)
    if not top:
        return

    labels = [c for c, _ in top][::-1]
    values = [v for _, v in top][::-1]

    fig = go.Figure(data=[go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=COLOR_INFO,
        marker_line_color="#222", marker_line_width=0.6,
        text=[f"<b>{v}</b>" for v in values],
        textposition="outside",
        textfont=dict(family=PAPER_FONT_FAMILY, size=12, color="#222"),
    )])
    fig.update_layout(
        title=dict(text="<b>Top-15 predicted countries</b>"),
        xaxis=dict(title="Number of final predictions"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    set_paper_style(fig, subtitle=f"Frequency of the Council's final country pick (n = {sum(counter.values())})")
    annotate_below(fig,
        "How often each country was selected as the Council's final answer. "
        "Highlights distributional bias of the model: Brazil, USA, Russia and Japan dominate, "
        "consistent with the prior distribution of GeoGuessr image locations and the LLM's "
        "tendency to fall back on high-frequency countries when evidence is ambiguous.")
    save(fig, out_dir, "03_top_predicted_countries")


# ── Plot 4: Timing ───────────────────────────────────────────────────────

def plot_timing(results: list[dict], out_dir: Path) -> None:
    overall = [r.get("timing", {}).get("total_seconds") for r in results]
    overall = [t for t in overall if t]
    with_debate = [r.get("timing", {}).get("total_seconds")
                   for r in results if r.get("debate", {}).get("pairings")]
    with_debate = [t for t in with_debate if t]
    no_debate = [r.get("timing", {}).get("total_seconds")
                 for r in results if not r.get("debate", {}).get("pairings")]
    no_debate = [t for t in no_debate if t]

    fig = go.Figure()
    fig.add_trace(go.Box(y=overall, name=f"All<br>(n = {len(overall)})",
                         marker_color=COLOR_INFO, line=dict(width=1.5),
                         boxmean="sd", fillcolor="rgba(0,114,178,0.25)"))
    fig.add_trace(go.Box(y=with_debate, name=f"With debate<br>(n = {len(with_debate)})",
                         marker_color=COLOR_WRONG, line=dict(width=1.5),
                         boxmean="sd", fillcolor="rgba(213,94,0,0.25)"))
    fig.add_trace(go.Box(y=no_debate, name=f"No debate<br>(n = {len(no_debate)})",
                         marker_color=COLOR_CORRECT, line=dict(width=1.5),
                         boxmean="sd", fillcolor="rgba(0,158,115,0.25)"))

    fig.update_layout(
        title=dict(text="<b>Per-image processing time</b>"),
        yaxis=dict(title="Seconds per image"),
        xaxis=dict(title=""),
        showlegend=False,
    )
    set_paper_style(fig,
        subtitle="Box: median + IQR · whiskers: 1.5·IQR · dashed line: mean ± SD")
    if overall:
        avg_d = sum(with_debate) / len(with_debate)
        avg_n = sum(no_debate) / len(no_debate)
        med_d = sorted(with_debate)[len(with_debate) // 2]
        med_n = sorted(no_debate)[len(no_debate) // 2]
        annotate_below(fig,
            f"<b>With debate</b> (n = {len(with_debate)}): mean {avg_d:.1f} s, median {med_d:.1f} s.&nbsp;&nbsp;"
            f"<b>Without debate</b> (n = {len(no_debate)}): mean {avg_n:.1f} s, median {med_n:.1f} s.<br>"
            f"Debates roughly double per-image inference time, because the moderator opens "
            f"additional pairings and each debating agent generates one or more extra messages "
            f"per round (max 3 rounds).")
    save(fig, out_dir, "04_timing")


# ── Plot 5: Debate overview (no debate vs with debate) ───────────────────

def plot_debate_overview(results: list[dict], out_dir: Path) -> None:
    no_debate = sum(1 for r in results if not r.get("debate", {}).get("pairings"))
    with_debate = len(results) - no_debate

    fig = go.Figure(data=[go.Pie(
        labels=[f"Consensus after R1 ({no_debate})",
                f"Debate triggered ({with_debate})"],
        values=[no_debate, with_debate],
        marker=dict(colors=[COLOR_CORRECT, COLOR_WRONG],
                    line=dict(color="white", width=2)),
        hole=0.5,
        textinfo="label+percent",
        textfont=dict(family=PAPER_FONT_FAMILY, size=14, color="#222"),
        insidetextorientation="horizontal",
        sort=False,
    )])
    fig.update_layout(
        title=dict(text="<b>Pipeline branching: consensus vs. debate</b>"),
        annotations=[dict(text=f"<b>{len(results)}</b><br><span style='font-size:13px'>images</span>",
                         x=0.5, y=0.5, showarrow=False,
                         font=dict(family=PAPER_FONT_FAMILY, size=22, color="#222"))],
        showlegend=False,
    )
    set_paper_style(fig, subtitle=f"n = {len(results)} images")
    annotate_below(fig,
        "<b>Consensus after R1:</b> All five specialist agents propose the same top-1 country in "
        "Round 1; the moderator skips the debate phase and forwards directly to the Judge.<br>"
        "<b>Debate triggered:</b> At least two agents disagreed in R1; the moderator opened one or "
        "more pairwise debates (up to 3 rounds each) before passing all transcripts to the Judge.")
    save(fig, out_dir, "05_debate_overview")


# ── Plot 6: Termination reasons (normalised) ─────────────────────────────

def plot_termination_reasons(results: list[dict], out_dir: Path) -> None:
    """Group raw moderator text into 4 buckets to make it readable."""
    buckets = Counter()
    for r in results:
        decisions = r.get("debate", {}).get("moderator_decisions", [])
        if not decisions:
            continue
        last = decisions[-1]
        reason = last.get("termination_reason", "") or ""
        rl = reason.lower()
        if "consensus" in rl or "agree on the same" in rl:
            buckets["Consensus"] += 1
        elif "stalemate" in rl or "no agent revised" in rl:
            buckets["Stalemate"] += 1
        elif "weak dissent" in rl or "low' or 'speculative" in rl or "low confidence" in rl or "speculative" in rl:
            buckets["Weak dissent"] += 1
        elif "max_rounds" in rl or "maximum debate" in rl:
            buckets["Max rounds"] += 1
        elif "error" in rl or "parse" in rl:
            buckets["Error"] += 1
        else:
            buckets["Other"] += 1

    if not buckets:
        return

    order = ["Consensus", "Weak dissent", "Stalemate", "Max rounds", "Error", "Other"]
    labels = [k for k in order if buckets.get(k, 0) > 0]
    values = [buckets[k] for k in labels]
    colors = {"Consensus": COLOR_CORRECT, "Weak dissent": COLOR_NEUTRAL,
              "Stalemate": COLOR_NEIGHBOR, "Max rounds": COLOR_WRONG,
              "Error": COLOR_BAD, "Other": "#7f8c8d"}
    bar_colors = [colors[l] for l in labels]
    total = sum(values)

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values,
        marker_color=bar_colors,
        marker_line_color="#222", marker_line_width=0.6,
        text=[f"<b>{v}</b><br>{v / total * 100:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family=PAPER_FONT_FAMILY, size=13, color="#222"),
    )])
    fig.update_layout(
        title=dict(text="<b>Moderator termination reason</b>"),
        yaxis=dict(title="Number of images", range=[0, max(values) * 1.20]),
        xaxis=dict(title=""),
        showlegend=False,
    )
    set_paper_style(fig,
        subtitle=f"Last moderator decision · n = {total} images")
    annotate_below(fig,
        "<b>Consensus:</b> All agents at ≥medium confidence agree on the same top-1 country.&nbsp;&nbsp;"
        "<b>Weak dissent:</b> Only one agent dissents at low/speculative confidence.<br>"
        "<b>Stalemate:</b> No agent revised in the last debate round.&nbsp;&nbsp;"
        "<b>Max rounds:</b> The hard cap of 3 debate rounds was reached.<br>"
        "Free-text moderator strings were grouped into 6 buckets via case-insensitive keyword matching.")
    save(fig, out_dir, "06_termination_reasons")


# ── Plot 7: Per-agent debate behavior (revision %) ───────────────────────

def plot_agent_debate_behavior(results: list[dict], out_dir: Path) -> None:
    debates = Counter()
    exchanges = Counter()
    revised = Counter()
    defended = Counter()

    for r in results:
        for p in r.get("debate", {}).get("pairings", []):
            for agent in (p.get("agent_a"), p.get("agent_b")):
                if agent in AGENT_NAMES:
                    debates[agent] += 1
            for ex in p.get("exchanges", []):
                name = ex.get("agent_name")
                if name in AGENT_NAMES:
                    exchanges[name] += 1
                    if ex.get("revised"):
                        revised[name] += 1
                    else:
                        defended[name] += 1

    rev_pct = [
        (revised[a] / exchanges[a] * 100) if exchanges[a] else 0
        for a in AGENT_NAMES
    ]
    deb_counts = [debates[a] for a in AGENT_NAMES]
    rev_counts = [revised[a] for a in AGENT_NAMES]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("(a) Pairings the agent participated in",
                        "(b) Share of messages flagged as <i>revised</i>"),
        column_widths=[0.45, 0.55],
        horizontal_spacing=0.13,
    )

    fig.add_trace(go.Bar(
        x=AGENT_NAMES, y=deb_counts,
        marker_color=[AGENT_COLORS[a] for a in AGENT_NAMES],
        marker_line_color="#222", marker_line_width=0.6,
        text=[f"<b>{v}</b>" for v in deb_counts], textposition="outside",
        textfont=dict(family=PAPER_FONT_FAMILY, size=13, color="#222"),
        name="Debates",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=AGENT_NAMES, y=rev_pct,
        marker_color=[AGENT_COLORS[a] for a in AGENT_NAMES],
        marker_line_color="#222", marker_line_width=0.6,
        text=[f"<b>{p:.0f}%</b><br>{rev_counts[i]}/{exchanges[AGENT_NAMES[i]]}"
              for i, p in enumerate(rev_pct)],
        textposition="outside",
        textfont=dict(family=PAPER_FONT_FAMILY, size=12, color="#222"),
        name="Revision %",
        showlegend=False,
    ), row=1, col=2)

    fig.update_yaxes(title_text="Number of pairings",
                     range=[0, max(deb_counts) * 1.25], row=1, col=1)
    fig.update_yaxes(title_text="Revision rate (%)",
                     range=[0, 110], row=1, col=2)

    fig.update_layout(
        title=dict(text="<b>Per-agent debate behavior</b>"),
    )
    set_paper_style(fig,
        subtitle="Counted across all images that triggered at least one debate")
    # Subplot titles need the paper font too
    for anno in fig.layout.annotations:
        if anno.text and anno.text.startswith("(a)") or (anno.text and anno.text.startswith("(b)")):
            anno.font = dict(family=PAPER_FONT_FAMILY, size=13, color="#222")
    annotate_below(fig,
        "<b>(a)</b> A pairing is counted once per debate the agent participated in (an agent can appear "
        "in several pairings on the same image).<br>"
        "<b>(b)</b> Revision rate = share of the agent's exchanges where the agent flagged its own "
        "message as <i>revised=true</i>, i.e. it changed its top-1 country since the previous turn.<br>"
        "<b>Regulatory</b> is paired most often (moderator priority for hard textual constraints) yet "
        "almost never revises, it defends its position. <b>Landscape</b> revises most readily.")
    save(fig, out_dir, "07_per_agent_debate_behavior")


# ── Plot 8: Pairing frequency ────────────────────────────────────────────

def plot_pairing_frequency(results: list[dict], out_dir: Path) -> None:
    counter: Counter[tuple[str, str]] = Counter()
    for r in results:
        for p in r.get("debate", {}).get("pairings", []):
            a = p.get("agent_a", "?")
            b = p.get("agent_b", "?")
            counter[tuple(sorted([a, b]))] += 1

    pairs = counter.most_common()
    if not pairs:
        return
    labels = [f"{a} vs {b}" for (a, b), _ in pairs][::-1]
    values = [v for _, v in pairs][::-1]

    fig = go.Figure(data=[go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=COLOR_INFO,
        marker_line_color="#222", marker_line_width=0.6,
        text=[f"<b>{v}</b>" for v in values],
        textposition="outside",
        textfont=dict(family=PAPER_FONT_FAMILY, size=12, color="#222"),
    )])
    fig.update_layout(
        title=dict(text="<b>Pairing frequency by agent pair</b>"),
        xaxis=dict(title="Number of debates"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    set_paper_style(fig,
        subtitle=f"Pairs the moderator opened across all images (total = {sum(values)} debates)")
    annotate_below(fig,
        "Each row counts how often the moderator paired the two agents into a debate.<br>"
        "The moderator prompt explicitly prioritises <b>regulatory</b> (hard textual constraints "
        "such as license plates, signs, script) and <b>linguistic</b>,<br>"
        "while it discourages <b>landscape</b> vs. <b>meta</b> pairings (both reason on coarse, "
        "holistic features and would generate redundant exchange).")
    save(fig, out_dir, "08_pairing_frequency")


# ── Plot 9: Convergence (basic, no GT) ───────────────────────────────────

def plot_convergence_basic(results: list[dict], out_dir: Path) -> None:
    converged = 0
    disagreed = 0
    for r in results:
        pairings = r.get("debate", {}).get("pairings", [])
        if not pairings:
            continue
        positions = set()
        for p in pairings:
            for ex in p.get("exchanges", []):
                pos = (ex.get("position") or "").lower().strip()
                if pos:
                    pass
            # final positions per agent in this pairing
            for agent in (p.get("agent_a"), p.get("agent_b")):
                last = _last_position_in_pairing(p, agent)
                if last:
                    positions.add(last.lower().strip())
        if len(positions) <= 1:
            converged += 1
        else:
            disagreed += 1

    total = converged + disagreed
    if total == 0:
        return

    fig = go.Figure(data=[go.Pie(
        labels=[f"Converged ({converged})",
                f"Stayed split ({disagreed})"],
        values=[converged, disagreed],
        marker=dict(colors=[COLOR_CORRECT, COLOR_WRONG],
                    line=dict(color="white", width=2)),
        hole=0.5,
        textinfo="label+percent",
        textfont=dict(family=PAPER_FONT_FAMILY, size=14, color="#222"),
        insidetextorientation="horizontal",
        sort=False,
    )])
    fig.update_layout(
        title=dict(text="<b>Debate convergence (without ground truth)</b>"),
        annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:13px'>debates</span>",
                         x=0.5, y=0.5, showarrow=False,
                         font=dict(family=PAPER_FONT_FAMILY, size=22, color="#222"))],
        showlegend=False,
    )
    set_paper_style(fig, subtitle=f"n = {total} images that triggered ≥1 debate")
    annotate_below(fig,
        "<b>Converged:</b> Across all pairings opened on this image, every debating agent's "
        "<i>last</i> stated country is identical.<br>"
        "<b>Stayed split:</b> At least two end positions differ. "
        "This view does not measure whether the converged answer is correct, that is decomposed "
        "in the GT-based convergence plot.")
    save(fig, out_dir, "09_convergence_basic")


# ── Plot 10: GT-based convergence ────────────────────────────────────────

def plot_convergence_gt(results: list[dict], gt: dict, out_dir: Path) -> None:
    correct = wrong = no_conv = 0
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        pairings = r.get("debate", {}).get("pairings", [])
        if not pairings:
            continue
        debating: dict[str, str] = {}
        for p in pairings:
            for agent in (p.get("agent_a"), p.get("agent_b")):
                if not agent:
                    continue
                last = _last_position_in_pairing(p, agent)
                if last:
                    debating[agent] = last
        if not debating:
            continue
        positions = {pos.lower().strip() for pos in debating.values() if pos}
        if len(positions) > 1:
            no_conv += 1
        else:
            only = next(iter(positions))
            if _agent_correct(only, truth["country_code"]):
                correct += 1
            else:
                wrong += 1

    total = correct + wrong + no_conv
    if total == 0:
        return

    labels = ["Converged on truth", "Converged on falsehood", "Did not converge"]
    values = [correct, wrong, no_conv]
    colors = [COLOR_CORRECT, COLOR_WRONG, COLOR_NEUTRAL]
    pcts = [v / total * 100 for v in values]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_color="#222", marker_line_width=0.6,
        text=[f"<b>{v}</b><br>{p:.1f}%" for v, p in zip(values, pcts)],
        textposition="outside",
        textfont=dict(family=PAPER_FONT_FAMILY, size=14, color="#222"),
    )])
    fig.update_layout(
        title=dict(text="<b>GT-based debate convergence</b>"),
        yaxis=dict(title="Number of images", range=[0, max(values) * 1.30]),
        xaxis=dict(title=""),
        showlegend=False,
    )
    converged = correct + wrong
    correct_ratio = correct / converged * 100 if converged else 0
    set_paper_style(fig,
        subtitle=f"n = {total} images that triggered ≥1 debate")
    annotate_below(fig,
        f"<b>Converged on truth:</b> Every debating agent's last stated country equals the GT, debate found the right answer.<br>"
        f"<b>Converged on falsehood:</b> All debaters agree, but on the wrong country, the Council "
        f"collectively talked itself into a false consensus.<br>"
        f"<b>Did not converge:</b> At least two end positions differ; the Judge has to arbitrate.<br>"
        f"<b>Key result:</b> of the {converged} converged debates, only "
        f"<b>{correct} ({correct_ratio:.1f}%) land on the truth</b>, convergence is <i>not</i> a "
        f"reliable proxy for correctness.")
    save(fig, out_dir, "10_convergence_gt")


# ── Plot 11: Constructive vs destructive (per pairing) ───────────────────

def plot_constructive_destructive(results: list[dict], gt: dict, out_dir: Path) -> None:
    cats = Counter()
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        for p in r.get("debate", {}).get("pairings", []):
            cats[_classify_pairing(p, truth["country_code"])["category"]] += 1

    total = sum(cats.values())
    if total == 0:
        return

    labels = ["Constructive<br>(truth wins)",
              "Destructive<br>(truth loses)",
              "Stand correct<br>(truth holds)",
              "Both wrong<br>(neutral)",
              "Both correct"]
    keys = ["CONSTRUCTIVE", "DESTRUCTIVE", "STAND_CORRECT", "BOTH_WRONG_NEUTRAL", "BOTH_CORRECT"]
    values = [cats.get(k, 0) for k in keys]
    colors = [COLOR_CORRECT, COLOR_WRONG, COLOR_NEIGHBOR, COLOR_NEUTRAL, COLOR_INFO]
    pcts = [v / total * 100 for v in values]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_color="#222", marker_line_width=0.6,
        text=[f"<b>{v}</b><br>{p:.1f}%" for v, p in zip(values, pcts)],
        textposition="outside",
        textfont=dict(family=PAPER_FONT_FAMILY, size=13, color="#222"),
    )])
    fig.update_layout(
        title=dict(text="<b>Constructive vs. destructive debate (per pairing)</b>"),
        yaxis=dict(title="Number of pairings", range=[0, max(values) * 1.25]),
        xaxis=dict(title=""),
        showlegend=False,
    )
    set_paper_style(fig,
        subtitle=f"Each pairing classified by initial vs. final positions vs. ground truth · n = {total} pairings")

    truth_bearer = cats["CONSTRUCTIVE"] + cats["DESTRUCTIVE"] + cats["STAND_CORRECT"]
    if truth_bearer:
        c_pct = cats["CONSTRUCTIVE"] / truth_bearer * 100
        d_pct = cats["DESTRUCTIVE"] / truth_bearer * 100
        annotate_below(fig,
            f"<b>Constructive:</b> The initially-correct agent convinces the wrong one to switch onto the GT, debate created truth.<br>"
            f"<b>Destructive:</b> The initially-wrong agent pulls the correct one onto its wrong position, debate destroyed truth.<br>"
            f"<b>Stand correct:</b> The correct agent holds its ground; the wrong one neither flips nor convinces. "
            f"<b>Both wrong / both correct:</b> No truth-carrier asymmetry.<br>"
            f"<b>Across the {truth_bearer} pairings with exactly one initially-correct agent: "
            f"{c_pct:.1f}% constructive vs. {d_pct:.1f}% destructive</b>, the debate phase is <i>net "
            f"neutral</i> at the truth level on this dataset.")
    save(fig, out_dir, "11_constructive_destructive")


# ── Plot 12: Per-agent convergence participation ─────────────────────────

def plot_per_agent_convergence(results: list[dict], gt: dict, out_dir: Path) -> None:
    per_agent: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}

    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        pairings = r.get("debate", {}).get("pairings", [])
        if not pairings:
            continue
        gt_code = truth["country_code"]
        debating: dict[str, str] = {}
        for p in pairings:
            for agent in (p.get("agent_a"), p.get("agent_b")):
                if not agent:
                    continue
                last = _last_position_in_pairing(p, agent)
                if last:
                    debating[agent] = last
        if not debating:
            continue
        positions = {pos.lower().strip() for pos in debating.values() if pos}
        if len(positions) > 1:
            outcome = "not_converged"
        else:
            only = next(iter(positions))
            outcome = "converged_correct" if _agent_correct(only, gt_code) else "converged_wrong"

        for agent, end_pos in debating.items():
            if agent not in per_agent:
                continue
            per_agent[agent][outcome] += 1
            per_agent[agent]["total"] += 1
            if _agent_correct(end_pos, gt_code):
                per_agent[agent]["end_correct"] += 1

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("(a) Outcome of the image, broken down per agent",
                        "(b) EndOK%, agent's own end position equals GT"),
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.13,
    )

    cor = [per_agent[a]["converged_correct"] for a in AGENT_NAMES]
    wro = [per_agent[a]["converged_wrong"] for a in AGENT_NAMES]
    nocv = [per_agent[a]["not_converged"] for a in AGENT_NAMES]

    fig.add_trace(go.Bar(name="Converged on truth", x=AGENT_NAMES, y=cor,
                         marker_color=COLOR_CORRECT,
                         marker_line_color="#222", marker_line_width=0.5,
                         text=[f"<b>{v}</b>" if v else "" for v in cor],
                         textfont=dict(family=PAPER_FONT_FAMILY, size=12, color="white"),
                         textposition="inside"),
                  row=1, col=1)
    fig.add_trace(go.Bar(name="Converged on falsehood", x=AGENT_NAMES, y=wro,
                         marker_color=COLOR_WRONG,
                         marker_line_color="#222", marker_line_width=0.5,
                         text=[f"<b>{v}</b>" if v else "" for v in wro],
                         textfont=dict(family=PAPER_FONT_FAMILY, size=12, color="white"),
                         textposition="inside"),
                  row=1, col=1)
    fig.add_trace(go.Bar(name="Did not converge", x=AGENT_NAMES, y=nocv,
                         marker_color=COLOR_NEUTRAL,
                         marker_line_color="#222", marker_line_width=0.5,
                         text=[f"<b>{v}</b>" if v else "" for v in nocv],
                         textfont=dict(family=PAPER_FONT_FAMILY, size=12, color="white"),
                         textposition="inside"),
                  row=1, col=1)

    end_pcts = []
    end_text = []
    for a in AGENT_NAMES:
        total = per_agent[a]["total"]
        ec = per_agent[a]["end_correct"]
        p = (ec / total * 100) if total else 0
        end_pcts.append(p)
        end_text.append(f"<b>{p:.1f}%</b><br>{ec}/{total}")

    fig.add_trace(go.Bar(
        x=AGENT_NAMES, y=end_pcts,
        marker_color=[AGENT_COLORS[a] for a in AGENT_NAMES],
        marker_line_color="#222", marker_line_width=0.6,
        text=end_text, textposition="outside",
        textfont=dict(family=PAPER_FONT_FAMILY, size=12, color="#222"),
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text="<b>Per-agent convergence participation</b>"),
        barmode="stack",
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center", yanchor="top",
            font=dict(family=PAPER_FONT_FAMILY, size=12, color="#222"),
        ),
    )
    fig.update_yaxes(title_text="Number of images with debate", row=1, col=1)
    fig.update_yaxes(title_text="EndOK (%)", range=[0, 110], row=1, col=2)
    set_paper_style(fig,
        subtitle="Each agent counted once per debating image · stacked counts must equal the agent's row total")
    for anno in fig.layout.annotations:
        if anno.text and (anno.text.startswith("(a)") or anno.text.startswith("(b)")):
            anno.font = dict(family=PAPER_FONT_FAMILY, size=13, color="#222")

    annotate_below(fig,
        "<b>(a)</b> For every debate-triggering image, we record the convergence outcome and "
        "attribute it to all agents that participated.<br>"
        "Tall bars = agent debates often; the colour split shows whether those debates ended on "
        "truth (green), on a false consensus (red) or with an unresolved split (grey).<br>"
        "<b>(b) EndOK%</b> = share of pairings where the agent's <i>own last stated country</i> "
        "equals GT, regardless of what the rest of the Council ended on.<br>"
        "<b>Linguistic</b> has the highest EndOK rate but the lowest debate count; "
        "<b>landscape</b> is the least reliable per-debate truth-bearer.",
        has_legend=True)
    save(fig, out_dir, "12_per_agent_convergence")


# ── Plot 13: Win/loss matrix per agent ───────────────────────────────────

def plot_win_loss_matrix(results: list[dict], gt: dict, out_dir: Path) -> None:
    per_agent: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}

    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        for p in r.get("debate", {}).get("pairings", []):
            cls = _classify_pairing(p, truth["country_code"])
            cat = cls["category"]
            a = cls["agent_a"]
            b = cls["agent_b"]
            if cat == "CONSTRUCTIVE":
                if cls["winner"] in per_agent:
                    per_agent[cls["winner"]]["constructive_win"] += 1
                if cls["loser"] in per_agent:
                    per_agent[cls["loser"]]["constructive_loss"] += 1
            elif cat == "DESTRUCTIVE":
                if cls["winner"] in per_agent:
                    per_agent[cls["winner"]]["destructive_win"] += 1
                if cls["loser"] in per_agent:
                    per_agent[cls["loser"]]["destructive_loss"] += 1
            elif cat == "STAND_CORRECT":
                correct = a if cls["init_a_correct"] else b
                wrong = b if cls["init_a_correct"] else a
                if correct in per_agent:
                    per_agent[correct]["stand_correct"] += 1
                if wrong in per_agent:
                    per_agent[wrong]["stand_wrong"] += 1

    metrics = [
        ("C-Win",   "constructive_win",  COLOR_CORRECT),
        ("C-Loss",  "constructive_loss", "#5DBB99"),
        ("D-Win",   "destructive_win",   COLOR_WRONG),
        ("D-Loss",  "destructive_loss",  "#A04000"),
        ("StandOK", "stand_correct",     COLOR_NEIGHBOR),
        ("StandX",  "stand_wrong",       COLOR_NEUTRAL),
    ]

    fig = go.Figure()
    for label, key, color in metrics:
        ys = [per_agent[a][key] for a in AGENT_NAMES]
        fig.add_trace(go.Bar(
            name=label, x=AGENT_NAMES, y=ys,
            marker_color=color,
            marker_line_color="#222", marker_line_width=0.5,
            text=[f"<b>{v}</b>" if v else "" for v in ys],
            textposition="inside",
            textfont=dict(family=PAPER_FONT_FAMILY, size=11, color="white"),
        ))

    fig.update_layout(
        title=dict(text="<b>Per-agent win/loss matrix</b>"),
        yaxis=dict(title="Number of pairings"),
        xaxis=dict(title=""),
        barmode="group",
        bargap=0.18, bargroupgap=0.05,
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center", yanchor="top",
            font=dict(family=PAPER_FONT_FAMILY, size=12, color="#222"),
        ),
    )
    set_paper_style(fig,
        subtitle="Outcome of each pairing, attributed to the two participating agents")

    nettruth_lines = []
    for a in AGENT_NAMES:
        c = per_agent[a]
        net = (c["constructive_win"] + c["constructive_loss"] + c["stand_correct"]) \
              - (c["destructive_win"] + c["destructive_loss"] + c["stand_wrong"])
        nettruth_lines.append(f"<b>{a}</b>: {net:+d}")
    annotate_below(fig,
        "<b>C-Win:</b> Agent was initially correct and convinced its wrong opponent → truth.&nbsp;&nbsp;"
        "<b>D-Win:</b> Agent was initially wrong and convinced its correct opponent → falsehood.<br>"
        "<b>C-Loss:</b> Agent was wrong and let itself be corrected (good for the Council).&nbsp;&nbsp;"
        "<b>D-Loss:</b> Agent was correct and gave in to the wrong opponent (bad).<br>"
        "<b>StandOK / StandX:</b> Agent held its initial position; it was respectively right or wrong.<br>"
        "<b>NetTruth = (C-Win + C-Loss + StandOK) − (D-Win + D-Loss + StandX)</b> measures whether "
        "an agent's behavior is net truth-promoting:&nbsp;&nbsp;"
        + "&nbsp;&nbsp; ".join(nettruth_lines),
        has_legend=True)
    save(fig, out_dir, "13_win_loss_matrix")


# ── Plot 14: Pair-specific outcomes (stacked bars) ───────────────────────

def plot_pair_specific_outcomes(results: list[dict], gt: dict, out_dir: Path) -> None:
    pair_stats: dict[tuple[str, str], Counter] = {}
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        for p in r.get("debate", {}).get("pairings", []):
            cls = _classify_pairing(p, truth["country_code"])
            key = tuple(sorted([cls["agent_a"], cls["agent_b"]]))
            pair_stats.setdefault(key, Counter())[cls["category"]] += 1

    if not pair_stats:
        return

    # Sort by NetConstructive (constructive − destructive). Plotly horizontal bars
    # render the first item at the BOTTOM, so sort ascending → most-destructive at
    # the bottom of the chart and most-constructive at the top, as promised by the
    # subtitle.
    items = sorted(
        pair_stats.items(),
        key=lambda kv: kv[1].get("CONSTRUCTIVE", 0) - kv[1].get("DESTRUCTIVE", 0),
    )
    labels = [f"{a} vs {b}" for (a, b), _ in items]
    constructive = [c.get("CONSTRUCTIVE", 0) for _, c in items]
    destructive = [c.get("DESTRUCTIVE", 0) for _, c in items]
    stand = [c.get("STAND_CORRECT", 0) for _, c in items]
    bothw = [c.get("BOTH_WRONG_NEUTRAL", 0) for _, c in items]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Constructive",  y=labels, x=constructive,
                         orientation="h", marker_color=COLOR_CORRECT,
                         marker_line_color="#222", marker_line_width=0.5,
                         text=[str(v) if v else "" for v in constructive],
                         textfont=dict(family=PAPER_FONT_FAMILY, size=11, color="white"),
                         textposition="inside"))
    fig.add_trace(go.Bar(name="Destructive",   y=labels, x=destructive,
                         orientation="h", marker_color=COLOR_WRONG,
                         marker_line_color="#222", marker_line_width=0.5,
                         text=[str(v) if v else "" for v in destructive],
                         textfont=dict(family=PAPER_FONT_FAMILY, size=11, color="white"),
                         textposition="inside"))
    fig.add_trace(go.Bar(name="Stand correct", y=labels, x=stand,
                         orientation="h", marker_color=COLOR_NEIGHBOR,
                         marker_line_color="#222", marker_line_width=0.5,
                         text=[str(v) if v else "" for v in stand],
                         textfont=dict(family=PAPER_FONT_FAMILY, size=11, color="white"),
                         textposition="inside"))
    fig.add_trace(go.Bar(name="Both wrong",    y=labels, x=bothw,
                         orientation="h", marker_color=COLOR_NEUTRAL,
                         marker_line_color="#222", marker_line_width=0.5,
                         text=[str(v) if v else "" for v in bothw],
                         textfont=dict(family=PAPER_FONT_FAMILY, size=11, color="white"),
                         textposition="inside"))

    fig.update_layout(
        title=dict(text="<b>Pairing-classification distribution per agent pair</b>"),
        xaxis=dict(title="Number of pairings"),
        barmode="stack",
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center", yanchor="top",
            font=dict(family=PAPER_FONT_FAMILY, size=12, color="#222"),
        ),
    )
    set_paper_style(fig,
        subtitle="Sorted by NetConstructive = constructive − destructive (most constructive at top)")
    annotate_below(fig,
        "Each row shows how the pairings between two given agents distributed across the four "
        "ground-truth-based categories defined in the constructive-vs-destructive plot.<br>"
        "Pairs at the top are <i>net truth-promoting</i> (more constructive than destructive);<br>"
        "pairs at the bottom systematically destroy truth. The largest pair,<br>"
        "<b>landscape vs. regulatory</b>, is also the most destructive on this dataset.",
        has_legend=True)
    save(fig, out_dir, "14_pair_specific_outcomes")


# ── Plot 15: R1 majority vs final ────────────────────────────────────────

def plot_r1_majority_match(results: list[dict], out_dir: Path) -> None:
    matches = 0
    no_majority = 0
    differs = 0

    for r in results:
        country_result = r.get("country_result", "")
        final_country = _extract_country(country_result)

        r1 = r.get("round_1_assessments", {})
        votes = Counter()
        for agent in AGENT_NAMES:
            cands = r1.get(agent, {}).get("candidates", [])
            if cands:
                top = (cands[0].get("country") or "").strip()
                if top:
                    votes[top.lower()] += 1
        if not votes:
            no_majority += 1
            continue
        majority, _ = votes.most_common(1)[0]
        if final_country.strip().lower() == majority:
            matches += 1
        else:
            differs += 1

    total = matches + differs + no_majority
    fig = go.Figure(data=[go.Pie(
        labels=[f"Final = R1 majority ({matches})",
                f"Final ≠ R1 majority ({differs})",
                f"No R1 majority ({no_majority})"],
        values=[matches, differs, no_majority],
        marker=dict(colors=[COLOR_CORRECT, COLOR_WRONG, COLOR_NEUTRAL],
                    line=dict(color="white", width=2)),
        hole=0.5,
        textinfo="label+percent",
        textfont=dict(family=PAPER_FONT_FAMILY, size=13, color="#222"),
        insidetextorientation="horizontal",
        sort=False,
    )])
    fig.update_layout(
        title=dict(text="<b>Judge alignment with Round-1 plurality vote</b>"),
        annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:13px'>images</span>",
                         x=0.5, y=0.5, showarrow=False,
                         font=dict(family=PAPER_FONT_FAMILY, size=22, color="#222"))],
        showlegend=False,
    )
    set_paper_style(fig, subtitle=f"n = {total} images")
    pct_match = matches / total * 100 if total else 0
    annotate_below(fig,
        f"<b>R1 majority:</b> the country picked as top-1 by the largest number of agents in Round 1 "
        f"(ties broken by Counter ordering).<br>"
        f"<b>Final:</b> the Judge's answer after the debate phase (or directly after R1 if consensus held).<br>"
        f"In <b>{pct_match:.1f}%</b> of images the Judge follows the R1 plurality. The {differs} "
        f"divergences arise mainly from debate-induced concessions, hard textual constraints, or "
        f"explicit overrides by the Judge.")
    save(fig, out_dir, "15_r1_majority_vs_final")


# ── Plot 16: Geographic bias (GT → Pred arrows on world map) ─────────────

def plot_geographic_bias(results: list[dict], gt: dict, out_dir: Path) -> None:
    """World map with GT→Prediction vectors plus N-S / W-E bias statistics.

    Each pair (GT, Prediction) is rendered as a thin line on a world map; the
    aggregate Δlat / Δlng tell whether the Council systematically under- or
    over-shoots in any cardinal direction.
    """
    import math

    pairs = []  # (gt_lat, gt_lng, pred_lat, pred_lng)
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        pred = _extract_coordinates(r.get("country_result", ""))
        if not pred:
            continue
        pairs.append((truth["lat"], truth["lng"], pred[0], pred[1]))

    if not pairs:
        return

    n = len(pairs)
    # Δlat: positive = prediction NORTH of GT, negative = SOUTH
    # Δlng: positive = prediction EAST of GT, negative = WEST
    # We compute the W-E km using the cosine of the GT latitude (great-circle
    # approximation along a parallel), so a 1° shift near the equator weighs
    # more than a 1° shift near the poles.
    KM_PER_DEG = 111.0  # ≈ 111.32 km per degree of latitude
    dlat_deg = [pred_la - gt_la for gt_la, _gl, pred_la, _pl in pairs]
    dlng_deg = []
    ns_km = []
    we_km = []
    for gt_la, gt_lo, pred_la, pred_lo in pairs:
        dlo = pred_lo - gt_lo
        # Wrap longitude diff into [-180, 180] so a NZ ↔ Chile pair doesn't
        # falsely count as a 300° shift.
        if dlo > 180:
            dlo -= 360
        elif dlo < -180:
            dlo += 360
        dlng_deg.append(dlo)
        ns_km.append((pred_la - gt_la) * KM_PER_DEG)
        we_km.append(dlo * KM_PER_DEG * math.cos(math.radians(gt_la)))

    def _mean(xs): return sum(xs) / len(xs)
    def _median(xs):
        s = sorted(xs)
        m = len(s) // 2
        return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])

    mean_dlat = _mean(dlat_deg)
    mean_dlng = _mean(dlng_deg)
    mean_ns = _mean(ns_km)
    mean_we = _mean(we_km)
    med_ns = _median(ns_km)
    med_we = _median(we_km)

    fig = go.Figure()

    # GT→Pred lines: one mini-trace per pair so each line is independent.
    # We bundle them into a single trace using None separators in the coordinate
    # arrays, Plotly draws a polyline that lifts the pen on None.
    line_lats: list[float | None] = []
    line_lngs: list[float | None] = []
    for gt_la, gt_lo, pred_la, pred_lo in pairs:
        # Same wrap fix as above for the actual drawn line so trans-Pacific
        # pairs go the short way.
        plo = pred_lo
        if pred_lo - gt_lo > 180:
            plo = pred_lo - 360
        elif pred_lo - gt_lo < -180:
            plo = pred_lo + 360
        line_lats.extend([gt_la, pred_la, None])
        line_lngs.extend([gt_lo, plo, None])

    fig.add_trace(go.Scattergeo(
        lon=line_lngs, lat=line_lats,
        mode="lines",
        line=dict(width=0.7, color="#888"),
        name="GT → Prediction",
        opacity=0.55,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergeo(
        lon=[gt_lo for _, gt_lo, _, _ in pairs],
        lat=[gt_la for gt_la, _, _, _ in pairs],
        mode="markers",
        marker=dict(size=5, color=COLOR_CORRECT, line=dict(width=0.4, color="#222")),
        name="Ground truth",
    ))
    fig.add_trace(go.Scattergeo(
        lon=[pred_lo for _, _, _, pred_lo in pairs],
        lat=[pred_la for _, _, pred_la, _ in pairs],
        mode="markers",
        marker=dict(size=5, color=COLOR_WRONG, line=dict(width=0.4, color="#222"),
                    symbol="diamond"),
        name="Council prediction",
    ))

    fig.update_layout(
        title=dict(text="<b>Geographic bias: prediction vs. ground truth</b>"),
        geo=dict(
            projection_type="natural earth",
            showland=True, landcolor="#f3efe6",
            showocean=True, oceancolor="#eaf3f8",
            showcountries=True, countrycolor="#bbb",
            showcoastlines=True, coastlinecolor="#888", coastlinewidth=0.5,
            lataxis=dict(showgrid=True, gridcolor="#dcdcdc", gridwidth=0.5),
            lonaxis=dict(showgrid=True, gridcolor="#dcdcdc", gridwidth=0.5),
            domain=dict(x=[0.02, 0.98], y=[0.02, 0.95]),
        ),
        legend=dict(
            orientation="h", y=-0.02, x=0.5, xanchor="center", yanchor="top",
            font=dict(family=PAPER_FONT_FAMILY, size=12, color="#222"),
        ),
        height=900,
    )
    set_paper_style(fig,
        subtitle=f"n = {n} images with parsed prediction coordinates")

    ns_label = "north" if mean_ns >= 0 else "south"
    we_label = "east" if mean_we >= 0 else "west"

    annotate_below(fig,
        f"<b>Bias statistics (Prediction − Ground truth):</b><br>"
        f"Mean Δlat = <b>{mean_dlat:+.2f}°</b>  ·  "
        f"Mean N-S offset = <b>{mean_ns:+.0f} km</b> "
        f"(median {med_ns:+.0f} km, i.e. predictions tend {ns_label} of GT)<br>"
        f"Mean Δlng = <b>{mean_dlng:+.2f}°</b>  ·  "
        f"Mean W-E offset = <b>{mean_we:+.0f} km</b> "
        f"(median {med_we:+.0f} km, i.e. predictions tend {we_label} of GT)<br>"
        f"<i>Sign convention:</i> positive Δlat = prediction north of GT; "
        f"positive Δlng = prediction east of GT. W-E km uses cos(lat<sub>GT</sub>) so "
        f"high-latitude shifts are not over-counted.",
        has_legend=True)
    # Use a taller canvas, a world map needs more vertical room than the
    # standard 750 px height used by the other 15 plots.
    out_path = out_dir / "16_geographic_bias.png"
    fig.write_image(str(out_path), width=WIDTH, height=900, scale=SCALE)
    print(f"  → {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Plotly plots for all VLM Council statistics")
    parser.add_argument("results_dir", help="Directory with result.json files")
    parser.add_argument("ground_truth", help="Path to georc_locations.csv")
    parser.add_argument("--out", default="debate_analysis_images",
                        help="Output directory for PNG plots (default: debate_analysis_images)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    gt_path = Path(args.ground_truth)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir} ...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} successful results")

    print(f"Loading ground truth from {gt_path} ...")
    gt = _load_ground_truth(gt_path)
    print(f"Loaded {len(gt)} GT entries")

    print(f"\nWriting plots to {out_dir}/ ...\n")

    plot_country_accuracy(results, gt, out_dir)
    plot_distance_buckets(results, gt, out_dir)
    plot_top_countries(results, out_dir)
    plot_timing(results, out_dir)
    plot_debate_overview(results, out_dir)
    plot_termination_reasons(results, out_dir)
    plot_agent_debate_behavior(results, out_dir)
    plot_pairing_frequency(results, out_dir)
    plot_convergence_basic(results, out_dir)
    plot_convergence_gt(results, gt, out_dir)
    plot_constructive_destructive(results, gt, out_dir)
    plot_per_agent_convergence(results, gt, out_dir)
    plot_win_loss_matrix(results, gt, out_dir)
    plot_pair_specific_outcomes(results, gt, out_dir)
    plot_r1_majority_match(results, out_dir)
    plot_geographic_bias(results, gt, out_dir)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
