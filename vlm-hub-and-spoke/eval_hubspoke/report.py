"""Compose a single markdown + HTML report from all eval outputs.

Reads (when present):
  <out>/geo_metrics.json
  <out>/agent_metrics.json
  <out>/heatmap_metrics.json
  <out>/dynamics_metrics.json
  <out>/judge_summary.json

Writes <out>/report.md and (if Jinja2 available) <out>/report.html.

Section order: TL;DR, then 1. Ground-Truth Statistics, 2. Approach Dynamics,
3. LLM-as-Judge Verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval_hubspoke.loader import AGENT_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1%}"


def _fmt_km(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:,.0f} km"


def _fmt_float(x: float | None, decimals: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{decimals}f}"


# ---------------------------------------------------------------------------
# TL;DR
# ---------------------------------------------------------------------------

def _section_tldr(geo: dict | None, agents: dict | None, judge: dict | None) -> list[str]:
    out = ["# VLM Council Hub and Spoke Evaluation Report", ""]
    tldr: list[str] = []
    if geo:
        acc = geo.get("country_accuracy")
        hav = geo.get("haversine_km", {}) or {}
        tldr.append(
            f"- **Country accuracy:** {_fmt_pct(acc)} "
            f"(n = {geo.get('n_total', 'n/a')})"
        )
        if hav.get("n"):
            tldr.append(
                f"- **Haversine error:** median {_fmt_km(hav.get('median'))}, "
                f"mean {_fmt_km(hav.get('mean'))}"
            )
    if agents:
        tldr.append(
            f"- **Mean discussion rounds:** "
            f"{_fmt_float(agents.get('mean_discussion_rounds'), 2)}"
        )
        tldr.append(f"- **Images with discussion:** {agents.get('n_with_discussion', 0)}")
    if judge and not judge.get("_error"):
        synth = judge.get("judge_synthesis_quality") or {}
        if synth.get("n"):
            tldr.append(
                f"- **Judge synthesis quality:** {_fmt_float(synth.get('mean'))} mean "
                f"({_fmt_float(synth.get('median'))} median)"
            )
    if tldr:
        out += tldr + [""]
    return out


# ---------------------------------------------------------------------------
# 1. Ground-Truth Statistics
# ---------------------------------------------------------------------------

def _section_ground_truth(out_dir: Path, geo: dict | None, agents: dict | None) -> list[str]:
    out = ["## 1. Ground-Truth Statistics", ""]
    plots_dir = out_dir / "plots"

    if geo:
        hav = geo.get("haversine_km", {}) or {}
        out += ["### Headline Metrics", ""]
        out += [
            "| Metric | Value |",
            "|---|---|",
            f"| Country accuracy | {_fmt_pct(geo.get('country_accuracy'))} |",
            f"| Median haversine | {_fmt_km(hav.get('median'))} |",
            f"| Mean haversine | {_fmt_km(hav.get('mean'))} |",
            f"| N images | {geo.get('n_total', 'n/a')} |",
            "",
        ]

        out += ["### Geographic Bias", ""]
        nb = geo.get("north_bias_test", {}) or {}
        eb = geo.get("east_bias_test", {}) or {}
        out.append(f"- North/south bias: {nb.get('interpretation', 'n/a')}")
        out.append(f"- East/west bias: {eb.get('interpretation', 'n/a')}")
        quads = geo.get("quadrants", {}) or {}
        if quads:
            ordered = ", ".join(f"{k}={v}" for k, v in sorted(quads.items()))
            out.append(f"- Error quadrants: {ordered}")
        out.append("")
        for plot_file, caption in [
            ("error_distribution.png", "Lat/lng/haversine error distributions"),
            ("bearing_rose.png", "Direction of prediction errors (truth to prediction)"),
        ]:
            if (plots_dir / plot_file).exists():
                out += [f"![{caption}](plots/{plot_file})", f"_{caption}_", ""]

        confs = geo.get("top_confusions", []) or []
        if confs:
            out += ["### Top Confusion Pairs", ""]
            out += ["| Truth | Predicted | Count |", "|---|---|---|"]
            for row in confs[:15]:
                out.append(f"| {row['truth']} | {row['predicted']} | {row['count']} |")
            out.append("")
            if (plots_dir / "confusion_matrix.png").exists():
                out += ["![Top confusion pairs (truth to predicted)](plots/confusion_matrix.png)",
                        "_Top confusion pairs (truth to predicted)_", ""]

    if agents:
        initial = agents.get("initial_round", {}) or {}
        out += ["### Per-agent Accuracy (Initial Round)", ""]
        out += [
            "| Agent | n | Top-1 | Top-3 | Coverage | Discussion rate | Update rate |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in AGENT_NAMES:
            m = initial.get(a, {}) or {}
            out.append(
                f"| {a} | {m.get('n', 0)} | "
                f"{_fmt_pct(m.get('top1_accuracy', 0))} | "
                f"{_fmt_pct(m.get('top3_hit_rate', 0))} | "
                f"{_fmt_pct(m.get('coverage', 0))} | "
                f"{_fmt_pct(m.get('discussion_rate', 0))} | "
                f"{_fmt_pct(m.get('response_update_rate', 0))} |"
            )
        out += [
            "",
            "- **Discussion rate**: fraction of images where this agent was questioned by the judge",
            "- **Update rate**: fraction of times agent changed top-1 country when questioned",
            "",
        ]
        if (plots_dir / "agent_initial_top1.png").exists():
            out += ["![Per-agent initial top-1 accuracy](plots/agent_initial_top1.png)",
                    "_Per-agent initial top-1 accuracy_", ""]

    # World maps (pure ground truth geography)
    out += _section_world_maps(out_dir)
    return out


def _section_world_maps(out_dir: Path) -> list[str]:
    """Geographic world-map subsection, gated on heatmap_metrics.json + PNGs."""
    heatmap = _load_json(out_dir / "heatmap_metrics.json")
    if not heatmap:
        return []
    maps = [
        ("world_map_accuracy.png", "Per-country true-positive rate (green) with false-positive outlines (red)."),
        ("world_map_f1.png", "Per-country F1, divergent around the run's macro-F1. Green = above average, red = below."),
        ("world_map_error_bias.png", "Per-country error bias (FP-FN)/(FP+FN). Red = over-predicted, blue = missed."),
    ]
    present = [(f, cap) for f, cap in maps if (out_dir / "plots" / f).exists()]
    if not present:
        return []
    out = ["### Geographic World Maps", ""]
    out.append(
        f"Per-country accuracy across {heatmap.get('n_countries_with_truth', 0)} "
        f"countries with truth. Macro-averaged TPR: **{_fmt_pct(heatmap.get('macro_avg_tpr'))}**."
    )
    out.append("")
    for fname, caption in present:
        out.append(f"![{caption}](plots/{fname})")
        out.append(f"_{caption}_")
        out.append("")
    return out


# ---------------------------------------------------------------------------
# 2. Approach Dynamics
# ---------------------------------------------------------------------------

def _section_dynamics(out_dir: Path, agents: dict | None, dynamics: dict | None) -> list[str]:
    plots_dir = out_dir / "plots"
    out = ["## 2. Approach Dynamics", ""]
    out.append(
        "Hub and spoke topology: five agents give independent assessments, then the "
        "judge hub interrogates specific agents with targeted questions over up to "
        "three discussion rounds. This section covers discussion rounds, plurality "
        "convergence, and per agent update behaviour."
    )
    out.append("")

    if agents and (plots_dir / "agent_discussion_rate.png").exists():
        out += ["![Per-agent discussion rate and update rate](plots/agent_discussion_rate.png)",
                "_Per-agent discussion participation and top-1 update rate when questioned_", ""]

    if not (dynamics and not dynamics.get("_error")):
        return out

    disc = dynamics.get("discussion", {}) or {}
    conv = dynamics.get("convergence", {}) or {}
    gtp = dynamics.get("gt_plurality", {}) or {}
    pau = dynamics.get("per_agent_update_behaviour", {}) or {}

    # Discussion activation
    out += ["### Discussion Activation", ""]
    out += [
        f"- Images with no discussion (judge accepted initial picks): "
        f"**{disc.get('n_no_discussion', 0)}**",
        f"- Images with discussion: **{disc.get('n_with_discussion', 0)}** "
        f"({_fmt_pct(disc.get('discussion_rate'))})",
        f"- Total judge questions: **{disc.get('total_questions', 0)}** "
        f"(answered: {disc.get('answered_questions', 0)})",
    ]
    mq = disc.get("mean_questions_per_discussion")
    if mq is not None:
        out.append(f"- Mean questions per discussed image: **{_fmt_float(mq, 1)}**")
    out.append("")

    rounds = disc.get("rounds_distribution", {}) or {}
    if rounds:
        out += ["**Discussion rounds distribution:**", ""]
        out += ["| Rounds | Count |", "|---|---|"]
        for k, v in sorted(rounds.items(), key=lambda x: int(x[0])):
            out.append(f"| {k} | {v} |")
        out.append("")

    # Convergence
    out += ["### Convergence (initial vs final plurality)", ""]
    out += [
        "| Metric | Value |",
        "|---|---|",
        f"| Initial plurality reached (>= 3/5) | {conv.get('init_plurality', 0)} |",
        f"| Final plurality reached (>= 3/5) | {conv.get('final_plurality', 0)} |",
        f"| Initial unanimous (5/5) | {conv.get('init_unanimous', 0)} |",
        f"| Final unanimous (5/5) | {conv.get('final_unanimous', 0)} |",
        f"| Same plurality top country in both phases | {conv.get('same_top_country', 0)} |",
        "",
    ]

    # GT-based plurality convergence
    if gtp.get("n_images_with_gt"):
        out += ["### Ground-Truth Plurality Convergence", ""]
        out += ["| Category | Count |", "|---|---|"]
        out += [
            f"| Plurality on ground truth (correct) | {gtp.get('plurality_correct', 0)} |",
            f"| Plurality on wrong country | {gtp.get('plurality_wrong', 0)} |",
            f"| No plurality (top <= 2/5) | {gtp.get('no_plurality', 0)} |",
            "",
        ]
        rate = gtp.get("landed_on_gt_rate")
        if rate is not None:
            out.append(
                f"Of the plurality-converged images, **{_fmt_pct(rate)}** landed on the "
                f"ground truth."
            )
            out.append("")

    # Per-agent update behaviour
    per = pau.get("per_agent", {}) or {}
    if per:
        out += ["### Per-agent Update Behaviour", ""]
        out.append(
            f"Across the {pau.get('n_answered_queries', 0)} answered queries with a "
            f"parseable initial and last pick, how did each agent shift relative to the "
            f"ground truth?"
        )
        out.append("")
        out += [
            "| Agent | Answered | Constructive | Destructive | Stayed OK | Stayed wrong | Lateral | Net truth |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for a in AGENT_NAMES:
            m = per.get(a, {}) or {}
            net = m.get("net_truth", 0)
            out.append(
                f"| {a} | {m.get('n_answered', 0)} | "
                f"{m.get('constructive', 0)} | {m.get('destructive', 0)} | "
                f"{m.get('stayed_correct', 0)} | {m.get('stayed_wrong', 0)} | "
                f"{m.get('wrong_to_wrong', 0)} | {net:+d} |"
            )
        out += [
            "",
            "- **Constructive**: initial pick wrong, response moved onto the ground truth",
            "- **Destructive**: initial pick correct, response moved away from the ground truth",
            "- **Stayed OK**: both initial and final equal the ground truth",
            "- **Stayed wrong**: both wrong on the same country",
            "- **Lateral**: both wrong on different countries",
            "- **Net truth**: constructive minus destructive",
            "",
        ]
    return out


# ---------------------------------------------------------------------------
# 3. LLM-as-Judge Verdicts
# ---------------------------------------------------------------------------

def _section_judge(out_dir: Path, judge: dict | None) -> list[str]:
    if not judge or judge.get("_error"):
        return []
    plots_dir = out_dir / "plots"
    out = ["## 3. LLM-as-Judge Verdicts", ""]
    out.append(
        f"Verdicts: {judge.get('n_with_verdict', 0)}/{judge.get('n_total_judge_files', 0)}"
    )
    errs = judge.get("errors", {}) or {}
    if errs:
        out.append(f"Errors: {', '.join(f'{k}={v}' for k, v in sorted(errs.items()))}")
    out.append("")

    strat = judge.get("judge_strategy_score") or {}
    synth = judge.get("judge_synthesis_quality") or {}
    conv = judge.get("discussion_convergence_score") or {}
    out += ["### System-level Scores", ""]
    out += ["| Metric | Mean | Median | n |", "|---|---|---|---|"]
    if strat.get("n"):
        out.append(
            f"| Judge question strategy | {_fmt_float(strat.get('mean'))} | "
            f"{_fmt_float(strat.get('median'))} | {strat.get('n', 0)} |"
        )
    if synth.get("n"):
        out.append(
            f"| Judge synthesis quality | {_fmt_float(synth.get('mean'))} | "
            f"{_fmt_float(synth.get('median'))} | {synth.get('n', 0)} |"
        )
    if conv.get("n"):
        out.append(
            f"| Discussion convergence | {_fmt_float(conv.get('mean'))} | "
            f"{_fmt_float(conv.get('median'))} | {conv.get('n', 0)} |"
        )
    out.append("")

    per = judge.get("per_agent", {}) or {}
    if per:
        out += ["### Per-agent Scores", ""]
        out += [
            "| Agent | n | Role adher. | Halluc. down | Visual cons. up | Calib. up | Q-relevance up | Resp. update up |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for a in AGENT_NAMES:
            m = per.get(a, {}) or {}

            def _mean(key: str) -> str:
                st = m.get(key) or {}
                v = st.get("mean")
                return f"{v:.2f}" if v is not None else "n/a"

            out.append(
                f"| {a} | {m.get('n', 0)} | "
                f"{_fmt_pct(m.get('role_adherence_rate', 0))} | "
                f"{_mean('hallucination_score')} | "
                f"{_mean('visual_consistency_score')} | "
                f"{_mean('confidence_calibration_score')} | "
                f"{_mean('question_relevance_score')} | "
                f"{_mean('response_update_quality')} |"
            )
        out.append("")
        for plot_file, caption in [
            ("judge_role_adherence.png", "Role adherence per agent"),
            ("judge_hallucination.png", "Mean hallucination score per agent (0 = clean, 1 = severe)"),
            ("judge_question_relevance.png", "Question relevance per agent"),
            ("judge_response_update.png", "Response update quality per agent"),
            ("judge_strategy.png", "Distribution of judge question strategy scores"),
        ]:
            if (plots_dir / plot_file).exists():
                out += [f"![{caption}](plots/{plot_file})", f"_{caption}_", ""]

    # Hallucination examples
    any_ex = any((per.get(a, {}) or {}).get("hallucination_examples") for a in AGENT_NAMES)
    if any_ex:
        out += ["### Hallucination Examples", ""]
        out.append("Concrete claims the judge flagged as not supported by the image.")
        out.append("")
        for agent in AGENT_NAMES:
            ex_list = (per.get(agent, {}) or {}).get("hallucination_examples") or []
            if not ex_list:
                continue
            out += [f"**{agent}:**", ""]
            out += ["| Image | Hallucinated claim |", "|---|---|"]
            for ex in ex_list:
                claim = (ex.get("example") or "").replace("|", r"\|")
                out.append(f"| {ex.get('image_id', '?')} | {claim} |")
            out.append("")
    return out


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    geo = _load_json(out_dir / "geo_metrics.json")
    agents = _load_json(out_dir / "agent_metrics.json")
    judge = _load_json(out_dir / "judge_summary.json")
    dynamics = _load_json(out_dir / "dynamics_metrics.json")

    lines: list[str] = []
    lines += _section_tldr(geo, agents, judge)
    lines += _section_ground_truth(out_dir, geo, agents)
    lines += _section_dynamics(out_dir, agents, dynamics)
    lines += _section_judge(out_dir, judge)

    if not (geo or agents or judge):
        lines.append("_No eval outputs found in this directory._")

    out_file = out_dir / "report.md"
    out_file.write_text("\n".join(lines))
    print(f"[report] wrote {out_file}")

    # HTML (best-effort)
    try:
        from eval_hubspoke.render_html import render as render_html
        render_html(out_dir)
    except Exception as e:
        print(f"[report] HTML rendering skipped: {e}")

    return out_file
