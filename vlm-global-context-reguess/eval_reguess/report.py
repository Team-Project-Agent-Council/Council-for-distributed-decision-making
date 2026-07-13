"""Compose a single markdown + HTML report from all eval_reguess outputs.

Reads (when present):
  <out>/geo_metrics.json
  <out>/agent_metrics.json
  <out>/judge_summary.json
  <out>/heatmap_metrics.json
  <out>/dynamics_metrics.json

Writes <out>/report.md and calls eval_reguess.render_html.render(out_dir).

Section order:
  TL;DR (accuracy, haversine, judge synthesis if present)
  1. Ground-Truth Statistics (headline metrics, geo bias, per-agent GT accuracy, world maps)
  2. Approach Dynamics (Round 1 vs Round 2 shift, constructive vs destructive revision)
  3. LLM-as-Judge Verdicts (judge scores, hallucination examples)
"""

from __future__ import annotations

import json
from pathlib import Path

from eval_reguess.loader import AGENT_NAMES


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": f"failed to load {path}: {e}"}


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1%}"


def _fmt_km(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:,.0f} km"


def _fmt_f(x: float | None, decimals: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{decimals}f}"


class _FigCounter:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


def _fig(fc: _FigCounter, caption: str) -> str:
    return f"_Figure {fc.next()}: {caption}_"


# ── TL;DR ────────────────────────────────────────────────────────────────


def _section_tldr(geo: dict | None, judge: dict | None) -> list[str]:
    out = ["# VLM Council Global Context Reguess Evaluation Report", ""]
    if geo and not geo.get("_error"):
        acc = geo.get("country_accuracy", 0.0)
        hav = geo.get("haversine_km", {}) or {}
        out.append(f"- **Images evaluated:** {geo.get('n_total', 0)}")
        out.append(f"- **Country accuracy:** {_fmt_pct(acc)}")
        if hav.get("n"):
            out.append(
                f"- **Haversine error:** mean {_fmt_km(hav.get('mean', 0))}, "
                f"median {_fmt_km(hav.get('median', 0))}, "
                f"p90 {_fmt_km(hav.get('p90', 0))}"
            )
    if judge and not judge.get("_error"):
        mr2 = judge.get("mean_round2_improvement")
        if mr2 is not None:
            out.append(f"- **Mean round2 improvement (judge):** {_fmt_f(mr2, 3)}")
        jsq = judge.get("judge_synthesis_quality") or {}
        if jsq.get("n"):
            out.append(
                f"- **Judge synthesis quality:** mean {_fmt_f(jsq.get('mean'), 3)} "
                f"(median {_fmt_f(jsq.get('median'), 3)}, n={jsq['n']})"
            )
    out.append("")
    return out


# ── 1. Ground-Truth Statistics ───────────────────────────────────────────


def _section_ground_truth(geo: dict | None, agents: dict | None, out_dir: Path,
                          fc: _FigCounter) -> list[str]:
    out = ["## 1. Ground-Truth Statistics", ""]

    if geo and not geo.get("_error"):
        hav = geo.get("haversine_km", {}) or {}
        out.append("### Headline Metrics")
        out.append("")
        out.append("| Metric | Value |")
        out.append("|---|---|")
        out.append(f"| Country accuracy | {_fmt_pct(geo.get('country_accuracy'))} |")
        out.append(f"| Median haversine | {_fmt_km(hav.get('median'))} |")
        out.append(f"| Mean haversine | {_fmt_km(hav.get('mean'))} |")
        out.append(f"| N images | {geo.get('n_total', 'n/a')} |")
        out.append("")

        out.append("### Geo-spatial Bias")
        out.append("")
        nb = geo.get("north_bias_test") or {}
        eb = geo.get("east_bias_test") or {}
        out.append(f"- North/south bias: {nb.get('interpretation', 'n/a')}")
        out.append(f"- East/west bias: {eb.get('interpretation', 'n/a')}")
        quads = geo.get("quadrants") or {}
        if quads:
            ordered = ", ".join(f"{k}={v}" for k, v in sorted(quads.items()))
            out.append(f"- Error quadrants: {ordered}")
        abs_lat = geo.get("abs_lat_error_deg") or {}
        abs_lng = geo.get("abs_lng_error_deg") or {}
        if abs_lat.get("n") and abs_lng.get("n"):
            out.append(
                f"- Mean absolute lat error: {abs_lat.get('mean', 0):.2f} deg, "
                f"mean absolute lng error: {abs_lng.get('mean', 0):.2f} deg"
            )
        out.append("")
        out.append("![Error distribution](plots/error_distribution.png)")
        out.append("")
        out.append(_fig(fc, "Latitude and longitude error and haversine distribution."))
        out.append("")
        out.append("![Bearing rose](plots/bearing_rose.png)")
        out.append("")
        out.append(_fig(fc, "Bearing of prediction errors (truth to prediction)."))
        out.append("")

        confs = geo.get("top_confusions") or []
        if confs:
            out.append("### Top Confusion Pairs")
            out.append("")
            out.append("| Truth | Predicted | Count |")
            out.append("|---|---|---|")
            for row in confs[:15]:
                out.append(f"| {row['truth']} | {row['predicted']} | {row['count']} |")
            out.append("")
            out.append("![Confusion matrix](plots/confusion_matrix.png)")
            out.append("")
            out.append(_fig(fc, "Top 15 confusion pairs (truth to predicted)."))
            out.append("")

    # Per-agent ground-truth accuracy (R1 and R2)
    if agents and not agents.get("_error"):
        r1 = agents.get("round_1") or {}
        r2 = agents.get("round_2") or {}
        if r1:
            out.append("### Per-agent Ground-Truth Accuracy (Round 1)")
            out.append("")
            out.append("| Agent | n | Top-1 | Top-3 | Coverage |")
            out.append("|---|---|---|---|---|")
            for a in AGENT_NAMES:
                m = r1.get(a) or {}
                out.append(
                    f"| {a} | {m.get('n', 0)} | "
                    f"{_fmt_pct(m.get('top1_accuracy'))} | "
                    f"{_fmt_pct(m.get('top3_hit_rate'))} | "
                    f"{_fmt_pct(m.get('coverage'))} |"
                )
            out.append("")
        if r2:
            out.append("### Per-agent Ground-Truth Accuracy (Round 2)")
            out.append("")
            out.append("| Agent | n | Top-1 | Top-3 | Coverage |")
            out.append("|---|---|---|---|---|")
            for a in AGENT_NAMES:
                m = r2.get(a) or {}
                out.append(
                    f"| {a} | {m.get('n', 0)} | "
                    f"{_fmt_pct(m.get('top1_accuracy'))} | "
                    f"{_fmt_pct(m.get('top3_hit_rate'))} | "
                    f"{_fmt_pct(m.get('coverage'))} |"
                )
            out.append("")

    # World maps (pure ground-truth geographic plots)
    heatmap = _load_json(out_dir / "heatmap_metrics.json")
    maps = [
        ("world_map_accuracy.png", "Per-country true-positive rate (green) with false-positive outlines (red)."),
        ("world_map_f1.png", "Per-country F1, divergent around the run's macro-F1. Green above average, red below."),
        ("world_map_error_bias.png", "Per-country error bias (FP-FN)/(FP+FN). Red over-predicted, blue missed."),
    ]
    present = [(f, cap) for f, cap in maps if (out_dir / "plots" / f).exists()]
    if heatmap and not heatmap.get("_error") and present:
        out.append("### Geographic World Maps")
        out.append("")
        out.append(
            f"Per-country accuracy across {heatmap.get('n_countries_with_truth', 0)} "
            f"countries with truth. Macro-averaged TPR: "
            f"**{_fmt_pct(heatmap.get('macro_avg_tpr'))}**."
        )
        out.append("")
        for fname, caption in present:
            out.append(f"![{caption}](plots/{fname})")
            out.append(f"_{caption}_")
            out.append("")

    return out


# ── 2. Approach Dynamics ─────────────────────────────────────────────────


def _section_dynamics(agents: dict | None, dynamics: dict | None,
                      fc: _FigCounter) -> list[str]:
    if not (agents or dynamics):
        return []
    out = ["## 2. Approach Dynamics", ""]
    out.append(
        "Each image runs Round 1 (independent per-agent guesses) then Round 2 "
        "(each agent re-guesses with the full set of Round 1 assessments as "
        "global context). The dynamics below track how that context shifted "
        "agents between the two rounds and whether those shifts moved the "
        "council toward or away from the ground truth."
    )
    out.append("")

    # Agent round comparison (R1 vs R2 top-1, change rate, confidence shift)
    if agents and not agents.get("_error"):
        r1 = agents.get("round_1") or {}
        r2 = agents.get("round_2") or {}
        chg = agents.get("change_metrics") or {}
        if r1 or r2:
            out.append("### Agent Round Comparison")
            out.append("")
            out.append(
                "Per-agent top-1 accuracy in Round 1 vs Round 2. Change rate is "
                "the fraction of images where the top-1 country changed. "
                "Confidence shift is the fraction where Round 2 confidence was "
                "higher than Round 1."
            )
            out.append("")
            out.append("| Agent | R1 Top-1 | R2 Top-1 | Change rate | Conf. shift |")
            out.append("|---|---|---|---|---|")
            for a in AGENT_NAMES:
                r1m = r1.get(a) or {}
                r2m = r2.get(a) or {}
                cm = chg.get(a) or {}
                out.append(
                    f"| {a} | {_fmt_pct(r1m.get('top1_accuracy'))} | "
                    f"{_fmt_pct(r2m.get('top1_accuracy'))} | "
                    f"{_fmt_pct(cm.get('change_rate'))} | "
                    f"{_fmt_pct(cm.get('confidence_shift_rate'))} |"
                )
            out.append("")
            out.append("![Agent R1 vs R2 top-1](plots/agent_top1_r1_vs_r2.png)")
            out.append("")
            out.append(_fig(fc, "Per-agent top-1 accuracy: Round 1 vs Round 2."))
            out.append("")

    if dynamics and not dynamics.get("_error"):
        rm = dynamics.get("round_movement") or {}
        ad = dynamics.get("agreement_dynamics") or {}
        sc = dynamics.get("shift_classification") or {}
        pas = dynamics.get("per_agent_shift") or {}
        conv = dynamics.get("r2_convergence") or {}

        # Round movement summary
        if rm:
            out.append("### Round 1 to Round 2 Movement")
            out.append("")
            out.append("| Metric | Value |")
            out.append("|---|---|")
            out.append(f"| Comparable agent-image pairs | {rm.get('comparable_pairs', 0)} |")
            out.append(
                f"| Top pick changed in R2 | {rm.get('changed', 0)} "
                f"({_fmt_pct(rm.get('change_rate'))}) |"
            )
            out.append(f"| Top pick unchanged in R2 | {rm.get('stayed', 0)} |")
            out.append(f"| Confidence up in R2 | {rm.get('conf_up', 0)} |")
            out.append(f"| Confidence down in R2 | {rm.get('conf_down', 0)} |")
            out.append("")

        # Agreement dynamics
        if ad:
            out.append("### Agreement Dynamics (R1 plurality vs R2 plurality)")
            out.append("")
            out.append("| Transition | Count |")
            out.append("|---|---|")
            out.append(f"| R1 unanimous (5/5 same country) | {ad.get('r1_unanimous', 0)} |")
            out.append(f"| R2 unanimous (5/5 same country) | {ad.get('r2_unanimous', 0)} |")
            out.append(f"| R1 split to R2 unanimous (context built consensus) | {ad.get('became_unanimous', 0)} |")
            out.append(f"| R1 unanimous to R2 split (context broke consensus) | {ad.get('lost_unanimous', 0)} |")
            out.append(f"| R1 sub-plurality to R2 plurality (context built majority) | {ad.get('became_plurality', 0)} |")
            out.append(f"| R1 plurality to R2 sub-plurality (context broke majority) | {ad.get('lost_plurality', 0)} |")
            out.append(f"| Same plurality top country in both rounds | {ad.get('same_top_country', 0)} |")
            out.append("")

        # Constructive vs destructive shift classification
        counts = sc.get("counts") or {}
        if counts:
            n_pairs = sc.get("n_pairs", 0)
            label = {
                "CONSTRUCTIVE": "Constructive (R1 wrong, R2 corrected onto GT)",
                "DESTRUCTIVE": "Destructive (R1 correct, R2 moved away from GT)",
                "STAYED_CORRECT": "Stayed correct (both rounds on GT)",
                "STAYED_WRONG": "Stayed wrong (both rounds on the same wrong country)",
                "WRONG_TO_WRONG": "Lateral (both rounds wrong, different countries)",
            }
            out.append("### Constructive vs Destructive R1 to R2 Shifts")
            out.append("")
            out.append(
                f"Classification of each comparable agent-image pair "
                f"(n = {n_pairs} pairs with ground truth)."
            )
            out.append("")
            out.append("| Category | Count | Share |")
            out.append("|---|---|---|")
            for key in ["CONSTRUCTIVE", "DESTRUCTIVE", "STAYED_CORRECT",
                        "STAYED_WRONG", "WRONG_TO_WRONG"]:
                if key in counts:
                    c = counts[key]
                    share = _fmt_pct(c / n_pairs) if n_pairs else "n/a"
                    out.append(f"| {label[key]} | {c} | {share} |")
            out.append("")
            decisive = counts.get("CONSTRUCTIVE", 0) + counts.get("DESTRUCTIVE", 0)
            if decisive:
                c = counts.get("CONSTRUCTIVE", 0)
                d = counts.get("DESTRUCTIVE", 0)
                out.append(
                    f"Among the {decisive} pairs where exactly one round had the GT: "
                    f"constructive {c}/{decisive} ({_fmt_pct(c / decisive)}), "
                    f"destructive {d}/{decisive} ({_fmt_pct(d / decisive)})."
                )
                out.append("")

        # Per-agent shift matrix
        if pas:
            out.append("### Per-agent R1 to R2 Shift Matrix")
            out.append("")
            out.append(
                "Net truth is constructive minus destructive shifts. Positive means "
                "context pulled the agent toward the ground truth; negative means away."
            )
            out.append("")
            out.append("| Agent | n | Constr | Destr | StayOK | StayX | Lateral | Net truth | R1 acc | R2 acc | Delta |")
            out.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for a in AGENT_NAMES:
                m = pas.get(a) or {}
                net = m.get("net_truth", 0)
                net_str = f"+{net}" if net >= 0 else str(net)
                out.append(
                    f"| {a} | {m.get('n', 0)} | {m.get('constructive', 0)} | "
                    f"{m.get('destructive', 0)} | {m.get('stayed_correct', 0)} | "
                    f"{m.get('stayed_wrong', 0)} | {m.get('wrong_shift', 0)} | "
                    f"{net_str} | {_fmt_pct(m.get('r1_accuracy'))} | "
                    f"{_fmt_pct(m.get('r2_accuracy'))} | {_fmt_pct(m.get('accuracy_delta'))} |"
                )
            out.append("")

        # R2 convergence per image
        if conv and conv.get("n_images"):
            out.append("### Round 2 Convergence (per image, plurality at least 3/5)")
            out.append("")
            n = conv.get("n_images", 0)
            out.append("| Outcome | Count | Share |")
            out.append("|---|---|---|")
            out.append(
                f"| Plurality on GT (correct) | {conv.get('plurality_correct', 0)} | "
                f"{_fmt_pct(conv.get('plurality_correct_rate'))} |"
            )
            pw = conv.get("plurality_wrong", 0)
            out.append(f"| Plurality on wrong country | {pw} | {_fmt_pct(pw / n) if n else 'n/a'} |")
            sp = conv.get("no_plurality", 0)
            out.append(f"| No plurality (top at most 2/5) | {sp} | {_fmt_pct(sp / n) if n else 'n/a'} |")
            out.append("")

    return out


# ── 3. LLM-as-Judge Verdicts ─────────────────────────────────────────────


def _section_judge(judge: dict | None, fc: _FigCounter) -> list[str]:
    if not judge or judge.get("_error"):
        return []
    out = ["## 3. LLM-as-Judge Verdicts", ""]
    out.append(
        f"- Verdicts: {judge.get('n_with_verdict', 0)} / {judge.get('n_total', 0)}"
    )
    errs = judge.get("errors") or {}
    if errs:
        out.append(f"- Errors: {', '.join(f'{k}={v}' for k, v in sorted(errs.items()))}")
    mr2 = judge.get("mean_round2_improvement")
    if mr2 is not None:
        out.append(
            f"- Mean round2 improvement: **{_fmt_f(mr2, 3)}** "
            f"(1=genuine synthesis, 0=rubber-stamp)"
        )
    out.append("")

    per = judge.get("per_agent") or {}
    if per:
        out.append("### Per-agent Quantitative Scores")
        out.append("")
        out.append("| Agent | n | Role adher. | Hallucination low is better | Visual cons. high is better | Calibration high is better | R2 improvement high is better |")
        out.append("|---|---|---|---|---|---|---|")
        for a in AGENT_NAMES:
            m = per.get(a) or {}
            hall = m.get("hallucination_score") or {}
            vis = m.get("visual_consistency_score") or {}
            cal = m.get("confidence_calibration_score") or {}
            r2i = m.get("round2_improvement") or {}
            out.append(
                f"| {a} | {m.get('n', 0)} | "
                f"{_fmt_pct(m.get('role_adherence_rate'))} | "
                f"{_fmt_f(hall.get('mean'))} | "
                f"{_fmt_f(vis.get('mean'))} | "
                f"{_fmt_f(cal.get('mean'))} | "
                f"{_fmt_f(r2i.get('mean'))} |"
            )
        out.append("")

    jsq = judge.get("judge_synthesis_quality") or {}
    if jsq.get("n"):
        out.append(f"### Judge Synthesis Quality (n={jsq['n']})")
        out.append("")
        out.append(
            f"Mean: **{_fmt_f(jsq.get('mean'), 3)}**, median: {_fmt_f(jsq.get('median'), 3)}, "
            f"stdev: {_fmt_f(jsq.get('stdev'), 3)}"
        )
        out.append("")

    out.append("![Role adherence](plots/judge_role_adherence.png)")
    out.append("")
    out.append(_fig(fc, "Per-agent role adherence rate."))
    out.append("")
    out.append("![Hallucination](plots/judge_hallucination.png)")
    out.append("")
    out.append(_fig(fc, "Per-agent mean hallucination score (0 clean, 1 severe)."))
    out.append("")
    out.append("![Round 2 improvement](plots/judge_round2_improvement.png)")
    out.append("")
    out.append(_fig(fc, "Per-agent mean Round 2 improvement score (reference line at 0.5)."))
    out.append("")
    out.append("![Judge synthesis](plots/judge_synthesis.png)")
    out.append("")
    out.append(_fig(fc, "Judge synthesis quality distribution."))
    out.append("")

    # Hallucination examples
    any_examples = any(
        (per.get(a) or {}).get("hallucination_examples") for a in AGENT_NAMES
    )
    if any_examples:
        out.append("### Hallucination Examples")
        out.append("")
        out.append(
            "Concrete claims flagged by the judge as not supported by the image. "
            "Up to 10 examples per agent."
        )
        out.append("")
        for agent in AGENT_NAMES:
            examples = (per.get(agent) or {}).get("hallucination_examples") or []
            if not examples:
                continue
            out.append(f"**{agent}:**")
            out.append("")
            out.append("| Image | Score | Hallucinated claim |")
            out.append("|---|---|---|")
            for ex in examples:
                score = ex.get("score")
                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
                claim = (ex.get("example") or "").replace("|", r"\|")
                out.append(f"| {ex.get('image_id', '?')} | {score_str} | {claim} |")
            out.append("")
    return out


def run(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    geo = _load_json(out_dir / "geo_metrics.json")
    agents = _load_json(out_dir / "agent_metrics.json")
    judge = _load_json(out_dir / "judge_summary.json")
    dynamics = _load_json(out_dir / "dynamics_metrics.json")

    fc = _FigCounter()
    lines: list[str] = []
    lines += _section_tldr(geo, judge)
    lines += _section_ground_truth(geo, agents, out_dir, fc)
    lines += _section_dynamics(agents, dynamics, fc)
    lines += _section_judge(judge, fc)

    if not (geo or agents or judge):
        lines.append("_No eval outputs found in this directory._")

    out_file = out_dir / "report.md"
    out_file.write_text("\n".join(lines))
    print(f"[report] wrote {out_file}")

    # Render HTML (best-effort)
    try:
        from eval_reguess.render_html import render as render_html
        render_html(out_dir)
    except Exception as e:
        print(f"[report] HTML rendering skipped: {e}")

    return out_file
