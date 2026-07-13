"""Compose a markdown report from all eval outputs.

Reads (when present):
  <out>/geo_metrics.json
  <out>/agent_metrics.json
  <out>/funnel_metrics.json
  <out>/dynamics_metrics.json
  <out>/judge_summary.json

Plus static PNGs under <out>/plots/.

Writes <out>/report.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.loader import AGENT_NAMES


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _km(x: float | None) -> str:
    return "n/a" if x is None else f"{x:,.0f} km"


def _score(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _section_world_maps(out_dir: Path) -> list[str]:
    """Geographic world-map section, gated on heatmap_metrics.json + PNGs."""
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
        f"countries with truth. Macro-averaged TPR: **{_pct(heatmap.get('macro_avg_tpr'))}**."
    )
    out.append("")
    for fname, caption in present:
        out.append(f"![{caption}](plots/{fname})")
        out.append(f"_{caption}_")
        out.append("")
    return out


class _FC:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    geo = _load_json(out_dir / "geo_metrics.json")
    agents = _load_json(out_dir / "agent_metrics.json")
    judge = _load_json(out_dir / "judge_summary.json")
    funnel = _load_json(out_dir / "funnel_metrics.json")
    dynamics = _load_json(out_dir / "dynamics_metrics.json")

    fc = _FC()
    lines: list[str] = []

    lines += ["# VLM Council Progressive Narrowing + Parallel Hypotheses Evaluation Report", ""]

    # ── TL;DR ──────────────────────────────────────────────────────────────
    tldr: list[str] = []
    if geo:
        hav = geo.get("haversine_km") or {}
        tldr.append(
            f"- **Country accuracy:** {_pct(geo.get('country_accuracy'))} "
            f"(n = {geo.get('n_total', 'n/a')})"
        )
        tldr.append(
            f"- **Haversine error:** median {_km(hav.get('median'))}, "
            f"mean {_km(hav.get('mean'))}"
        )
    if agents:
        tldr.append(
            f"- **Path split:** Path A (consensus) {agents.get('n_path_a', 'n/a')}, "
            f"Path B (no consensus) {agents.get('n_path_b', 'n/a')}"
        )
    if judge and not judge.get("_error"):
        rn_all = (judge.get("region_narrowing_quality") or {}).get("all") or {}
        if rn_all.get("n"):
            tldr.append(
                f"- **Region narrowing quality:** {_score(rn_all.get('mean'))} mean "
                f"({_score(rn_all.get('median'))} median)"
            )
    if tldr:
        lines += tldr + [""]

    # ══ 1. GROUND-TRUTH STATISTICS ═════════════════════════════════════════
    lines += ["## 1. Ground-Truth Statistics", ""]

    if geo:
        hav = geo.get("haversine_km") or {}
        lines += ["### Headline Metrics", ""]
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Country accuracy | {_pct(geo.get('country_accuracy'))} |",
            f"| Median haversine distance | {_km(hav.get('median'))} |",
            f"| Mean haversine distance | {_km(hav.get('mean'))} |",
            f"| N images | {geo.get('n_total', 'n/a')} |",
        ]
        if agents:
            lines += [
                f"| Path A (consensus) | {agents.get('n_path_a', 'n/a')} |",
                f"| Path B (no consensus) | {agents.get('n_path_b', 'n/a')} |",
            ]
        lines += [""]

        lines += ["### Geographic Bias", ""]
        nb = geo.get("north_bias_test") or {}
        eb = geo.get("east_bias_test") or {}
        lines += [
            f"- North bias: {nb.get('interpretation', 'n/a')}",
            f"- East bias:  {eb.get('interpretation', 'n/a')}",
            "",
        ]
        if (plots_dir / "error_distribution.png").exists():
            lines += [
                f"![Error distribution](plots/error_distribution.png)",
                f"_{fc.next()}. Lat/lng/haversine error distributions_",
                "",
            ]
        if (plots_dir / "bearing_rose.png").exists():
            lines += [
                f"![Bearing rose](plots/bearing_rose.png)",
                f"_{fc.next()}. Error bearing rose_",
                "",
            ]
        if (plots_dir / "confusion_matrix.png").exists():
            lines += [
                f"![Confusion matrix](plots/confusion_matrix.png)",
                f"_{fc.next()}. Top confusion pairs_",
                "",
            ]
        top_conf = geo.get("top_confusions") or []
        if top_conf:
            lines += ["**Top confusion pairs:**", ""]
            lines += ["| Truth | Predicted | Count |", "|-------|-----------|-------|"]
            for p in top_conf[:10]:
                lines.append(f"| {p['truth']} | {p['predicted']} | {p['count']} |")
            lines += [""]

    if agents:
        lines += ["### Per-agent Accuracy (Initial Round)", ""]
        ir = agents.get("initial_round") or {}
        lines += ["| Agent | Top-1 | Top-3 | Coverage | n |", "|-------|-------|-------|----------|---|"]
        for a in AGENT_NAMES:
            m = ir.get(a) or {}
            lines.append(
                f"| {a} | {_pct(m.get('top1_accuracy'))} | "
                f"{_pct(m.get('top3_hit_rate'))} | "
                f"{_pct(m.get('coverage'))} | {m.get('n', 0)} |"
            )
        lines += [""]
        if (plots_dir / "agent_top1.png").exists():
            lines += [
                f"![Agent top-1](plots/agent_top1.png)",
                f"_{fc.next()}. Per-agent top-1 accuracy_",
                "",
            ]
        if (plots_dir / "agent_calibration.png").exists():
            lines += [
                f"![Agent calibration](plots/agent_calibration.png)",
                f"_{fc.next()}. Confidence calibration by agent_",
                "",
            ]

    lines += _section_world_maps(out_dir)

    # ══ 2. APPROACH DYNAMICS ═══════════════════════════════════════════════
    if funnel or dynamics:
        lines += ["## 2. Approach Dynamics", ""]
        lines += [
            "Progressive Narrowing routes each image down Path A (region consensus, "
            "jump straight to the judge) or Path B (parallel hypotheses + a country "
            "re-assessment inside the confirmed region). This section traces where "
            "the truth country survives or is lost.",
            "",
        ]

    # Path A / Path B split + region + country narrowing funnel (from dynamics_metrics.json)
    if dynamics and not dynamics.get("_error"):
        ps = dynamics.get("path_split") or {}
        rf = dynamics.get("region_funnel") or {}
        cf = dynamics.get("country_funnel") or {}

        lines += ["### Path A / Path B Split", ""]
        lines += ["| Metric | Value |", "|--------|-------|"]
        lines += [
            f"| Path A (region consensus) | {ps.get('n_path_a', 0)} ({_pct(ps.get('path_a_rate'))}) |",
            f"| Path B (no consensus) | {ps.get('n_path_b', 0)} ({_pct(ps.get('path_b_rate'))}) |",
            f"| Region consensus reached | {ps.get('region_consensus', 0)} ({_pct(ps.get('region_consensus_rate'))}) |",
            "",
        ]

        if rf.get("n"):
            pa = rf.get("path_a") or {}
            pb = rf.get("path_b") or {}
            lines += ["### Region Narrowing Funnel", ""]
            lines += [
                "Does the confirmed region actually contain the ground-truth country?",
                "",
                "| Split | Region matches GT | n |",
                "|-------|-------------------|---|",
                f"| All | {_pct(rf.get('match_rate'))} | {rf.get('n', 0)} |",
                f"| Path A | {_pct(pa.get('match_rate'))} | {pa.get('n', 0)} |",
                f"| Path B | {_pct(pb.get('match_rate'))} | {pb.get('n', 0)} |",
                "",
            ]

        counts = cf.get("counts") or {}
        if counts:
            n_ai = cf.get("n_agent_images", 0)
            label = {
                "CONSTRUCTIVE": "Constructive (initial wrong, re-assessment moved onto GT)",
                "DESTRUCTIVE": "Destructive (initial correct, re-assessment moved off GT)",
                "STAYED_CORRECT": "Stayed correct (both initial and re-assessment on GT)",
                "STAYED_WRONG": "Stayed wrong (both wrong, same country)",
                "WRONG_TO_WRONG": "Lateral (both wrong, different countries)",
            }
            lines += ["### Country Narrowing Funnel (Path B initial to re-assessment)", ""]
            lines += [
                f"Per-agent initial pick vs country re-assessment pick "
                f"(n = {n_ai} Path B agent images):",
                "",
                "| Category | Count | Share |",
                "|----------|-------|-------|",
            ]
            for key in ["CONSTRUCTIVE", "DESTRUCTIVE", "STAYED_CORRECT", "STAYED_WRONG", "WRONG_TO_WRONG"]:
                if key in counts:
                    c = counts[key]
                    share = _pct(c / n_ai) if n_ai else "n/a"
                    lines.append(f"| {label[key]} | {c} | {share} |")
            lines += [""]

    # Pipeline funnel + oracle ceilings + agreement + Path comparison + severity
    if funnel:
        fn_data = funnel.get("funnel") or {}
        ceilings = funnel.get("oracle_ceilings") or {}
        paths = funnel.get("path_comparison") or {}
        severity = funnel.get("severity") or {}
        agree = funnel.get("agreement_curve") or {}

        lines += ["### Pipeline Funnel", ""]
        lines += ["Where does the truth country get lost?", ""]

        stages = fn_data.get("stages") or []
        if stages:
            lines += ["| Stage | Description | Cumul. survival | Conditional |",
                      "|-------|-------------|----------------|-------------|"]
            for s in stages:
                cum = s.get("cumulative_survival") or {}
                cond = s.get("conditional_on_prev") or {}
                lines.append(
                    f"| {s['code']} | {s['description']} | "
                    f"{_pct(cum.get('rate'))} ({cum.get('correct', 0)}/{cum.get('n', 0)}) | "
                    f"{_pct(cond.get('rate'))} |"
                )
            lines += [""]

        bottleneck = fn_data.get("bottleneck")
        if bottleneck:
            lines += [
                f"> **Bottleneck:** {bottleneck['stage_code']}, {bottleneck['description']}  "
                f"(conditional rate {_pct(bottleneck['conditional_rate'])}, "
                f"95% CI [{_pct(bottleneck['ci_low'])}, {_pct(bottleneck['ci_high'])}])",
                "",
            ]

        if (plots_dir / "funnel.png").exists():
            lines += [
                f"![Funnel](plots/funnel.png)",
                f"_{fc.next()}. Cumulative truth-survival through the PN pipeline_",
                "",
            ]

        lines += ["#### Oracle Ceilings", ""]
        lines += ["| Scenario | Accuracy |", "|----------|----------|"]
        for key, label in [
            ("actual", "Actual"),
            ("majority_vote_baseline", "Majority-vote baseline"),
            ("oracle_region", "Oracle region (perfect region step)"),
            ("oracle_pool", "Oracle pool (truth always in hypothesis pool)"),
            ("oracle_decision", "Oracle decision (perfect final judge)"),
        ]:
            blk = ceilings.get(key) or {}
            lines.append(f"| {label} | {_pct(blk.get('rate'))} |")
        lines += [""]

        if (plots_dir / "oracle_ceilings.png").exists():
            lines += [
                f"![Oracle ceilings](plots/oracle_ceilings.png)",
                f"_{fc.next()}. Counterfactual accuracy if each stage were perfect_",
                "",
            ]

        if agree.get("levels"):
            lines += ["#### Agreement vs. Accuracy", ""]
            lines += ["| Agents agree | n | Accuracy |", "|-------------|---|----------|"]
            for lv in agree.get("levels", []):
                lines.append(f"| {lv['agreement']} | {lv.get('n', 0)} | {_pct(lv.get('rate'))} |")
            lines += [""]

        if (plots_dir / "agreement_curve.png").exists():
            lines += [
                f"![Agreement curve](plots/agreement_curve.png)",
                f"_{fc.next()}. Final accuracy by initial-round agent agreement level_",
                "",
            ]

        pa = paths.get("path_a") or {}
        pb = paths.get("path_b") or {}
        if pa.get("n") or pb.get("n"):
            lines += ["#### Path A vs. Path B", ""]
            lines += ["| Metric | Path A | Path B |", "|--------|--------|--------|"]
            lines += [
                f"| n | {pa.get('n', 0)} | {pb.get('n', 0)} |",
                f"| Accuracy | {_pct(pa.get('rate'))} | {_pct(pb.get('rate'))} |",
                f"| Median haversine | {_km(pa.get('median_haversine_km'))} | {_km(pb.get('median_haversine_km'))} |",
                f"| Truth in pool | {_pct(pa.get('truth_in_pool_rate'))} | {_pct(pb.get('truth_in_pool_rate'))} |",
            ]
            lines += [""]

        if severity:
            lines += ["#### Error Severity", ""]
            n_wrong = severity.get("n_wrong", 0)
            lines += [
                f"- Near miss (< 500 km): {severity.get('near_miss_count', 0)} / {n_wrong}",
                f"- Same region, wrong country: {severity.get('same_region_wrong_count', 0)} / {n_wrong}",
                f"- Wrong region: {severity.get('wrong_region_count', 0)} / {n_wrong}",
                "",
            ]

    # ══ 3. LLM-AS-JUDGE VERDICTS ═══════════════════════════════════════════
    if judge and not judge.get("_error"):
        lines += ["## 3. LLM-as-Judge Verdicts", ""]
        lines += [
            f"Verdicts: {judge.get('n_with_verdict', 0)}/{judge.get('n_total_judge_files', 0)}  "
            f"Constructive synthesis: {_pct(judge.get('constructive_synthesis_rate'))}",
            "",
        ]

        # PN-specific scores
        rn = judge.get("region_narrowing_quality") or {}
        hp = judge.get("hypothesis_pool_quality") or {}
        lines += ["### Progressive Narrowing Scores", ""]
        lines += [
            "| Metric | Mean | Median | n |",
            "|--------|------|--------|---|",
        ]
        for label, st in [
            ("Region narrowing, all", rn.get("all") or {}),
            ("Region narrowing, Path A", rn.get("path_a") or {}),
            ("Region narrowing, Path B", rn.get("path_b") or {}),
            ("Hypothesis pool, all", hp.get("all") or {}),
            ("Hypothesis pool, truth in pool", hp.get("when_truth_in_pool") or {}),
            ("Hypothesis pool, truth NOT in pool", hp.get("when_truth_not_in_pool") or {}),
        ]:
            lines.append(
                f"| {label} | {_score(st.get('mean'))} | "
                f"{_score(st.get('median'))} | {st.get('n', 0)} |"
            )
        lines += [""]
        if (plots_dir / "judge_pn_scores.png").exists():
            lines += [
                f"![PN scores](plots/judge_pn_scores.png)",
                f"_{fc.next()}. Region narrowing quality and hypothesis pool quality_",
                "",
            ]

        # Per-agent judge table
        lines += ["### Per-agent Judge Scores", ""]
        lines += [
            "| Agent | Role adherence | Hall. (down) | Visual cons. | Conf. calib. |",
            "|-------|----------------|--------------|--------------|--------------|",
        ]
        for a in AGENT_NAMES:
            pa = (judge.get("per_agent") or {}).get(a) or {}
            ra = pa.get("role_adherence_rate")
            hall = (pa.get("hallucination_score") or {}).get("mean")
            vis = (pa.get("visual_consistency_score") or {}).get("mean")
            calib = (pa.get("confidence_calibration_score") or {}).get("mean")
            lines.append(
                f"| {a} | {_pct(ra)} | {_score(hall)} | "
                f"{_score(vis)} | {_score(calib)} |"
            )
        lines += [""]
        for plot_file, caption in [
            ("judge_role_adherence.png", "Role adherence per agent"),
            ("judge_quality.png", "Argumentative quality histogram"),
            ("judge_hallucination.png", "Hallucination score per agent"),
            ("judge_visual_consistency.png", "Visual consistency per agent"),
            ("judge_confidence_calibration.png", "Confidence calibration per agent"),
        ]:
            if (plots_dir / plot_file).exists():
                lines += [
                    f"![{caption}](plots/{plot_file})",
                    f"_{fc.next()}. {caption}_",
                    "",
                ]

        # Hallucination examples
        lines += ["### Hallucination Examples", ""]
        for a in AGENT_NAMES:
            pa = (judge.get("per_agent") or {}).get(a) or {}
            examples = pa.get("hallucination_examples") or []
            if examples:
                lines += [f"**{a}:**", ""]
                lines += ["| Image | Score | Claim |", "|-------|-------|-------|"]
                for ex in examples[:5]:
                    sc = ex.get("score")
                    lines.append(
                        f"| {ex.get('image_id', '')} | "
                        f"{_score(sc) if sc is not None else 'n/a'} | {ex.get('example', '')} |"
                    )
                lines += [""]

    out_file = out_dir / "report.md"
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    print(f"[report] wrote {out_file}")

    from eval.render_html import render as render_html
    render_html(out_dir)
