"""Markdown + HTML report for the Debate approach."""

from __future__ import annotations

import json
from pathlib import Path

from eval_debate.loader import AGENT_NAMES


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}


def _pct(x) -> str:
    try:
        return f"{float(x):.1%}" if x is not None else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def _km(x) -> str:
    try:
        return f"{float(x):,.0f} km" if x is not None else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def _score(x) -> str:
    try:
        return f"{float(x):.3f}" if x is not None else ", "
    except (TypeError, ValueError):
        return ", "


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
    macro = heatmap.get("macro_avg_tpr")
    macro_str = f"{float(macro):.1%}" if macro is not None else "n/a"
    out = ["### Geographic World Maps", ""]
    out.append(
        f"Per-country accuracy across {heatmap.get('n_countries_with_truth', 0)} "
        f"countries with truth. Macro-averaged TPR: **{macro_str}**."
    )
    out.append("")
    for fname, caption in present:
        out.append(f"![{caption}](plots/{fname})")
        out.append(f"_{caption}_")
        out.append("")
    return out


def run(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    geo = _load_json(out_dir / "geo_metrics.json")
    agents = _load_json(out_dir / "agent_metrics.json")
    judge = _load_json(out_dir / "judge_summary.json")
    debate_stats = _load_json(out_dir / "debate_stats.json")
    dynamics = _load_json(out_dir / "dynamics_metrics.json")

    lines: list[str] = ["# VLM Council Debate Evaluation Report", ""]

    # ── TL;DR ────────────────────────────────────────────────────────────────
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
    if judge and not judge.get("_error"):
        syn = judge.get("judge_synthesis_quality") or {}
        if syn.get("n"):
            tldr.append(
                f"- **Judge synthesis quality:** {_score(syn.get('mean'))} mean "
                f"({_score(syn.get('median'))} median)"
            )
    if debate_stats:
        tldr.append(f"- **Debate rate:** {_pct(debate_stats.get('debate_rate'))}")
    if tldr:
        lines += tldr + [""]

    # ══ 1. GROUND-TRUTH STATISTICS ═══════════════════════════════════════════
    lines += ["## 1. Ground-Truth Statistics", ""]

    if geo:
        hav = geo.get("haversine_km") or {}
        lines += ["### Headline Metrics", ""]
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Country accuracy | {_pct(geo.get('country_accuracy'))} |",
            f"| Median haversine | {_km(hav.get('median'))} |",
            f"| Mean haversine | {_km(hav.get('mean'))} |",
            f"| N images | {geo.get('n_total', 'n/a')} |",
            "",
        ]

        lines += ["### Geographic Bias", ""]
        nb = geo.get("north_bias_test") or {}
        eb = geo.get("east_bias_test") or {}
        lines += [
            f"- North bias: {nb.get('interpretation', 'n/a')}",
            f"- East bias:  {eb.get('interpretation', 'n/a')}",
            "",
        ]
        for plot_file, caption in [
            ("error_distribution.png", "Lat/lng/haversine error distributions"),
            ("bearing_rose.png", "Direction of prediction errors"),
            ("confusion_matrix.png", "Top confusion pairs (truth → predicted)"),
        ]:
            if (plots_dir / plot_file).exists():
                lines += [f"![{caption}](plots/{plot_file})", f"_{caption}_", ""]

    if agents:
        ir = agents.get("initial_round") or {}
        lines += ["### Per-agent Accuracy (Initial Round)", ""]
        lines += ["| Agent | Top-1 | Top-3 | Coverage | n |",
                  "|-------|-------|-------|----------|---|"]
        for a in AGENT_NAMES:
            m = ir.get(a) or {}
            lines.append(
                f"| {a} | {_pct(m.get('top1_accuracy'))} | "
                f"{_pct(m.get('top3_hit_rate'))} | "
                f"{_pct(m.get('coverage'))} | {m.get('n', 0)} |"
            )
        lines += [""]
        if (plots_dir / "agent_top1.png").exists():
            lines += ["![Per-agent top-1 accuracy](plots/agent_top1.png)",
                      "_Per-agent top-1 accuracy_", ""]

    lines += _section_world_maps(out_dir)

    # ══ 2. APPROACH DYNAMICS ═════════════════════════════════════════════════
    if debate_stats or agents:
        lines += ["## 2. Approach Dynamics", ""]

    if debate_stats:
        lines += ["### Debate Activation", ""]
        me = debate_stats.get("mean_exchanges_per_pairing")
        me_str = f"{float(me):.2f}" if isinstance(me, (int, float)) else "n/a"
        lines += [
            f"- Images with no debate: **{debate_stats.get('n_no_debate', 0)}** "
            f"({_pct(1 - debate_stats.get('debate_rate', 0))})",
            f"- Images with ≥1 debate round: **{debate_stats.get('n_debate_1plus', 0)}** "
            f"({_pct(debate_stats.get('debate_rate', 0))})",
            f"- Mean exchanges per pairing: **{me_str}**",
            "",
        ]
        rounds = debate_stats.get("rounds_distribution", {})
        if rounds:
            lines += ["**Debate rounds distribution:**", ""]
            lines += ["| Rounds | Count |", "|--------|-------|"]
            for k, v in sorted(rounds.items(), key=lambda x: int(x[0])):
                lines.append(f"| {k} | {v} |")
            lines += [""]
        terms = debate_stats.get("termination_reasons", {})
        if terms:
            # Keep only the clean enum-style codes (short, snake_case); the raw
            # free-text LLM rationales are collapsed into a single "other" bucket.
            _CLEAN = {"consensus", "weak_dissent", "stalemate", "max_rounds_reached"}
            clean = {k: v for k, v in terms.items() if k in _CLEAN}
            other = sum(v for k, v in terms.items() if k not in _CLEAN)
            lines += ["**Termination reasons:**", ""]
            lines += ["| Reason | Count |", "|--------|-------|"]
            for reason, count in sorted(clean.items(), key=lambda x: -x[1]):
                lines.append(f"| {reason} | {count} |")
            if other:
                lines.append(f"| other | {other} |")
            lines += [""]

    if agents:
        dp = agents.get("debate_participation") or {}
        lines += ["### Debate Participation", ""]
        lines += ["| Agent | Debated | Revised | Won | Correct side | Rev. rate |",
                  "|-------|---------|---------|-----|--------------|-----------|"]
        for a in AGENT_NAMES:
            d = dp.get(a) or {}
            lines.append(
                f"| {a} | {d.get('n_debated', 0)} | {d.get('n_revised', 0)} | "
                f"{d.get('n_won', 0)} | {d.get('n_correct_side', 0)} | "
                f"{_pct(d.get('revision_rate'))} |"
            )
        lines += [""]
        if (plots_dir / "agent_debate_stats.png").exists():
            lines += ["![Per-agent debate participation](plots/agent_debate_stats.png)",
                      "_Per-agent debate participation_", ""]

    # Debate dynamics vs. ground truth (from dynamics_metrics.json)
    if dynamics and not dynamics.get("_error"):
        conv = dynamics.get("convergence") or {}
        r1m = dynamics.get("round1_majority_matches_final") or {}
        gtp = dynamics.get("gt_pairing_classification") or {}

        lines += ["### Convergence & Ground-Truth Effect", ""]
        if conv.get("converged_rate") is not None:
            lines += [
                f"- Debating agents converged to one position: "
                f"**{conv.get('converged', 0)}/{dynamics.get('n_images_with_debate', 0)}** "
                f"({_pct(conv.get('converged_rate'))})",
                f"- Still disagreed after debate: **{conv.get('still_disagreed', 0)}**",
            ]
        if r1m.get("rate") is not None:
            lines.append(
                f"- Final result matches Round-1 majority: "
                f"**{r1m.get('n_match', 0)}/{r1m.get('n_total', 0)}** ({_pct(r1m.get('rate'))})"
            )
        lines += [""]

        counts = gtp.get("counts") or {}
        if counts:
            n_p = gtp.get("n_pairings", 0)
            label = {
                "CONSTRUCTIVE": "Constructive (truth-bearer convinced the wrong agent)",
                "DESTRUCTIVE": "Destructive (truth-bearer abandoned the truth)",
                "STAND_CORRECT": "Stand-correct (truth-bearer held the line)",
                "BOTH_WRONG_NEUTRAL": "Both wrong (no truth-bearer in pairing)",
                "BOTH_CORRECT": "Both correct",
            }
            lines += [
                f"**Ground-truth pairing classification** (n = {n_p} debated pairings):",
                "",
                "| Category | Count | Share |",
                "|----------|-------|-------|",
            ]
            for key in ["CONSTRUCTIVE", "DESTRUCTIVE", "STAND_CORRECT", "BOTH_WRONG_NEUTRAL", "BOTH_CORRECT"]:
                if key in counts:
                    c = counts[key]
                    share = _pct(c / n_p) if n_p else "n/a"
                    lines.append(f"| {label[key]} | {c} | {share} |")
            lines += [""]

    # ══ 3. LLM-AS-JUDGE VERDICTS ═════════════════════════════════════════════
    if judge and not judge.get("_error"):
        lines += ["## 3. LLM-as-Judge Verdicts", ""]
        lines += [
            f"Verdicts: {judge.get('n_with_verdict', 0)}/{judge.get('n_total_judge_files', 0)}",
            "",
        ]
        lines += ["### Per-agent Scores", ""]
        lines += [
            "| Agent | Role adh. | Hall. ↓ | Arg. qual. ↑ | Rev. just. ↑ | Debate contrib. ↑ |",
            "|-------|-----------|---------|-------------|-------------|-------------------|",
        ]
        for a in AGENT_NAMES:
            pa = (judge.get("per_agent") or {}).get(a) or {}
            lines.append(
                f"| {a} | {_pct(pa.get('role_adherence_rate'))} | "
                f"{_score((pa.get('hallucination_score') or {}).get('mean'))} | "
                f"{_score((pa.get('argument_quality_score') or {}).get('mean'))} | "
                f"{_score((pa.get('revision_justification_score') or {}).get('mean'))} | "
                f"{_score((pa.get('debate_contribution_score') or {}).get('mean'))} |"
            )
        lines += [""]

        mod = judge.get("moderator_pairing_quality_score") or {}
        syn = judge.get("judge_synthesis_quality") or {}
        lines += ["### System-level Scores", ""]
        lines += ["| Metric | Mean | Median | n |", "|--------|------|--------|---|"]
        lines += [
            f"| Moderator pairing quality | {_score(mod.get('mean'))} | "
            f"{_score(mod.get('median'))} | {mod.get('n', 0)} |",
            f"| Judge synthesis quality | {_score(syn.get('mean'))} | "
            f"{_score(syn.get('median'))} | {syn.get('n', 0)} |",
            "",
        ]

        for plot_file, caption in [
            ("judge_role_adherence.png", "Role adherence per agent"),
            ("judge_argument_quality.png", "Argument quality per agent"),
            ("judge_revision_justification.png", "Revision justification per agent"),
            ("judge_debate_contribution.png", "Debate contribution per agent"),
            ("judge_synthesis_by_debate.png", "Judge synthesis quality: debate vs. no-debate"),
        ]:
            if (plots_dir / plot_file).exists():
                lines += [f"![{caption}](plots/{plot_file})", f"_{caption}_", ""]

        # Hallucination examples
        lines += ["### Hallucination Examples", ""]
        for a in AGENT_NAMES:
            pa = (judge.get("per_agent") or {}).get(a) or {}
            examples = pa.get("hallucination_examples") or []
            if examples:
                lines += [f"**{a}:**", ""]
                lines += ["| Image | Score | Claim |", "|-------|-------|-------|"]
                for ex in examples[:5]:
                    lines.append(
                        f"| {ex.get('image_id', '')} | "
                        f"{_score(ex.get('score'))} | {ex.get('example', '')} |"
                    )
                lines += [""]

    out_file = out_dir / "report.md"
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    print(f"[report] wrote {out_file}")

    from eval_debate.render_html import render as render_html
    render_html(out_dir)
