"""Compose a single markdown + HTML report from all eval outputs.

Reads (when present):
  <out>/geo_metrics.json
  <out>/agent_metrics.json
  <out>/funnel_metrics.json
  <out>/judge_summary.json
  <out>/heatmap_metrics.json        (NEW v2)
  <out>/calibration_metrics.json    (NEW v2)

Plus the static PNGs under <out>/plots/.

Writes <out>/report.md and (if Jinja2 is available) <out>/report.html.

Section order is intentional: the funnel + diagnostic sections appear right
after the headline because they are the primary tool for identifying which
pipeline stage to invest engineering effort in. Geo bias, heatmap, calibration,
per-agent breakdowns and judge verdicts come below as supporting detail.
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

from eval_tourn.loader import AGENT_NAMES


# Figure numbering, incremented across the document
class _FigCounter:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


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


def _fmt_pct_ci(block: dict | None) -> str:
    if not block or block.get("n", 0) == 0:
        return "n/a"
    return (
        f"{block['rate']:.1%} "
        f"[{block['ci_low']:.1%}, {block['ci_high']:.1%}]"
    )


def _fmt_km(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:,.0f} km"


def _fmt_num(x: float | None, decimals: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{decimals}f}"


def _fig(fc: _FigCounter, caption: str) -> str:
    return f"_Figure {fc.next()}: {caption}_"


def _section_headline(geo: dict | None, agents: dict | None, judge: dict | None) -> list[str]:
    out = ["# VLM Council Tournament Only Evaluation Report", ""]
    if geo:
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
    if judge and (judge.get("overall_quality_score") or {}).get("n"):
        oq = judge["overall_quality_score"]
        out.append(
            f"- **Judge synthesis quality:** {oq['mean']:.3f} mean "
            f"({oq['median']:.3f} median, n={oq['n']})"
        )
    out.append("")
    return out


def _section_funnel(fn: dict | None, fc: _FigCounter) -> list[str]:
    if not fn:
        return []
    funnel = fn.get("funnel") or {}
    stages = funnel.get("stages") or []
    if not stages:
        return []

    out = ["## Pipeline funnel, where does the truth get lost?", ""]
    out.append(
        "Each stage is a binary check on whether the **ground-truth country** "
        "was still reachable after that stage of the pipeline. Cumulative "
        "survival = truth survived all stages from S0 up to and including this "
        "one. Conditional rate = survivors / survivors of the previous stage "
        "(i.e. attrition at *this* step). Wilson 95% confidence intervals."
    )
    out.append("")

    bn = funnel.get("bottleneck")
    if bn:
        out.append(
            f"> **Bottleneck:** stage {bn['stage_code']}, _{bn['description']}_. "
            f"Conditional survival here is **{bn['conditional_rate']:.1%}** "
            f"[{bn['ci_low']:.1%}, {bn['ci_high']:.1%}]. "
            f"This is the single largest attrition step in the pipeline."
        )
        out.append("")

    if not funnel.get("monotone", True):
        out.append(
            "> :warning: Funnel is not monotone, a downstream stage shows more "
            "survivors than an upstream one. Check stage definitions / "
            "country-name normalization."
        )
        out.append("")

    out.append("| Stage | Description | Cumulative survival | Conditional on prev |")
    out.append("|---|---|---|---|")
    for s in stages:
        out.append(
            f"| **{s['code']}** | {s['description']} | "
            f"{_fmt_pct_ci(s['cumulative_survival'])} | "
            f"{_fmt_pct_ci(s['conditional_on_prev'])} |"
        )
    out.append("")
    out.append("![Pipeline funnel](plots/funnel.png)")
    out.append("")
    out.append(_fig(fc, "Pipeline funnel survival per stage with Wilson 95% CI."))
    out.append("")
    return out


def _section_oracle(fn: dict | None, fc: _FigCounter) -> list[str]:
    if not fn:
        return []
    c = fn.get("oracle_ceilings") or {}
    if not c.get("n_total"):
        return []
    out = ["## Oracle ceilings, counterfactual accuracy if a stage were perfect", ""]
    out.append(
        "Each oracle row shows what the end-to-end accuracy *would* be if a "
        "particular pipeline stage made no mistakes. The gap between **Actual** "
        "and an oracle is the upper bound of accuracy gain achievable by "
        "fixing that stage."
    )
    out.append("")

    out.append("| Scenario | Accuracy (95% CI) | Δ vs. actual |")
    out.append("|---|---|---|")
    actual_rate = c.get("actual", {}).get("rate", 0.0)
    for key, label in [
        ("actual", "Actual pipeline"),
        ("majority_vote_baseline", "Baseline: majority-vote of 5 agents"),
        ("oracle_region", "Oracle region (region always correct)"),
        ("oracle_pool", "Oracle pool (truth always in pool)"),
        ("oracle_tournament", "Oracle tournament (truth wins if in pool)"),
    ]:
        block = c.get(key) or {}
        delta = block.get("rate", 0.0) - actual_rate
        delta_str = f"{delta:+.1%}" if key != "actual" else ", "
        out.append(f"| {label} | {_fmt_pct_ci(block)} | {delta_str} |")
    out.append("")
    out.append("![Oracle ceilings](plots/oracle_ceilings.png)")
    out.append("")
    out.append(_fig(fc, "Counterfactual accuracy under perfect-stage oracles."))
    out.append("")
    return out


def _section_tournament_diag(fn: dict | None, fc: _FigCounter) -> list[str]:
    if not fn:
        return []
    diag = fn.get("tournament_diagnostics") or {}
    if not diag.get("n_matches"):
        return []
    out = ["## Tournament diagnostics", ""]
    out.append(
        "The tournament judge runs every match twice (forward + reverse "
        "positions) in parallel and only commits to a winner if both runs "
        "agree. On disagreement we fall back to pool-rank (higher seed wins). "
        "`country_a` / `country_b` are bracket slots, not prompt positions, so "
        "the relevant bias signal is the **agreement distribution** plus a "
        "**pool-seed-bias** check, not slot-win-rate."
    )
    out.append("")

    ag = diag.get("agreement_distribution") or {}
    out.append(f"**Agreement distribution** (n={diag['n_matches']} matches):")
    for key, label in [
        ("agree", "both runs agree"),
        ("forward_only", "only forward run produced a winner"),
        ("reverse_only", "only reverse run produced a winner"),
        ("disagree", "runs picked different winners → pool-rank tiebreak"),
        ("both_empty", "both runs failed to parse → pool-rank tiebreak"),
    ]:
        block = ag.get(key, {})
        if block.get("n", 0) > 0:
            out.append(f"- {label}: {_fmt_pct_ci(block)}")
    out.append("")

    tb = diag.get("tiebreak", {})
    if tb.get("n_total" if False else "n", 0) is not None:
        out.append(
            f"**Pool-rank tiebreak rate**: {_fmt_pct_ci(tb)} "
            f"of all matches resolved without unanimous symmetric agreement."
        )
        out.append("")

    seed = diag.get("pool_seed_bias", {})
    if seed.get("n", 0) > 0:
        seed_n = seed["n"]
        seed_lo, seed_hi = seed.get("ci_low", 0.0), seed.get("ci_high", 0.0)
        out.append(
            f"**Pool-seed bias** (does the higher-seeded country win? "
            f"n={seed_n} matches with distinct seeds): {_fmt_pct_ci(seed)}"
        )
        if seed_lo > 0.55:
            out.append(
                "- :warning: Higher-seeded country wins significantly more "
                "than 55%. The pool-rank tiebreak and/or the judge are leaning "
                "on seed information."
            )
        elif seed_hi < 0.45:
            out.append(
                "- :warning: Lower-seeded country wins significantly more "
                "than expected, judge may be over-correcting against seed."
            )
        else:
            out.append("- No significant pool-seed bias (50% is inside the CI).")
        out.append("")

    truth_block = diag.get("truth_match_win_rate", {})
    out.append(
        f"**Truth match-win rate** (matches where truth was one of the two "
        f"contenders, n={diag.get('n_truth_matches', 0)}): "
        f"{_fmt_pct_ci(truth_block)}"
    )
    out.append("")

    by_ag = diag.get("truth_outcomes_by_agreement") or {}
    if by_ag:
        out.append("**Truth outcomes split by agreement type:**")
        out.append("")
        out.append("| Agreement | Truth-in-match | Truth won | Win rate |")
        out.append("|---|---|---|---|")
        for key in ("agree", "forward_only", "reverse_only", "disagree", "both_empty"):
            blk = by_ag.get(key)
            if not blk or not blk.get("n"):
                continue
            wr = blk.get("win_rate")
            wr_str = f"{wr:.1%}" if isinstance(wr, (int, float)) else "n/a"
            out.append(f"| {key} | {blk['n']} | {blk['won']} | {wr_str} |")
        out.append("")

    pairs = diag.get("specific_pair_failures") or []
    if pairs:
        out.append("**Specific pair failures** (truth lost to the same opponent ≥ 2 times):")
        out.append("")
        out.append("| Truth | Lost to | Count |")
        out.append("|---|---|---|")
        for p in pairs:
            out.append(f"| {p['truth']} | {p['lost_to']} | {p['count']} |")
        out.append("")

    out.append("![Tournament symmetry](plots/tournament_symmetry.png)")
    out.append("")
    out.append(_fig(fc, "Symmetric judge agreement distribution and pool-seed-bias check."))
    out.append("")
    return out


def _section_agreement(fn: dict | None, fc: _FigCounter) -> list[str]:
    if not fn:
        return []
    agree = fn.get("agreement_curve") or {}
    levels = agree.get("levels") or []
    if not levels:
        return []
    out = ["## Agreement vs. accuracy", ""]
    out.append(
        "Bucketed by how many of the 5 agents picked the same top-1 country "
        "in the initial round. A steep gradient indicates that initial-round "
        "agent agreement is a strong predictor of correctness."
    )
    out.append("")
    out.append("| Agents agreeing on top-1 | n | Accuracy (95% CI) |")
    out.append("|---|---|---|")
    for lv in levels:
        out.append(f"| {lv['agreement']} | {lv['n']} | {_fmt_pct_ci(lv)} |")
    out.append("")
    out.append("![Agreement curve](plots/agreement_curve.png)")
    out.append("")
    out.append(_fig(fc, "Initial-round agent agreement vs. final accuracy."))
    out.append("")
    return out


def _section_path(fn: dict | None) -> list[str]:
    if not fn:
        return []
    paths = fn.get("path_comparison") or {}
    a = paths.get("path_a") or {}
    b = paths.get("path_b") or {}
    if not (a.get("n") or b.get("n")):
        return []
    out = ["## Path A vs. Path B", ""]
    out.append(
        "Path A = region consensus reached after the initial round. "
        "Path B = no consensus, agents do a second region-constrained "
        "assessment before the tournament."
    )
    out.append("")
    out.append("| Metric | Path A | Path B |")
    out.append("|---|---|---|")
    out.append(f"| n | {a.get('n', 0)} | {b.get('n', 0)} |")
    out.append(f"| Country accuracy | {_fmt_pct_ci(a)} | {_fmt_pct_ci(b)} |")
    out.append(
        f"| Median haversine | {_fmt_km(a.get('median_haversine_km', 0))} | "
        f"{_fmt_km(b.get('median_haversine_km', 0))} |"
    )
    out.append(
        f"| Truth in pool | {_fmt_pct(a.get('truth_in_pool_rate', 0))} "
        f"({a.get('truth_in_pool_n', 0)}) | "
        f"{_fmt_pct(b.get('truth_in_pool_rate', 0))} "
        f"({b.get('truth_in_pool_n', 0)}) |"
    )
    out.append(
        f"| Tournament survival when in pool | "
        f"{_fmt_pct(a.get('tournament_survival_when_in_pool', 0))} | "
        f"{_fmt_pct(b.get('tournament_survival_when_in_pool', 0))} |"
    )
    out.append(
        f"| Mean total seconds | {a.get('mean_total_seconds', 0):.1f} s | "
        f"{b.get('mean_total_seconds', 0):.1f} s |"
    )
    out.append("")
    return out


def _section_severity(fn: dict | None) -> list[str]:
    if not fn:
        return []
    sev = fn.get("severity") or {}
    if not sev.get("n_total"):
        return []
    out = ["## Severity of errors", ""]
    out.append(
        f"Of **{sev.get('n_wrong', 0)}** wrong predictions "
        f"(out of {sev.get('n_total', 0)} total):"
    )
    out.append(
        f"- **Near-miss** (haversine < 500 km): {sev.get('near_miss_count', 0)}"
    )
    out.append(
        f"- **Same-region wrong**: {sev.get('same_region_wrong_count', 0)}"
    )
    out.append(
        f"- **Wrong region**: {sev.get('wrong_region_count', 0)}"
    )
    out.append("")

    nm = sev.get("near_miss_examples") or []
    if nm:
        out.append("### Near-miss examples (first 15)")
        out.append("")
        out.append("| Image | Truth | Pred | Haversine |")
        out.append("|---|---|---|---|")
        for e in nm:
            out.append(
                f"| {e['image_id']} | {e['truth']} | {e['pred']} | "
                f"{_fmt_km(e['haversine_km']) if e.get('haversine_km') is not None else 'n/a'} |"
            )
        out.append("")

    sr = sev.get("same_region_wrong_examples") or []
    if sr:
        out.append("### Same-region but wrong country (first 15)")
        out.append("")
        out.append("| Image | Truth | Pred |")
        out.append("|---|---|---|")
        for e in sr:
            out.append(f"| {e['image_id']} | {e['truth']} | {e['pred']} |")
        out.append("")

    return out


def _section_anchoring(anchoring: dict | None, fc: _FigCounter) -> list[str]:
    """How often does the tournament merely confirm the pre-tournament top-seed?"""
    if not anchoring or not anchoring.get("n_with_pool"):
        return []
    n_pool = anchoring["n_with_pool"]
    out = ["## Tournament anchoring, does the bracket actually re-rank?", ""]
    out.append(
        f"Of {n_pool} images with a non-trivial pool (≥ 2 candidates), "
        f"the tournament winner equals the top-seeded candidate "
        f"**{_fmt_pct(anchoring.get('anchoring_rate'))}** of the time "
        f"(n={anchoring.get('n_anchored', 0)}). When anchored, the seed was "
        f"correct **{_fmt_pct(anchoring.get('anchoring_correctness'))}** of "
        f"the time."
    )
    out.append("")
    out.append(
        f"Of the {anchoring.get('n_re_ranked', 0)} re-ranked images "
        f"(winner ≠ top seed): "
        f"**{anchoring.get('n_re_rank_helped', 0)}** flipped *toward* truth, "
        f"**{anchoring.get('n_re_rank_hurt', 0)}** flipped *away* from truth, "
        f"**{anchoring.get('n_re_rank_neutral', 0)}** were neutral (truth missed "
        f"either way). Net help: **{anchoring.get('re_rank_net_help', 0):+d}** "
        f"images."
    )
    out.append("")
    out.append("![Anchoring outcomes](plots/anchoring.png)")
    out.append("")
    out.append(_fig(
        fc,
        "Anchoring vs. re-ranking outcomes. Left: stacked outcomes split "
        "by whether the tournament confirmed the top seed; right: net "
        "effect of re-ranking on truth recovery."
    ))
    out.append("")

    helped = anchoring.get("examples_helped") or []
    if helped:
        out.append("### Re-ranking saved truth (examples)")
        out.append("")
        out.append("| Image | Top seed | Tournament winner | Truth |")
        out.append("|---|---|---|---|")
        for ex in helped[:8]:
            out.append(
                f"| {ex.get('image_id', '?')} | {ex.get('seed_top1', '?')} | "
                f"**{ex.get('tournament_winner', '?')}** | {ex.get('truth', '?')} |"
            )
        out.append("")

    hurt = anchoring.get("examples_hurt") or []
    if hurt:
        out.append("### Re-ranking flipped away from truth (examples)")
        out.append("")
        out.append("| Image | Top seed (= truth) | Tournament winner | Truth |")
        out.append("|---|---|---|---|")
        for ex in hurt[:8]:
            out.append(
                f"| {ex.get('image_id', '?')} | {ex.get('seed_top1', '?')} | "
                f"{ex.get('tournament_winner', '?')} | **{ex.get('truth', '?')}** |"
            )
        out.append("")

    return out


def _section_rag_utility(rag: dict | None, fc: _FigCounter) -> list[str]:
    """How often does the RAG path actually swing a tournament match?"""
    if not rag or not rag.get("n_with_pool"):
        return []
    n_pool = rag["n_with_pool"]
    n_matches = rag.get("n_matches_total", 0)
    n_asym = rag.get("n_matches_asymmetric", 0)
    out = ["## RAG utility, does retrieval actually move the needle?", ""]
    out.append(
        f"Of {n_pool} images with a non-trivial pool, "
        f"**{_fmt_pct(rag.get('ref_coverage_rate'))}** had at least one "
        f"verified RAG reference surface "
        f"(n={rag.get('n_with_any_refs', 0)}; "
        f"mean refs/image = {rag.get('mean_refs_per_image', 0):.1f}, "
        f"mean countries with refs/image = "
        f"{rag.get('mean_countries_with_refs_per_image', 0):.1f})."
    )
    out.append("")

    if n_matches:
        out.append(
            f"Per-match ref availability (across {n_matches} tournament matches): "
            f"both sides have refs in {rag.get('n_matches_both_have_refs', 0)}, "
            f"asymmetric in **{n_asym}** "
            f"({_fmt_pct(rag.get('asymmetric_match_share'))}), "
            f"neither side has refs in "
            f"{rag.get('n_matches_neither_has_refs', 0)}."
        )
        out.append("")

    if n_asym:
        out.append(
            f"In the {n_asym} asymmetric matches, the ref-side wins "
            f"**{_fmt_pct(rag.get('asym_ref_side_win_rate'))}** of the time "
            f"(n={rag.get('n_asym_ref_side_won', 0)}). When the ref-side "
            f"wins, it equals the truth "
            f"**{_fmt_pct(rag.get('asym_ref_side_won_correctly_rate'))}** "
            f"of the time, i.e. ref-asymmetry steers toward truth in "
            f"{rag.get('n_asym_ref_side_was_truth_and_won', 0)} of "
            f"{rag.get('n_asym_ref_side_won', 0)} ref-side wins."
        )
        out.append("")

    out.append("![RAG utility](plots/rag_utility.png)")
    out.append("")
    out.append(_fig(
        fc,
        "RAG utility: coverage (left), per-match ref availability "
        "(middle), and outcome of asymmetric matches (right). "
        "'ref-side won (= truth)' is the only column that demonstrates "
        "RAG steering the bracket toward the correct answer."
    ))
    out.append("")

    examples_won_truth = rag.get("examples_ref_side_won_truth") or []
    if examples_won_truth:
        out.append("### RAG-asymmetric matches won by the ref-side (= truth)")
        out.append("")
        out.append("| Image | Round | Winner | Truth |")
        out.append("|---|---|---|---|")
        for ex in examples_won_truth[:8]:
            out.append(
                f"| {ex.get('image_id', '?')} | {ex.get('match', '?')} | "
                f"{ex.get('winner', '?')} | {ex.get('truth', '?')} |"
            )
        out.append("")

    examples_won_wrong = rag.get("examples_ref_side_won_wrong") or []
    if examples_won_wrong:
        out.append("### RAG-asymmetric matches where ref-side won but truth was the other side")
        out.append("")
        out.append("| Image | Round | Ref-side winner | Truth |")
        out.append("|---|---|---|---|")
        for ex in examples_won_wrong[:8]:
            out.append(
                f"| {ex.get('image_id', '?')} | {ex.get('match', '?')} | "
                f"{ex.get('ref_side_winner', '?')} | "
                f"**{ex.get('truth_was', '?')}** |"
            )
        out.append("")

    return out


def _section_heatmap(heatmap: dict | None, fc: _FigCounter) -> list[str]:
    if not heatmap:
        return []
    out = ["### Geographic heatmap", ""]
    out.append(
        f"Per-country true-positive rate (TPR = correct ÷ truth = country) and "
        f"false-positive counts. Macro-averaged TPR across "
        f"{heatmap.get('n_countries_with_truth', 0)} countries with truth: "
        f"**{_fmt_pct(heatmap.get('macro_avg_tpr', 0))}**."
    )
    out.append("")
    out.append("![World heatmap](plots/world_map_accuracy.png)")
    out.append("")
    out.append(_fig(fc, "Per-country TPR (green) with FP outlines (red)."))
    out.append("")

    out.append("![F1 map](plots/world_map_f1.png)")
    out.append("")
    out.append(_fig(
        fc,
        "Per-country F1 = 2·P·R / (P+R), divergent around the run's "
        "macro-F1. Green = above-average, red = below-average. Alpha "
        "scales with √(TP+FP+FN) so low-evidence countries fade out. "
        "Used for an imbalanced dataset because raw TPR/recall ignores "
        "false positives."
    ))
    out.append("")

    out.append("![Error bias map](plots/world_map_error_bias.png)")
    out.append("")
    out.append(_fig(
        fc,
        "Per-country error bias (FP − FN)/(FP + FN), TP ignored. "
        "Red = the country is over-predicted (false positives only); "
        "blue = the country is missed (false negatives only); pastel "
        "tones = mixed FP/FN. Only countries with ≥ 1 error are drawn. "
        "Outline thickness ∝ error volume."
    ))
    out.append("")

    per_country = heatmap.get("per_country") or {}
    # Show worst 10 (lowest TPR) and best 10 (highest TPR), with at least 2 truth samples
    rows = [
        (name, blk) for name, blk in per_country.items()
        if (blk.get("n_truth") or 0) >= 2 and blk.get("tpr") is not None
    ]
    if rows:
        rows_sorted = sorted(rows, key=lambda x: x[1]["tpr"])
        out.append("#### Worst-performing countries (lowest TPR, at least 2 truth samples)")
        out.append("")
        out.append("| Country | n_truth | n_correct | n_pred | n_fp | TPR |")
        out.append("|---|---|---|---|---|---|")
        for name, blk in rows_sorted[:10]:
            out.append(
                f"| {name.title()} | {blk['n_truth']} | {blk['n_correct']} | "
                f"{blk['n_predicted']} | {blk['n_false_positive']} | "
                f"{_fmt_pct(blk['tpr'])} |"
            )
        out.append("")

    # Most-frequently-falsely-predicted
    fp_rows = [
        (name, blk) for name, blk in per_country.items()
        if (blk.get("n_false_positive") or 0) > 0
    ]
    if fp_rows:
        fp_sorted = sorted(fp_rows, key=lambda x: -x[1]["n_false_positive"])
        out.append("#### Top false-positive predictions")
        out.append("")
        out.append("| Country | n_predicted | n_false_positive | PPV |")
        out.append("|---|---|---|---|")
        for name, blk in fp_sorted[:10]:
            out.append(
                f"| {name.title()} | {blk['n_predicted']} | "
                f"{blk['n_false_positive']} | "
                f"{_fmt_pct(blk['ppv']) if blk['ppv'] is not None else 'n/a'} |"
            )
        out.append("")
    return out


def _section_calibration(calibration: dict | None, fc: _FigCounter) -> list[str]:
    if not calibration:
        return []
    labels = calibration.get("labels") or ["high", "medium", "low", "speculative"]
    out = ["### Confidence calibration", ""]
    out.append(
        "Each specialist annotates **every candidate** in its list with a "
        "confidence label (`high` / `medium` / `low` / `speculative`). These "
        "labels are then handed to the tournament judge as evidence, so the "
        "relevant question is: when an agent assigns label X to a country, "
        "in what fraction of those (image, country) pairs was that country "
        "the ground truth?"
    )
    out.append("")
    out.append("#### Per-label hit-rate, P(truth | label)")
    out.append("")
    header = "| Agent | " + " | ".join(f"{lab} (n)" for lab in labels) + " |"
    sep = "|---|" + "|".join("---" for _ in labels) + "|"
    out.append(header)
    out.append(sep)
    per_agent = calibration.get("per_agent") or {}
    for agent in AGENT_NAMES:
        m = per_agent.get(agent, {})
        lh = m.get("label_hit_rate") or {}
        cells = []
        for lab in labels:
            blk = lh.get(lab) or {}
            n = blk.get("n", 0)
            rate = blk.get("rate")
            cells.append(f"{rate:.0%} ({n})" if (rate is not None and n) else f", ({n})")
        out.append(f"| {agent} | " + " | ".join(cells) + " |")
    avg = calibration.get("average") or {}
    avg_lh = avg.get("label_hit_rate") or {}
    cells = []
    for lab in labels:
        blk = avg_lh.get(lab) or {}
        n = blk.get("n", 0)
        rate = blk.get("rate")
        cells.append(f"**{rate:.0%}** ({n})" if (rate is not None and n) else f", ({n})")
    out.append("| **average** | " + " | ".join(cells) + " |")
    out.append("")
    out.append("![Label hit-rate](plots/calibration_label_hit_rate.png)")
    out.append("")
    out.append(_fig(
        fc,
        "Per-label hit-rate: how often a labeled candidate equals the ground truth. "
        "A well-calibrated agent shows monotonically falling bars (high > medium > low > speculative)."
    ))
    out.append("")
    out.append("#### Top-1 Brier / ECE")
    out.append("")
    out.append(
        "Single-number summary of how well the **top-1 pick's** confidence "
        "label correlates with being correct. Brier = mean squared error of "
        f"`p` (mapped via `{calibration.get('confidence_to_p', {})}`) against "
        "the 0/1 outcome; ECE = expected calibration error across confidence "
        "bins. Lower is better for both."
    )
    out.append("")
    out.append("| Agent | n | Brier | ECE |")
    out.append("|---|---|---|---|")
    for agent in AGENT_NAMES:
        m = per_agent.get(agent, {})
        out.append(
            f"| {agent} | {m.get('n_top1', 0)} | "
            f"{m.get('brier', 0):.3f} | {m.get('ece', 0):.3f} |"
        )
    out.append(
        f"| **average** | **{avg.get('n_top1', 0)}** | "
        f"**{avg.get('brier', 0):.3f}** | **{avg.get('ece', 0):.3f}** |"
    )
    out.append("")
    return out


def _section_geo(geo: dict | None, fc: _FigCounter) -> list[str]:
    if not geo:
        return []
    out = ["### Geo-spatial bias", ""]
    nb = geo.get("north_bias_test", {}) or {}
    eb = geo.get("east_bias_test", {}) or {}
    out.append(f"- North/south bias: {nb.get('interpretation', 'n/a')}")
    out.append(f"- East/west bias: {eb.get('interpretation', 'n/a')}")
    quads = geo.get("quadrants", {}) or {}
    if quads:
        ordered = ", ".join(f"{k}={v}" for k, v in sorted(quads.items()))
        out.append(f"- Error quadrants: {ordered}")
    abs_lat = geo.get("abs_lat_error_deg", {}) or {}
    abs_lng = geo.get("abs_lng_error_deg", {}) or {}
    if abs_lat.get("n") and abs_lng.get("n"):
        out.append(
            f"- Mean |lat error|: {abs_lat.get('mean', 0):.2f}°, "
            f"mean |lng error|: {abs_lng.get('mean', 0):.2f}°"
        )
    out.append("")
    out.append("![Error distribution](plots/error_distribution.png)")
    out.append("")
    out.append(_fig(fc, "Latitude/longitude error distribution."))
    out.append("")
    out.append("![Bearing rose](plots/bearing_rose.png)")
    out.append("")
    out.append(_fig(fc, "Bearing of prediction errors."))
    out.append("")

    confs = geo.get("top_confusions", []) or []
    if confs:
        out.append("#### Top confusion pairs")
        out.append("")
        out.append("| Truth | Predicted | Count |")
        out.append("|---|---|---|")
        for row in confs[:15]:
            out.append(f"| {row['truth']} | {row['predicted']} | {row['count']} |")
        out.append("")
        out.append("![Confusion matrix](plots/confusion_matrix.png)")
        out.append("")
        out.append(_fig(fc, "Confusion matrix of top-15 confused country pairs."))
        out.append("")

    asym = geo.get("asymmetric_confusion_pairs", []) or []
    if asym:
        out.append("#### Asymmetric confusions")
        out.append("")
        out.append("| Country A | Country B | A→B | B→A | Asymmetry |")
        out.append("|---|---|---|---|---|")
        for row in asym[:10]:
            out.append(
                f"| {row['country_a']} | {row['country_b']} | "
                f"{row['a_predicted_as_b']} | {row['b_predicted_as_a']} | "
                f"{row['asymmetry']:+d} |"
            )
        out.append("")
    return out


def _section_agents(agents: dict | None, fc: _FigCounter) -> list[str]:
    if not agents:
        return []
    out = ["## Per-agent metrics", ""]
    initial = agents.get("initial_round", {}) or {}
    country = agents.get("country_round_path_b", {}) or {}

    out.append("### Initial round")
    out.append("")
    out.append("| Agent | n | Top-1 | Top-3 | Coverage |")
    out.append("|---|---|---|---|---|")
    for a in AGENT_NAMES:
        m = initial.get(a, {})
        out.append(
            f"| {a} | {m.get('n', 0)} | "
            f"{_fmt_pct(m.get('top1_accuracy', 0))} | "
            f"{_fmt_pct(m.get('top3_hit_rate', 0))} | "
            f"{_fmt_pct(m.get('coverage', 0))} |"
        )
    out.append("")

    if country and any(country.get(a, {}).get("n") for a in AGENT_NAMES):
        out.append("### Country round (Path B)")
        out.append("")
        out.append("| Agent | n | Top-1 | Top-3 | Coverage |")
        out.append("|---|---|---|---|---|")
        for a in AGENT_NAMES:
            m = country.get(a, {})
            out.append(
                f"| {a} | {m.get('n', 0)} | "
                f"{_fmt_pct(m.get('top1_accuracy', 0))} | "
                f"{_fmt_pct(m.get('top3_hit_rate', 0))} | "
                f"{_fmt_pct(m.get('coverage', 0))} |"
            )
        out.append("")

    out.append("![Agent top-1](plots/agent_top1.png)")
    out.append("")
    out.append(_fig(fc, "Per-agent top-1 accuracy."))
    out.append("")
    out.append("![Agent calibration](plots/agent_calibration.png)")
    out.append("")
    out.append(_fig(fc, "Per-agent top-1 vs. confidence calibration."))
    out.append("")
    return out


def _section_judge(judge: dict | None, fc: _FigCounter) -> list[str]:
    if not judge:
        return []
    out = ["## 3. LLM-as-Judge Verdicts", ""]
    out.append(
        f"- Verdicts: {judge.get('n_with_verdict', 0)}/"
        f"{judge.get('n_total_judge_files', 0)}"
    )
    errs = judge.get("errors", {}) or {}
    if errs:
        line = ", ".join(f"{k}={v}" for k, v in sorted(errs.items()))
        out.append(f"- Errors: {line}")
    out.append(
        f"- Constructive synthesis rate: "
        f"{_fmt_pct(judge.get('constructive_synthesis_rate', 0))} "
        f"(n={judge.get('constructive_synthesis_n', 0)})"
    )
    oq = judge.get("overall_quality_score") or {}
    if oq.get("n"):
        out.append(
            f"- Overall quality score: mean **{oq['mean']:.3f}**, "
            f"median {oq['median']:.3f}, σ {oq['stdev']:.3f} (n={oq['n']})"
        )
    out.append("")

    per = judge.get("per_agent", {}) or {}
    if per:
        out.append("### Per-agent quantitative scores")
        out.append("")
        out.append(
            "| Agent | n | Role adher. | Hallucination ↓ | Visual cons. ↑ | Calibration ↑ |"
        )
        out.append("|---|---|---|---|---|---|")
        for a in AGENT_NAMES:
            m = per.get(a, {})
            hall = m.get("hallucination_score") or {}
            vis = m.get("visual_consistency_score") or {}
            cal = m.get("confidence_calibration_score") or {}
            def _mean(blk: dict) -> str:
                return f"{blk['mean']:.2f}" if blk.get("mean") is not None else ", "
            out.append(
                f"| {a} | {m.get('n', 0)} | "
                f"{_fmt_pct(m.get('role_adherence_rate', 0))} | "
                f"{_mean(hall)} | {_mean(vis)} | {_mean(cal)} |"
            )
        out.append("")
        out.append("![Role adherence](plots/judge_role_adherence.png)")
        out.append("")
        out.append(_fig(fc, "Per-agent role adherence rate (overall, when run correct, when run wrong)."))
        out.append("")
        out.append("![Hallucination](plots/judge_hallucination.png)")
        out.append("")
        out.append(_fig(fc, "Per-agent mean hallucination score (0 = clean, 1 = severe)."))
        out.append("")
        out.append("![Visual consistency](plots/judge_visual_consistency.png)")
        out.append("")
        out.append(_fig(fc, "Per-agent mean visual consistency."))
        out.append("")
        out.append("![Argumentative quality](plots/judge_quality.png)")
        out.append("")
        out.append(_fig(fc, "Argumentative quality histogram per agent."))
        out.append("")
    return out


def _section_judge_failure(judge: dict | None, fc: _FigCounter) -> list[str]:
    if not judge:
        return []
    tf = judge.get("tournament_failure") or {}
    by_reason = tf.get("by_reason") or {}
    if not by_reason:
        return []
    out = ["### Tournament failure attribution", ""]
    out.append(
        "For matches where the truth was in the candidate pool but did not win, "
        "the judge classifies the cause. Counterfactual winnable rate: "
        f"**{_fmt_pct(tf.get('counterfactual_winnable_rate', 0))}** "
        f"({tf.get('counterfactual_n', 0)} attributable losses)."
    )
    out.append("")
    out.append("| Failure reason | Count |")
    out.append("|---|---|")
    for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
        out.append(f"| `{reason}` | {count} |")
    out.append("")
    out.append("![Failure attribution](plots/judge_failure_reasons.png)")
    out.append("")
    out.append(_fig(fc, "Tournament failure attribution histogram."))
    out.append("")

    examples = tf.get("examples") or []
    if examples:
        out.append("#### Failure examples (first 10)")
        out.append("")
        out.append("| Image | Reason | Lost to | Round | Counterfactual? |")
        out.append("|---|---|---|---|---|")
        for e in examples[:10]:
            cf = "yes" if e.get("counterfactual_winnable") else "no"
            out.append(
                f"| {e.get('image_id', '?')} | `{e.get('failure_reason', '?')}` | "
                f"{e.get('truth_lost_to') or ', '} | "
                f"{e.get('failure_match_round') or ', '} | {cf} |"
            )
        out.append("")
    return out


def _section_judge_hallucination(judge: dict | None) -> list[str]:
    if not judge:
        return []
    per = judge.get("per_agent") or {}
    any_examples = any((per.get(a, {}).get("hallucination_examples")) for a in AGENT_NAMES)
    if not any_examples:
        return []
    out = ["### Hallucination examples", ""]
    out.append(
        "Concrete claims the judge flagged as not supported by the image. "
        "Up to 5 examples per agent."
    )
    out.append("")
    for agent in AGENT_NAMES:
        ex_list = (per.get(agent, {}).get("hallucination_examples")) or []
        if not ex_list:
            continue
        out.append(f"#### {agent}")
        out.append("")
        out.append("| Image | Score | Hallucinated claim |")
        out.append("|---|---|---|")
        for ex in ex_list:
            score = ex.get("score")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else ", "
            claim = (ex.get("example") or "").replace("|", r"\|")
            out.append(f"| {ex.get('image_id', '?')} | {score_str} | {claim} |")
        out.append("")
    return out


def _section_dynamics(dynamics: dict | None, agents: dict | None, fc: _FigCounter) -> list[str]:
    """Tournament dynamics: candidate pool to bracket seeding, matches and GT ladder.

    Region gating was removed from this approach, so there is deliberately no
    region / Path-A-B / funnel-S1 content: only pool, bracket, seed and match
    dynamics appear here.
    """
    if not dynamics or dynamics.get("_error"):
        return []
    out = ["## 2. Approach Dynamics", ""]
    out.append(
        "The tournament only pipeline runs five independent agent assessments "
        "into a candidate pool, seeds the top four of that pool into a bracket "
        "(seed 0 vs seed 3, seed 1 vs seed 2, then a final), and returns the "
        "champion as the answer. There is no region gate, so these dynamics "
        "cover the candidate pool, the pool to bracket seeding step, the "
        "bracket matches and the ground truth survival ladder."
    )
    out.append("")

    pb = dynamics.get("pool_bracket") or {}
    if pb:
        out.append("### Candidate pool to bracket seeding")
        out.append("")
        out.append("| Metric | Value |")
        out.append("|---|---|")
        out.append(f"| Mean pool size | {_fmt_num(pb.get('pool_size_mean'))} |")
        out.append(f"| Median pool size | {_fmt_num(pb.get('pool_size_median'), 0)} |")
        out.append(f"| Max pool size | {pb.get('pool_size_max', 'n/a')} |")
        out.append(f"| Mean bracket size | {_fmt_num(pb.get('bracket_size_mean'))} |")
        out.append(
            f"| Mean pool to bracket slack | {_fmt_num(pb.get('slack_mean'))} "
            f"(max {pb.get('slack_max', 'n/a')}) |"
        )
        out.append(
            f"| Bracket equals pool top 4 (as a set) | "
            f"{pb.get('n_bracket_equals_pool_top4', 0)} "
            f"({_fmt_pct(pb.get('bracket_equals_pool_top4_rate'))}) |"
        )
        out.append(
            f"| Pool larger than 4 (some pool entries dropped) | "
            f"{pb.get('n_pool_beyond_4', 0)} "
            f"({_fmt_pct(pb.get('pool_beyond_4_rate'))}) |"
        )
        out.append("")

    seed0 = dynamics.get("seed0_origin") or {}
    if seed0.get("n"):
        out.append("### Seed 0 origin (which signal produces the top seed)")
        out.append("")
        out.append(
            f"Across {seed0['n']} images with a bracket, the seed 0 country "
            f"matches the initial plurality (at least 2 of 5 agents) "
            f"**{_fmt_pct(seed0.get('matches_initial_plurality_rate'))}** of the time "
            f"(n={seed0.get('matches_initial_plurality', 0)})."
        )
        out.append("")
        out.append("| Agent | Seed 0 matches this agent top pick |")
        out.append("|---|---|")
        for a in AGENT_NAMES:
            blk = (seed0.get("matches_agent_top_pick") or {}).get(a) or {}
            out.append(
                f"| {a} | {blk.get('n', 0)} ({_fmt_pct(blk.get('rate'))}) |"
            )
        out.append("")

    tour = dynamics.get("tournament") or {}
    if tour.get("n_matches"):
        out.append("### Tournament matches")
        out.append("")
        out.append(
            f"Across {tour['n_matches']} bracket matches: judge agrees with the "
            f"specialists **{_fmt_pct(tour.get('agree_rate'))}** "
            f"(n={tour.get('agree', 0)}), disagrees "
            f"**{_fmt_pct(tour.get('disagree_rate'))}** "
            f"(n={tour.get('disagree', 0)}), and a lower seed wins (upset) "
            f"**{_fmt_pct(tour.get('upset_rate'))}** (n={tour.get('upsets', 0)})."
        )
        out.append("")
        cbs = tour.get("champion_by_seed") or {}
        if cbs:
            out.append(f"**Champion by original seed** ({tour.get('finals_played', 0)} finals):")
            out.append("")
            out.append("| Original seed | Champions | Share |")
            out.append("|---|---|---|")
            for k in sorted(cbs, key=lambda s: int(s)):
                blk = cbs[k] or {}
                out.append(f"| {k} | {blk.get('n', 0)} | {_fmt_pct(blk.get('rate'))} |")
            out.append("")

    ratify = dynamics.get("initial_plurality_vs_champion") or {}
    if ratify.get("n_with_plurality"):
        out.append("### Initial plurality vs champion (ratify or revise)")
        out.append("")
        out.append(
            f"Of {ratify['n_with_plurality']} images with an initial plurality "
            f"(at least 2 of 5 agents naming the same country), the tournament "
            f"champion equals that plurality "
            f"**{_fmt_pct(ratify.get('champion_equals_plurality_rate'))}** "
            f"(n={ratify.get('champion_equals_plurality', 0)}) and revises it "
            f"**{_fmt_pct(ratify.get('champion_differs_rate'))}** "
            f"(n={ratify.get('champion_differs', 0)}). "
            f"{ratify.get('n_no_plurality', 0)} images had no initial plurality."
        )
        out.append("")

    ladder = dynamics.get("gt_ladder") or {}
    if ladder.get("n_with_gt"):
        gt_n = ladder["n_with_gt"]
        out.append("### Ground truth survival ladder")
        out.append("")
        out.append(
            "Each gate can drop the ground truth country, and no downstream "
            f"gate can recover it. Rates are over {gt_n} images with ground truth."
        )
        out.append("")
        out.append("| Gate | Survived | Rate |")
        out.append("|---|---|---|")
        out.append(f"| Ground truth in candidate pool | {ladder.get('gt_in_pool', 0)} | {_fmt_pct(ladder.get('gt_in_pool_rate'))} |")
        out.append(f"| Ground truth in bracket (top 4 seeded) | {ladder.get('gt_in_bracket', 0)} | {_fmt_pct(ladder.get('gt_in_bracket_rate'))} |")
        out.append(f"| Ground truth at seed 0 | {ladder.get('gt_at_seed0', 0)} | {_fmt_pct(ladder.get('gt_at_seed0_rate'))} |")
        out.append(f"| Ground truth reached the final | {ladder.get('gt_reached_final', 0)} | {_fmt_pct(ladder.get('gt_reached_final_rate'))} |")
        out.append(f"| Ground truth won the tournament | {ladder.get('gt_won_final', 0)} | {_fmt_pct(ladder.get('gt_won_final_rate'))} |")
        out.append("")
        surv = ladder.get("survival") or {}
        out.append("**Gate to gate survival of ground truth:**")
        out.append("")
        out.append(f"- In pool to in bracket: {_fmt_pct(surv.get('pool_to_bracket'))}")
        out.append(f"- In bracket to reached final: {_fmt_pct(surv.get('bracket_to_final'))}")
        out.append(f"- Reached final to won tournament: {_fmt_pct(surv.get('final_to_champion'))}")
        out.append("")

    fid = dynamics.get("seed_fidelity") or {}
    if fid.get("n_gt_in_bracket"):
        out.append("### Seed fidelity (does the tournament correct mis seedings?)")
        out.append("")
        out.append(
            f"Over the {fid['n_gt_in_bracket']} images where the ground truth "
            f"reached the bracket, the tournament picks it "
            f"**{_fmt_pct(fid.get('won_when_in_bracket_rate'))}** of the time "
            f"(n={fid.get('won_when_in_bracket', 0)})."
        )
        out.append("")
        by_seed = fid.get("by_seed") or {}
        if by_seed:
            out.append("| Ground truth seed | n | Won | Win rate |")
            out.append("|---|---|---|---|")
            for k in sorted(by_seed, key=lambda s: int(s)):
                blk = by_seed[k] or {}
                out.append(
                    f"| {k} | {blk.get('n', 0)} | {blk.get('won', 0)} | "
                    f"{_fmt_pct(blk.get('win_rate'))} |"
                )
            out.append("")

    if agents:
        initial = agents.get("initial_round", {}) or {}
        out.append("### Per-agent initial round")
        out.append("")
        out.append(
            "In tournament only there is no per-agent country reassessment, so "
            "each agent contributes only its initial top pick."
        )
        out.append("")
        out.append("| Agent | n | Top-1 | Top-3 | Coverage |")
        out.append("|---|---|---|---|---|")
        for a in AGENT_NAMES:
            m = initial.get(a, {})
            out.append(
                f"| {a} | {m.get('n', 0)} | "
                f"{_fmt_pct(m.get('top1_accuracy', 0))} | "
                f"{_fmt_pct(m.get('top3_hit_rate', 0))} | "
                f"{_fmt_pct(m.get('coverage', 0))} |"
            )
        out.append("")
        out.append("![Agent top-1](plots/agent_top1.png)")
        out.append("")
        out.append(_fig(fc, "Per-agent top-1 accuracy."))
        out.append("")

    return out


def _section_gt_stats(geo: dict | None, heatmap: dict | None,
                      calibration: dict | None, fc: _FigCounter) -> list[str]:
    """## 1. Ground-Truth Statistics: headline metrics, geo bias, heatmap, calibration."""
    if not (geo or heatmap or calibration):
        return []
    out = ["## 1. Ground-Truth Statistics", ""]
    if geo:
        hav = geo.get("haversine_km", {}) or {}
        out.append("### Headline metrics")
        out.append("")
        out.append("| Metric | Value |")
        out.append("|---|---|")
        out.append(f"| Country accuracy | {_fmt_pct(geo.get('country_accuracy'))} |")
        out.append(f"| Median haversine | {_fmt_km(hav.get('median'))} |")
        out.append(f"| Mean haversine | {_fmt_km(hav.get('mean'))} |")
        out.append(f"| N images | {geo.get('n_total', 'n/a')} |")
        out.append("")
    out += _section_geo(geo, fc)
    out += _section_heatmap(heatmap, fc)
    out += _section_calibration(calibration, fc)
    return out


def run(out_dir: Path) -> Path:
    geo = _load_json(out_dir / "geo_metrics.json")
    agents = _load_json(out_dir / "agent_metrics.json")
    judge = _load_json(out_dir / "judge_summary.json")
    heatmap = _load_json(out_dir / "heatmap_metrics.json")
    calibration = _load_json(out_dir / "calibration_metrics.json")
    dynamics = _load_json(out_dir / "dynamics_metrics.json")

    fc = _FigCounter()
    lines: list[str] = []
    # TL;DR
    lines += _section_headline(geo, agents, judge)
    # 1. Ground-truth statistics (headline metrics, geo bias, heatmap/world maps,
    #    calibration). Region gating was removed, so no region/Path/funnel stages.
    lines += _section_gt_stats(geo, heatmap, calibration, fc)
    # 2. Approach dynamics (candidate pool to bracket seeding, tournament, GT ladder)
    lines += _section_dynamics(dynamics, agents, fc)
    # 3. LLM-as-judge verdicts (scores, failure attribution, hallucination examples)
    lines += _section_judge(judge, fc)
    lines += _section_judge_failure(judge, fc)
    lines += _section_judge_hallucination(judge)

    if not (geo or agents or judge or heatmap or calibration or dynamics):
        lines.append("_No eval outputs found in this directory._")

    out_file = out_dir / "report.md"
    out_file.write_text("\n".join(lines))
    print(f"[report] wrote {out_file}")

    # Also render HTML (best-effort; doesn't fail if Jinja2 missing)
    try:
        from eval_tourn.render_html import render as render_html
        render_html(out_dir)
    except Exception as e:
        print(f"[report] HTML rendering skipped: {e}")

    return out_file
