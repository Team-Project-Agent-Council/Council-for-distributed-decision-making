"""Per-agent and debate-dynamics metrics for the Debate approach."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_debate._style import STYLE, setup_plot_style
from eval_debate.loader import (
    AGENT_NAMES, RunRecord, load_run,
    top1_country, topk_countries, countries_match,
)


def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate_block(k: int, n: int) -> dict:
    if n == 0:
        return {"n": 0, "correct": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    lo, hi = _wilson_ci(k, n)
    return {"n": n, "correct": k, "rate": k / n, "ci_low": lo, "ci_high": hi}


def _agent_r1_metrics(records: list[RunRecord], agent: str) -> dict:
    n, top1, top3, has_cands = 0, 0, 0, 0
    for r in records:
        a = r.r1_assessments.get(agent, {})
        cands = a.get("candidates", [])
        n += 1
        if cands:
            has_cands += 1
            t1 = top1_country(a)
            if t1 and countries_match(t1, r.truth_country_code):
                top1 += 1
            topk = topk_countries(a, 3)
            if any(countries_match(c, r.truth_country_code) for c in topk):
                top3 += 1
    return {
        "n": n,
        "top1_accuracy": top1 / n if n else 0.0,
        "top3_hit_rate": top3 / n if n else 0.0,
        "coverage": has_cands / n if n else 0.0,
        **_rate_block(top1, n),
    }


def _agent_debate_metrics(records: list[RunRecord], agent: str) -> dict:
    n_debated = 0
    n_revised = 0
    n_won = 0
    n_correct_side = 0  # debated FOR the correct country
    n_wrong_side = 0    # debated AGAINST the correct country

    for r in records:
        pairings = r.debate.get("pairings", [])
        for p in pairings:
            if agent not in (p.get("agent_a"), p.get("agent_b")):
                continue
            exchanges = p.get("exchanges", [])
            agent_exchanges = [ex for ex in exchanges if ex.get("agent_name") == agent]
            if not agent_exchanges:
                continue
            n_debated += 1
            # Did this agent revise?
            if any(ex.get("revised") for ex in agent_exchanges):
                n_revised += 1
            # Did the opponent revise (this agent "won")?
            opp = p.get("agent_b") if p.get("agent_a") == agent else p.get("agent_a")
            opp_exchanges = [ex for ex in exchanges if ex.get("agent_name") == opp]
            if any(ex.get("revised") for ex in opp_exchanges):
                n_won += 1
            # Was this agent debating for the correct country?
            # Use initial position for this pairing
            init_pos = (
                p.get("agent_a_initial_position") if p.get("agent_a") == agent
                else p.get("agent_b_initial_position")
            )
            if init_pos:
                if countries_match(init_pos, r.truth_country_code):
                    n_correct_side += 1
                else:
                    n_wrong_side += 1

    return {
        "n_debated": n_debated,
        "n_revised": n_revised,
        "n_won": n_won,
        "n_correct_side": n_correct_side,
        "n_wrong_side": n_wrong_side,
        "revision_rate": n_revised / n_debated if n_debated else None,
        "win_rate": n_won / n_debated if n_debated else None,
    }


def _debate_overall_stats(records: list[RunRecord]) -> dict:
    n = len(records)
    n_no_debate = sum(1 for r in records if not r.debate_happened)
    n_debate = n - n_no_debate

    rounds_dist: Counter = Counter()
    term_reasons: Counter = Counter()
    exchanges_per_pairing: list[int] = []
    n_revised_pairings = 0
    n_total_pairings = 0

    for r in records:
        rounds_dist[r.total_debate_rounds] += 1
        if r.termination_reason:
            # Normalise long termination reason strings
            reason = r.termination_reason
            if "consensus" in reason.lower():
                reason = "consensus"
            elif "stalemate" in reason.lower():
                reason = "stalemate"
            elif "max_rounds" in reason.lower():
                reason = "max_rounds_reached"
            elif "weak" in reason.lower():
                reason = "weak_dissent"
            term_reasons[reason] += 1
        for p in r.debate.get("pairings", []):
            exch = p.get("exchanges", [])
            if exch:
                n_total_pairings += 1
                exchanges_per_pairing.append(len(exch))
                if any(ex.get("revised") for ex in exch):
                    n_revised_pairings += 1

    return {
        "n_total": n,
        "n_no_debate": n_no_debate,
        "n_debate_1plus": n_debate,
        "debate_rate": n_debate / n if n else 0.0,
        "rounds_distribution": {str(k): v for k, v in sorted(rounds_dist.items())},
        "termination_reasons": {k: v for k, v in term_reasons.most_common()},
        "n_total_pairings": n_total_pairings,
        "mean_exchanges_per_pairing": (
            sum(exchanges_per_pairing) / len(exchanges_per_pairing)
            if exchanges_per_pairing else None
        ),
        "revision_rate_pairings": (
            n_revised_pairings / n_total_pairings if n_total_pairings else None
        ),
    }


def _plot_agent_top1(initial_round: dict, out_path: Path) -> None:
    setup_plot_style()
    rates = [initial_round.get(a, {}).get("top1_accuracy", 0.0) for a in AGENT_NAMES]
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, rates, color=STYLE.primary, alpha=0.85)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.01,
                f"{rate:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-1 accuracy (Round 1)")
    ax.set_title("Per-agent top-1 accuracy, initial round")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_debate_stats(debate_metrics: dict, out_path: Path) -> None:
    setup_plot_style()
    n_deb = [debate_metrics.get(a, {}).get("n_debated", 0) for a in AGENT_NAMES]
    n_rev = [debate_metrics.get(a, {}).get("n_revised", 0) for a in AGENT_NAMES]
    n_won = [debate_metrics.get(a, {}).get("n_won", 0) for a in AGENT_NAMES]
    x = np.arange(len(AGENT_NAMES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, n_deb, width, label="Times debated", color=STYLE.primary, alpha=0.85)
    ax.bar(x, n_rev, width, label="Revised (conceded)", color=STYLE.error, alpha=0.85)
    ax.bar(x + width, n_won, width, label="Opponent revised (won)", color=STYLE.success, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("Count")
    ax.set_title("Per-agent debate participation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(results_dir: Path, gt_csv: Path, out_dir: Path) -> dict:
    records = load_run(results_dir, gt_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    initial_round = {a: _agent_r1_metrics(records, a) for a in AGENT_NAMES}
    debate_metrics = {a: _agent_debate_metrics(records, a) for a in AGENT_NAMES}
    overall = _debate_overall_stats(records)

    summary = {
        "n_total": len(records),
        "n_path_a": overall["n_no_debate"],   # alias for report compatibility
        "n_path_b": overall["n_debate_1plus"],
        "initial_round": initial_round,
        "debate_participation": debate_metrics,
        "debate_overall": overall,
    }

    _plot_agent_top1(initial_round, plots_dir / "agent_top1.png")
    _plot_debate_stats(debate_metrics, plots_dir / "agent_debate_stats.png")

    # Write debate stats as separate file for render_html
    with open(out_dir / "debate_stats.json", "w") as f:
        json.dump(overall, f, indent=2)

    with open(out_dir / "agent_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[agents] initial round top-1: "
          + "  ".join(f"{a}={initial_round[a]['top1_accuracy']:.1%}" for a in AGENT_NAMES))
    print(f"[agents] debate rate={overall['debate_rate']:.1%}  "
          f"revision_rate={overall.get('revision_rate_pairings') or 0:.1%}")
    return summary
