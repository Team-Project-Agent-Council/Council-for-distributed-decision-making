"""Per-agent metrics for Hub-and-Spoke evaluation.

Covers:
- Initial round top-1 accuracy, top-3 hit rate, coverage, n
- Discussion participation stats (how often each agent was questioned)
- Response update quality (did agent change top-1 when questioned)

Outputs:
  agent_metrics.json
  plots/agent_initial_top1.png
  plots/agent_discussion_rate.png
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_hubspoke._style import STYLE, setup_plot_style
from eval_hubspoke.loader import (
    AGENT_NAMES,
    RunRecord,
    countries_match,
    load_run,
    parse_discussion_for_agent,
    top1_country,
    topk_countries,
    _parse_response_candidates,
    _normalize_country,
)


# ---------------------------------------------------------------------------
# Wilson confidence interval (z=1.96, 95%)
# ---------------------------------------------------------------------------

def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _matches_truth(country: str | None, truth_code: str) -> bool:
    if not country:
        return False
    return countries_match(country, truth_code)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(records: list[RunRecord]) -> dict:
    """Compute per-agent initial-round + discussion metrics."""
    per_agent: dict[str, dict] = {}

    # Discussion participation per image per agent
    # discussion_rate = how often each agent was targeted / n images
    # response_update_rate = fraction of times agent changed top-1 when questioned

    for agent in AGENT_NAMES:
        n = 0
        top1_correct = 0
        top3_correct = 0
        covered = 0

        # Discussion stats
        n_questioned = 0            # images where this agent was asked at least once
        n_changed_top1 = 0          # agent changed their top-1 after being questioned
        n_questionable = 0          # images where agent had initial assessment AND was questioned

        for r in records:
            assessment = (r.assessments or {}).get(agent)
            if not assessment:
                continue
            cands = assessment.get("candidates") or []
            if not cands:
                continue
            n += 1

            t1 = top1_country(assessment)
            if _matches_truth(t1, r.truth_country_code):
                top1_correct += 1

            t3 = topk_countries(assessment, k=3)
            if any(_matches_truth(c, r.truth_country_code) for c in t3):
                top3_correct += 1

            all_c = topk_countries(assessment, k=len(cands))
            if any(_matches_truth(c, r.truth_country_code) for c in all_c):
                covered += 1

            # Discussion participation
            disc_entries = parse_discussion_for_agent(r, agent)
            if disc_entries:
                n_questioned += 1
                n_questionable += 1
                # Check if agent updated their top-1
                initial_t1 = t1
                # Use the last response that has non-empty agent_response
                updated = False
                for entry in disc_entries:
                    resp = entry.get("agent_response", "") or ""
                    if resp.strip():
                        new_cands = _parse_response_candidates(resp)
                        if new_cands:
                            new_t1 = _normalize_country(new_cands[0].get("country", ""))
                            if new_t1 and new_t1 != (initial_t1 or ""):
                                updated = True
                                break
                if updated:
                    n_changed_top1 += 1

        ci_low, ci_high = _wilson_ci(top1_correct, n)
        per_agent[agent] = {
            "n": n,
            "top1_accuracy": top1_correct / n if n else 0.0,
            "top1_accuracy_ci_low": ci_low,
            "top1_accuracy_ci_high": ci_high,
            "top3_hit_rate": top3_correct / n if n else 0.0,
            "coverage": covered / n if n else 0.0,
            "n_questioned": n_questioned,
            "discussion_rate": n_questioned / len(records) if records else 0.0,
            "n_questionable": n_questionable,
            "n_changed_top1_when_questioned": n_changed_top1,
            "response_update_rate": (
                n_changed_top1 / n_questionable if n_questionable else 0.0
            ),
        }

    # Overall discussion stats
    total_discussion_entries = sum(
        len(r.discussion_log) for r in records
    )
    mean_discussion_rounds = (
        sum(r.discussion_rounds for r in records) / len(records) if records else 0.0
    )
    n_with_discussion = sum(1 for r in records if r.discussion_rounds > 0)

    return {
        "initial_round": per_agent,
        "n_images": len(records),
        "n_with_discussion": n_with_discussion,
        "mean_discussion_rounds": mean_discussion_rounds,
        "total_discussion_entries": total_discussion_entries,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_agent_initial_top1(metrics: dict, out_path: Path) -> None:
    setup_plot_style()
    initial = metrics.get("initial_round", {})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(AGENT_NAMES))
    vals = [initial.get(a, {}).get("top1_accuracy", 0.0) for a in AGENT_NAMES]
    ci_lows = [initial.get(a, {}).get("top1_accuracy_ci_low", 0.0) for a in AGENT_NAMES]
    ci_highs = [initial.get(a, {}).get("top1_accuracy_ci_high", 0.0) for a in AGENT_NAMES]
    yerr_low = [max(0.0, v - lo) for v, lo in zip(vals, ci_lows)]
    yerr_high = [max(0.0, hi - v) for v, hi in zip(vals, ci_highs)]
    ax.bar(x, vals, yerr=[yerr_low, yerr_high], capsize=5, color=STYLE.primary,
           edgecolor="black", alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + yerr_high[i] + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-agent initial top-1 accuracy (Hub-and-Spoke)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_agent_discussion_rate(metrics: dict, out_path: Path) -> None:
    setup_plot_style()
    initial = metrics.get("initial_round", {})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(AGENT_NAMES))
    disc_rates = [initial.get(a, {}).get("discussion_rate", 0.0) for a in AGENT_NAMES]
    update_rates = [initial.get(a, {}).get("response_update_rate", 0.0) for a in AGENT_NAMES]
    width = 0.35
    ax.bar(x - width / 2, disc_rates, width, label="discussion rate (per image)",
           color=STYLE.primary, edgecolor="black", alpha=0.85)
    ax.bar(x + width / 2, update_rates, width, label="top-1 changed when questioned",
           color=STYLE.warning, edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-agent discussion participation & response update (Hub-and-Spoke)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(results_dir: Path, gt_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    records = load_run(results_dir, gt_csv)
    metrics = compute(records)

    plot_agent_initial_top1(metrics, plots_dir / "agent_initial_top1.png")
    plot_agent_discussion_rate(metrics, plots_dir / "agent_discussion_rate.png")

    out_file = out_dir / "agent_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[agents] wrote {out_file}")
    print(f"[agents] n={metrics['n_images']}, mean_discussion_rounds={metrics['mean_discussion_rounds']:.2f}")
    for agent in AGENT_NAMES:
        m = metrics["initial_round"].get(agent, {})
        print(
            f"[agents] {agent:11s} top1={m.get('top1_accuracy', 0):.1%}  "
            f"top3={m.get('top3_hit_rate', 0):.1%}  coverage={m.get('coverage', 0):.1%}  "
            f"disc_rate={m.get('discussion_rate', 0):.1%}  n={m.get('n', 0)}"
        )
    return metrics
