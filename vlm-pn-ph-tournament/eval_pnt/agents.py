"""Per-agent metrics.

For each of the 5 specialist agents (linguistic, landscape, botanics, regulatory,
meta) and each round (initial assessments, country-level Path-B reassessments):

  - Top-1 accuracy: agent's #1 candidate vs. ground truth
  - Top-3 hit rate: truth in agent's first 3 candidates
  - Coverage: truth appears anywhere in candidates
  - Confidence calibration: per-confidence-bin accuracy

Also: per-region and per-confirmed-region performance to surface specialization.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlm_council.evaluate import _countries_match
from eval_pnt._style import STYLE, setup_plot_style
from eval_pnt.loader import AGENT_NAMES, RunRecord, countries_match, load_run, top1_country, topk_countries


CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low", "speculative")


def _matches_truth(country: str | None, truth_code: str) -> bool:
    if not country:
        return False
    return countries_match(country, truth_code)


def _agent_round_metrics(records: list[RunRecord], round_attr: str) -> dict[str, dict]:
    """``round_attr`` is either ``"assessments"`` or ``"country_assessments"``."""
    per_agent: dict[str, dict] = {}

    for agent in AGENT_NAMES:
        n = top1_correct = top3_correct = covered = 0
        per_conf_total: Counter = Counter()
        per_conf_correct: Counter = Counter()
        per_region_total: Counter = Counter()
        per_region_correct: Counter = Counter()

        for r in records:
            assessment = (getattr(r, round_attr) or {}).get(agent)
            if not assessment:
                continue
            cands = assessment.get("candidates") or []
            if not cands:
                continue
            n += 1

            top1 = top1_country(assessment)
            if _matches_truth(top1, r.truth_country_code):
                top1_correct += 1

            top3 = topk_countries(assessment, k=3)
            if any(_matches_truth(c, r.truth_country_code) for c in top3):
                top3_correct += 1

            all_cands = topk_countries(assessment, k=len(cands))
            if any(_matches_truth(c, r.truth_country_code) for c in all_cands):
                covered += 1

            # Confidence calibration on the top-1
            top1_conf = (cands[0].get("confidence") or "").strip().lower()
            if top1_conf in CONFIDENCE_LEVELS:
                per_conf_total[top1_conf] += 1
                if _matches_truth(top1, r.truth_country_code):
                    per_conf_correct[top1_conf] += 1

            # Per-region (use confirmed_region; falls back to "unknown")
            region = r.confirmed_region or "unknown"
            per_region_total[region] += 1
            if _matches_truth(top1, r.truth_country_code):
                per_region_correct[region] += 1

        per_agent[agent] = {
            "n": n,
            "top1_accuracy": top1_correct / n if n else 0.0,
            "top3_hit_rate": top3_correct / n if n else 0.0,
            "coverage": covered / n if n else 0.0,
            "calibration": {
                conf: {
                    "n": per_conf_total[conf],
                    "accuracy": (per_conf_correct[conf] / per_conf_total[conf])
                    if per_conf_total[conf] else 0.0,
                }
                for conf in CONFIDENCE_LEVELS
            },
            "per_region": {
                region: {
                    "n": per_region_total[region],
                    "top1_accuracy": (per_region_correct[region] / per_region_total[region])
                    if per_region_total[region] else 0.0,
                }
                for region in per_region_total
            },
        }

    return per_agent


def compute(records: list[RunRecord]) -> dict:
    return {
        "initial_round": _agent_round_metrics(records, "assessments"),
        "country_round_path_b": _agent_round_metrics(
            [r for r in records if r.path == "B"], "country_assessments"
        ),
        "n_path_a": sum(1 for r in records if r.path == "A"),
        "n_path_b": sum(1 for r in records if r.path == "B"),
    }


def plot_top1_bar(metrics: dict, out_path: Path) -> None:
    setup_plot_style()
    initial = metrics["initial_round"]
    country = metrics["country_round_path_b"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(AGENT_NAMES)))
    width = 0.35
    initial_vals = [initial[a]["top1_accuracy"] for a in AGENT_NAMES]
    country_vals = [country[a]["top1_accuracy"] for a in AGENT_NAMES]
    ax.bar([i - width / 2 for i in x], initial_vals, width,
           label="initial round", color=STYLE.primary)
    ax.bar([i + width / 2 for i in x], country_vals, width,
           label="country round (Path B)", color=STYLE.warning)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Per-agent top-1 country accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_calibration(metrics: dict, out_path: Path) -> None:
    """For the initial round only, per-confidence-bin accuracy across agents."""
    setup_plot_style()
    initial = metrics["initial_round"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = list(range(len(CONFIDENCE_LEVELS)))
    width = 0.15
    for i, agent in enumerate(AGENT_NAMES):
        vals = [initial[agent]["calibration"][c]["accuracy"] for c in CONFIDENCE_LEVELS]
        ax.bar([j + (i - 2) * width for j in x], vals, width, label=agent)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIDENCE_LEVELS)
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Confidence calibration (initial round)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(results_dir: Path, gt_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    records = load_run(results_dir, gt_csv)
    metrics = compute(records)

    plot_top1_bar(metrics, plots_dir / "agent_top1.png")
    plot_calibration(metrics, plots_dir / "agent_calibration.png")

    out_file = out_dir / "agent_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[agents] wrote {out_file}")
    for agent in AGENT_NAMES:
        m = metrics["initial_round"][agent]
        print(f"[agents] {agent:11s} initial: top1={m['top1_accuracy']:.1%}  "
              f"top3={m['top3_hit_rate']:.1%}  coverage={m['coverage']:.1%}  n={m['n']}")
    return metrics
