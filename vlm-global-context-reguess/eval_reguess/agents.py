"""Per-agent metrics for both rounds of the re-guess approach.

For each of the 5 specialist agents (linguistic, landscape, botanics, regulatory,
meta) and each round (R1 = round_1_assessments, R2 = round_2_assessments):

  - top1_accuracy: agent's #1 candidate vs. ground truth
  - top3_hit_rate: truth in agent's first 3 candidates
  - coverage: fraction with at least 1 candidate
  - n: number of images where agent had an assessment

Also:
  - change_rate: fraction of images where agent's top-1 changed R1 → R2
  - confidence_shift: fraction where R2 top-1 confidence > R1 top-1 confidence
  (high > medium > low > speculative)

Outputs:
  agent_metrics.json
  plots/agent_top1_r1_vs_r2.png
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_reguess._style import STYLE, setup_plot_style
from eval_reguess.loader import (
    AGENT_NAMES, RunRecord, countries_match, load_run, top1_country, topk_countries,
)


CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low", "speculative")
_CONF_RANK = {c: i for i, c in enumerate(CONFIDENCE_LEVELS)}  # lower index = higher confidence


def _matches_truth(country: str | None, truth_code: str) -> bool:
    if not country:
        return False
    return countries_match(country, truth_code)


def _conf_rank(conf: str) -> int:
    """Return confidence rank (0=high, 3=speculative). Unknown → 99."""
    return _CONF_RANK.get((conf or "").strip().lower(), 99)


def _agent_round_metrics(
    records: list[RunRecord],
    round_attr: str,
) -> dict[str, dict]:
    """Compute per-agent metrics for a single round.

    round_attr: 'r1_assessments' or 'r2_assessments'
    """
    per_agent: dict[str, dict] = {}

    for agent in AGENT_NAMES:
        n = top1_correct = top3_correct = covered = 0

        for r in records:
            assessments = getattr(r, round_attr) or {}
            assessment = assessments.get(agent)
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

        per_agent[agent] = {
            "n": n,
            "top1_accuracy": top1_correct / n if n else 0.0,
            "top3_hit_rate": top3_correct / n if n else 0.0,
            "coverage": covered / n if n else 0.0,
        }

    return per_agent


def _compute_change_metrics(records: list[RunRecord]) -> dict[str, dict]:
    """Compute change_rate and confidence_shift per agent (R1 → R2)."""
    per_agent: dict[str, dict] = {}

    for agent in AGENT_NAMES:
        n = changed = conf_improved = 0

        for r in records:
            r1 = (r.r1_assessments or {}).get(agent)
            r2 = (r.r2_assessments or {}).get(agent)
            if not r1 or not r2:
                continue
            r1_cands = r1.get("candidates") or []
            r2_cands = r2.get("candidates") or []
            if not r1_cands or not r2_cands:
                continue
            n += 1

            # Top-1 change
            r1_top1 = top1_country(r1) or ""
            r2_top1 = top1_country(r2) or ""
            if r1_top1 != r2_top1:
                changed += 1

            # Confidence shift
            r1_conf = (r1_cands[0].get("confidence") or "").strip().lower()
            r2_conf = (r2_cands[0].get("confidence") or "").strip().lower()
            # Improved = lower rank number (higher confidence)
            if _conf_rank(r2_conf) < _conf_rank(r1_conf):
                conf_improved += 1

        per_agent[agent] = {
            "n_paired": n,
            "change_rate": changed / n if n else 0.0,
            "confidence_shift_rate": conf_improved / n if n else 0.0,
        }

    return per_agent


def compute(records: list[RunRecord]) -> dict:
    r1_metrics = _agent_round_metrics(records, "r1_assessments")
    r2_metrics = _agent_round_metrics(records, "r2_assessments")
    change_metrics = _compute_change_metrics(records)
    return {
        "round_1": r1_metrics,
        "round_2": r2_metrics,
        "change_metrics": change_metrics,
        "n_total": len(records),
    }


def plot_top1_r1_vs_r2(metrics: dict, out_path: Path) -> None:
    """Grouped bar chart: R1 vs R2 top-1 accuracy per agent."""
    setup_plot_style()
    r1 = metrics["round_1"]
    r2 = metrics["round_2"]
    x = list(range(len(AGENT_NAMES)))
    width = 0.35
    r1_vals = [r1[a]["top1_accuracy"] for a in AGENT_NAMES]
    r2_vals = [r2[a]["top1_accuracy"] for a in AGENT_NAMES]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - width / 2 for i in x], r1_vals, width, label="Round 1", color=STYLE.primary)
    ax.bar([i + width / 2 for i in x], r2_vals, width, label="Round 2", color=STYLE.warning)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Per-agent top-1 accuracy: Round 1 vs Round 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(results_dir: Path, gt_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    records = load_run(results_dir, gt_csv)
    metrics = compute(records)

    plot_top1_r1_vs_r2(metrics, plots_dir / "agent_top1_r1_vs_r2.png")

    out_file = out_dir / "agent_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[agents] wrote {out_file}")

    r1 = metrics["round_1"]
    r2 = metrics["round_2"]
    chg = metrics["change_metrics"]
    for agent in AGENT_NAMES:
        print(
            f"[agents] {agent:11s} "
            f"R1 top1={r1[agent]['top1_accuracy']:.1%}  "
            f"R2 top1={r2[agent]['top1_accuracy']:.1%}  "
            f"change={chg[agent]['change_rate']:.1%}  "
            f"conf_shift={chg[agent]['confidence_shift_rate']:.1%}"
        )
    return metrics
