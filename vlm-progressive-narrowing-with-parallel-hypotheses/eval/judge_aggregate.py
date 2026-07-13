"""Aggregate per-image judge verdicts into a single summary.

Reads every ``<out>/judge/<image_id>.json`` and collapses them into:
  - per-agent role-adherence rate
  - per-agent argumentative-quality histogram
  - per-agent hallucination / visual_consistency / confidence_calibration stats
  - run-level constructive synthesis rate
  - NEW: region_narrowing_quality distribution (Path A vs B breakdown)
  - NEW: hypothesis_pool_quality distribution

Output: ``<out>/judge_summary.json`` and ``<out>/plots/judge_*.png``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval._style import STYLE, setup_plot_style
from eval.loader import AGENT_NAMES


QUALITY_LEVELS = ("very_weak", "weak", "normal", "strong", "very_strong")


def _empty_per_agent() -> dict[str, dict]:
    return {
        a: {
            "n": 0,
            "role_adherence_true": 0,
            "argumentative_quality": {q: 0 for q in QUALITY_LEVELS},
            "hallucination_scores": [],
            "visual_consistency_scores": [],
            "confidence_calibration_scores": [],
            "hallucination_examples": [],
        }
        for a in AGENT_NAMES
    }


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "stdev": None, "min": None, "max": None}
    arr = np.array(xs, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "stdev": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _aggregate(judge_dir: Path) -> dict:
    per_agent = _empty_per_agent()
    n_total = 0
    n_ok = 0
    n_errors: Counter = Counter()
    constructive_true = 0
    constructive_n = 0

    region_narrowing_scores: list[float] = []
    region_narrowing_path_a: list[float] = []
    region_narrowing_path_b: list[float] = []
    hypothesis_pool_scores: list[float] = []
    hypothesis_pool_in_pool: list[float] = []
    hypothesis_pool_not_in_pool: list[float] = []

    for path in sorted(judge_dir.glob("*.json")):
        n_total += 1
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception:
            n_errors["read_error"] += 1
            continue

        if "error" in payload:
            err = payload.get("error", "")
            if "image not found" in err:
                n_errors["no_image"] += 1
            elif "encode" in err:
                n_errors["encode_error"] += 1
            elif "llm" in err:
                n_errors["llm_error"] += 1
            else:
                n_errors["other_error"] += 1
            continue

        verdict = payload.get("verdict") or {}
        if not verdict:
            n_errors["missing_verdict"] += 1
            continue

        n_ok += 1
        run_path = payload.get("path", "")
        truth_in_pool = bool(payload.get("truth_in_hypothesis_pool", False))

        if "constructive_synthesis" in verdict:
            constructive_n += 1
            if verdict["constructive_synthesis"]:
                constructive_true += 1

        # PN-specific scores
        rn = verdict.get("region_narrowing_quality")
        if isinstance(rn, (int, float)):
            region_narrowing_scores.append(float(rn))
            if run_path == "A":
                region_narrowing_path_a.append(float(rn))
            elif run_path == "B":
                region_narrowing_path_b.append(float(rn))

        hp = verdict.get("hypothesis_pool_quality")
        if isinstance(hp, (int, float)):
            hypothesis_pool_scores.append(float(hp))
            if truth_in_pool:
                hypothesis_pool_in_pool.append(float(hp))
            else:
                hypothesis_pool_not_in_pool.append(float(hp))

        role = verdict.get("role_adherence") or {}
        qual = verdict.get("argumentative_quality") or {}
        hall = verdict.get("hallucination_score") or {}
        hall_examples = verdict.get("hallucination_examples") or {}
        vis = verdict.get("visual_consistency_score") or {}
        calib = verdict.get("confidence_calibration") or {}
        image_id = payload.get("image_id") or path.stem

        for agent in AGENT_NAMES:
            bucket = per_agent[agent]
            if agent in role:
                bucket["n"] += 1
                if bool(role[agent]):
                    bucket["role_adherence_true"] += 1
            q = (qual.get(agent) or "").lower()
            if q in bucket["argumentative_quality"]:
                bucket["argumentative_quality"][q] += 1
            h = hall.get(agent)
            if isinstance(h, (int, float)):
                bucket["hallucination_scores"].append(float(h))
            v = vis.get(agent)
            if isinstance(v, (int, float)):
                bucket["visual_consistency_scores"].append(float(v))
            c = calib.get(agent)
            if isinstance(c, (int, float)):
                bucket["confidence_calibration_scores"].append(float(c))
            ex = hall_examples.get(agent)
            if isinstance(ex, list) and ex and len(bucket["hallucination_examples"]) < 5:
                for s in ex[:2]:
                    if isinstance(s, str) and s.strip():
                        bucket["hallucination_examples"].append({
                            "image_id": image_id,
                            "example": s.strip(),
                            "score": float(h) if isinstance(h, (int, float)) else None,
                        })
                        if len(bucket["hallucination_examples"]) >= 5:
                            break

    summary_per_agent: dict[str, dict] = {}
    for agent, b in per_agent.items():
        n = b["n"]
        summary_per_agent[agent] = {
            "n": n,
            "role_adherence_rate": (b["role_adherence_true"] / n) if n else 0.0,
            "argumentative_quality": b["argumentative_quality"],
            "hallucination_score": _stats(b["hallucination_scores"]),
            "visual_consistency_score": _stats(b["visual_consistency_scores"]),
            "confidence_calibration_score": _stats(b["confidence_calibration_scores"]),
            "hallucination_examples": b["hallucination_examples"],
        }

    return {
        "n_total_judge_files": n_total,
        "n_with_verdict": n_ok,
        "errors": dict(n_errors),
        "constructive_synthesis_rate": (
            constructive_true / constructive_n if constructive_n else 0.0
        ),
        "constructive_synthesis_n": constructive_n,
        "region_narrowing_quality": {
            "all": _stats(region_narrowing_scores),
            "path_a": _stats(region_narrowing_path_a),
            "path_b": _stats(region_narrowing_path_b),
        },
        "hypothesis_pool_quality": {
            "all": _stats(hypothesis_pool_scores),
            "when_truth_in_pool": _stats(hypothesis_pool_in_pool),
            "when_truth_not_in_pool": _stats(hypothesis_pool_not_in_pool),
        },
        "per_agent": summary_per_agent,
    }


def _plot_role_adherence(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    rates = [summary["per_agent"][a]["role_adherence_rate"] for a in AGENT_NAMES]
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, rates, color=STYLE.primary, edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("role-adherence rate")
    ax.set_ylim(0, 1)
    ax.set_title("Per-agent role adherence (judge verdict)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_quality_histogram(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(QUALITY_LEVELS))
    width = 0.15
    for i, agent in enumerate(AGENT_NAMES):
        vals = [summary["per_agent"][agent]["argumentative_quality"][q] for q in QUALITY_LEVELS]
        ax.bar(x + (i - 2) * width, vals, width, label=agent)
    ax.set_xticks(x)
    ax.set_xticklabels(QUALITY_LEVELS)
    ax.set_ylabel("count")
    ax.set_title("Argumentative quality (judge verdict)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_score_distribution(summary: dict, key: str, title: str, ylabel: str, out_path: Path) -> None:
    setup_plot_style()
    means, stds, ns = [], [], []
    for a in AGENT_NAMES:
        st = summary["per_agent"][a].get(key) or {}
        means.append(st.get("mean") if st.get("mean") is not None else 0.0)
        stds.append(st.get("stdev") if st.get("stdev") is not None else 0.0)
        ns.append(st.get("n") or 0)
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x, means, yerr=stds, capsize=4, color=STYLE.primary, edgecolor="black", alpha=0.85)
    for i, n in enumerate(ns):
        ax.text(i, (means[i] or 0) + (stds[i] or 0) + 0.02, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_pn_scores(summary: dict, out_path: Path) -> None:
    """Side-by-side: region_narrowing_quality (Path A vs B) + hypothesis_pool_quality."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Region narrowing
    ax = axes[0]
    rn = summary.get("region_narrowing_quality") or {}
    labels = ["All", "Path A\n(consensus)", "Path B\n(no consensus)"]
    keys = ["all", "path_a", "path_b"]
    means = [rn.get(k, {}).get("mean") or 0 for k in keys]
    stds = [rn.get(k, {}).get("stdev") or 0 for k in keys]
    ns = [rn.get(k, {}).get("n") or 0 for k in keys]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=[STYLE.neutral, STYLE.success, STYLE.warning],
                  edgecolor="black", alpha=0.85)
    for i, n in enumerate(ns):
        ax.text(i, means[i] + stds[i] + 0.02, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean score [0,1]")
    ax.set_title("Region narrowing quality")

    # Hypothesis pool
    ax = axes[1]
    hp = summary.get("hypothesis_pool_quality") or {}
    labels2 = ["All", "Truth in pool", "Truth NOT in pool"]
    keys2 = ["all", "when_truth_in_pool", "when_truth_not_in_pool"]
    means2 = [hp.get(k, {}).get("mean") or 0 for k in keys2]
    stds2 = [hp.get(k, {}).get("stdev") or 0 for k in keys2]
    ns2 = [hp.get(k, {}).get("n") or 0 for k in keys2]
    x2 = np.arange(len(labels2))
    ax.bar(x2, means2, yerr=stds2, capsize=4,
           color=[STYLE.neutral, STYLE.success, STYLE.error],
           edgecolor="black", alpha=0.85)
    for i, n in enumerate(ns2):
        ax.text(i, means2[i] + stds2[i] + 0.02, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(x2)
    ax.set_xticklabels(labels2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean score [0,1]")
    ax.set_title("Hypothesis pool quality")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(out_dir: Path) -> dict:
    judge_dir = out_dir / "judge"
    if not judge_dir.is_dir():
        raise FileNotFoundError(f"no judge directory at {judge_dir}")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = _aggregate(judge_dir)

    _plot_role_adherence(summary, plots_dir / "judge_role_adherence.png")
    _plot_quality_histogram(summary, plots_dir / "judge_quality.png")
    _plot_score_distribution(
        summary, "hallucination_score",
        "Per-agent hallucination score (0=clean, 1=severe)",
        "mean hallucination score",
        plots_dir / "judge_hallucination.png",
    )
    _plot_score_distribution(
        summary, "visual_consistency_score",
        "Per-agent visual consistency (1=claims match image)",
        "mean visual consistency",
        plots_dir / "judge_visual_consistency.png",
    )
    _plot_score_distribution(
        summary, "confidence_calibration_score",
        "Per-agent confidence calibration",
        "mean calibration score",
        plots_dir / "judge_confidence_calibration.png",
    )
    _plot_pn_scores(summary, plots_dir / "judge_pn_scores.png")

    out_file = out_dir / "judge_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[judge_aggregate] wrote {out_file}")
    print(
        f"[judge_aggregate] verdicts={summary['n_with_verdict']}/"
        f"{summary['n_total_judge_files']}  "
        f"constructive={summary['constructive_synthesis_rate']:.1%}"
    )
    rn = summary.get("region_narrowing_quality", {}).get("all") or {}
    hp = summary.get("hypothesis_pool_quality", {}).get("all") or {}
    if rn.get("n"):
        print(f"[judge_aggregate] region_narrowing mean={rn['mean']:.3f} n={rn['n']}")
    if hp.get("n"):
        print(f"[judge_aggregate] hypothesis_pool  mean={hp['mean']:.3f} n={hp['n']}")
    for agent in AGENT_NAMES:
        a = summary["per_agent"][agent]
        hall = a["hallucination_score"]
        vis = a["visual_consistency_score"]
        hall_mean = hall["mean"] if hall["mean"] is not None else float("nan")
        vis_mean = vis["mean"] if vis["mean"] is not None else float("nan")
        print(
            f"[judge_aggregate] {agent:11s} role={a['role_adherence_rate']:.1%}  "
            f"hall={hall_mean:.2f}  vis={vis_mean:.2f}  n={a['n']}"
        )
    return summary
