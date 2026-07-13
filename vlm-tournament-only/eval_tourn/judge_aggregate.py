"""Aggregate per-image judge verdicts into a single summary.

Reads every ``<out>/judge/<image_id>.json`` written by ``eval.judge`` and
collapses them into:

  - per-agent role-adherence rate
  - per-agent argumentative-quality histogram
  - per-agent assertiveness histogram
  - per-agent hallucination_score distribution (mean, median, stdev)
  - per-agent visual_consistency_score distribution
  - per-agent confidence_calibration distribution
  - run-level overall_quality_score distribution
  - tournament_failure.failure_reason histogram
  - count of skipped / errored images (image not found, encode failure, llm failure)

Output: ``<out>/judge_summary.json`` and ``<out>/plots/judge_*.png``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_tourn._style import STYLE, setup_plot_style
from eval_tourn.loader import AGENT_NAMES


QUALITY_LEVELS = ("very_weak", "weak", "normal", "strong", "very_strong")
ASSERT_LEVELS = ("very_low", "low", "medium", "high", "very_high")
FAILURE_REASONS = (
    "judge_misjudgment",
    "missing_rag_refs",
    "ambiguous_evidence",
    "agent_misled_in_tournament",
    "agent_misled_pre_pool",
    "not_applicable",
    "other",
)


def _empty_per_agent() -> dict[str, dict]:
    return {
        a: {
            "n": 0,
            "role_adherence_true": 0,
            "argumentative_quality": {q: 0 for q in QUALITY_LEVELS},
            "assertiveness": {a_: 0 for a_ in ASSERT_LEVELS},
            "hallucination_scores": [],
            "hallucination_examples": [],
            "visual_consistency_scores": [],
            "confidence_calibration_scores": [],
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


def _load_agent_abstentions(results_dir: Path | None) -> dict[str, set[str]]:
    """Map image_id -> set of agents that produced no candidates in the council run.

    Role adherence is scored only over runs in which an agent actually
    contributed an assessment. When an agent abstains (for example the
    linguistic agent on an image without any legible text), the judge may or
    may not flag this as a role violation, so counting these runs would
    conflate "stayed silent when there was nothing to say" with "reasoned
    outside its domain". Excluding abstentions keeps the metric comparable
    across agents and approaches. Returns an empty map if ``results_dir`` is
    not available, in which case no run is excluded.
    """
    abstentions: dict[str, set[str]] = {}
    if results_dir is None or not Path(results_dir).is_dir():
        return abstentions
    for result_dir in sorted(Path(results_dir).iterdir()):
        if not result_dir.is_dir():
            continue
        f = result_dir / "result.json"
        if not f.exists():
            continue
        try:
            with open(f) as fp:
                raw = json.load(fp)
        except Exception:
            continue
        image_id = result_dir.name
        assessments = raw.get("assessments", {}) or {}
        empty = {
            agent
            for agent in AGENT_NAMES
            if not (assessments.get(agent, {}) or {}).get("candidates")
        }
        if empty:
            abstentions[image_id] = empty
    return abstentions


def _aggregate(judge_dir: Path, results_dir: Path | None = None) -> dict:
    per_agent = _empty_per_agent()
    abstentions = _load_agent_abstentions(results_dir)
    n_total = 0
    n_ok = 0
    n_errors: Counter = Counter()
    constructive_true = 0
    constructive_n = 0
    correct_runs = 0

    overall_quality_scores: list[float] = []
    failure_reason_counter: Counter = Counter()
    failure_examples: list[dict] = []
    counterfactual_winnable_count = 0
    counterfactual_n = 0

    role_adherence_by_correctness: dict[str, dict[str, list[bool]]] = {
        a: {"correct_run": [], "wrong_run": []} for a in AGENT_NAMES
    }

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
        is_correct = bool(payload.get("is_correct"))
        image_id = payload.get("image_id") or path.stem
        if is_correct:
            correct_runs += 1

        if "constructive_synthesis" in verdict:
            constructive_n += 1
            if verdict["constructive_synthesis"]:
                constructive_true += 1

        # v2: overall quality
        oq = verdict.get("overall_quality_score")
        if isinstance(oq, (int, float)):
            overall_quality_scores.append(float(oq))

        # v2: tournament failure
        tf = verdict.get("tournament_failure") or {}
        if isinstance(tf, dict):
            reason = tf.get("failure_reason")
            truth_in_pool = tf.get("truth_in_pool")
            # Deterministic remap: split agent_misled by where the failure
            # happened, and reclassify "not_applicable + truth missing from
            # pool" as a pre-pool agent failure (truth never made it in).
            if reason == "agent_misled":
                reason = (
                    "agent_misled_in_tournament" if truth_in_pool
                    else "agent_misled_pre_pool"
                )
            elif reason == "not_applicable" and truth_in_pool is False:
                reason = "agent_misled_pre_pool"
            if reason:
                failure_reason_counter[str(reason)] += 1
            if reason and reason != "not_applicable":
                cf = tf.get("counterfactual_winnable")
                if isinstance(cf, bool):
                    counterfactual_n += 1
                    if cf:
                        counterfactual_winnable_count += 1
                if len(failure_examples) < 20:
                    failure_examples.append({
                        "image_id": image_id,
                        "failure_reason": reason,
                        "truth_lost_to": tf.get("truth_lost_to"),
                        "failure_match_round": tf.get("failure_match_round"),
                        "failure_reasoning": tf.get("failure_reasoning"),
                        "counterfactual_winnable": tf.get("counterfactual_winnable"),
                    })

        role = verdict.get("role_adherence") or {}
        qual = verdict.get("argumentative_quality") or {}
        asse = verdict.get("assertiveness") or {}

        # v2 per-agent score maps
        hall = verdict.get("hallucination_score") or {}
        hall_examples = verdict.get("hallucination_examples") or {}
        vis = verdict.get("visual_consistency_score") or {}
        calib = verdict.get("confidence_calibration") or {}

        for agent in AGENT_NAMES:
            bucket = per_agent[agent]
            abstained = agent in abstentions.get(image_id, set())
            if agent in role and not abstained:
                bucket["n"] += 1
                if bool(role[agent]):
                    bucket["role_adherence_true"] += 1
                role_adherence_by_correctness[agent][
                    "correct_run" if is_correct else "wrong_run"
                ].append(bool(role[agent]))
            q = (qual.get(agent) or "").lower()
            if q in bucket["argumentative_quality"]:
                bucket["argumentative_quality"][q] += 1
            a_ = (asse.get(agent) or "").lower()
            if a_ in bucket["assertiveness"]:
                bucket["assertiveness"][a_] += 1

            # v2 quantitative scores
            h = hall.get(agent)
            if isinstance(h, (int, float)):
                bucket["hallucination_scores"].append(float(h))
            v = vis.get(agent)
            if isinstance(v, (int, float)):
                bucket["visual_consistency_scores"].append(float(v))
            c = calib.get(agent)
            if isinstance(c, (int, float)):
                bucket["confidence_calibration_scores"].append(float(c))

            # Top hallucination examples (up to 5 per agent across the run)
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

    # Derive per-agent rates + stats
    summary_per_agent: dict[str, dict] = {}
    for agent, b in per_agent.items():
        n = b["n"]
        ra_correct = role_adherence_by_correctness[agent]["correct_run"]
        ra_wrong = role_adherence_by_correctness[agent]["wrong_run"]
        summary_per_agent[agent] = {
            "n": n,
            "role_adherence_rate": (b["role_adherence_true"] / n) if n else 0.0,
            "role_adherence_when_run_correct": (
                sum(ra_correct) / len(ra_correct) if ra_correct else 0.0
            ),
            "role_adherence_when_run_wrong": (
                sum(ra_wrong) / len(ra_wrong) if ra_wrong else 0.0
            ),
            "argumentative_quality": b["argumentative_quality"],
            "assertiveness": b["assertiveness"],
            "hallucination_score": _stats(b["hallucination_scores"]),
            "visual_consistency_score": _stats(b["visual_consistency_scores"]),
            "confidence_calibration_score": _stats(b["confidence_calibration_scores"]),
            "hallucination_examples": b["hallucination_examples"],
        }

    return {
        "n_total_judge_files": n_total,
        "n_with_verdict": n_ok,
        "n_correct_runs": correct_runs,
        "errors": dict(n_errors),
        "constructive_synthesis_rate": (
            constructive_true / constructive_n if constructive_n else 0.0
        ),
        "constructive_synthesis_n": constructive_n,
        "overall_quality_score": _stats(overall_quality_scores),
        "tournament_failure": {
            "by_reason": dict(failure_reason_counter),
            "counterfactual_winnable_rate": (
                counterfactual_winnable_count / counterfactual_n if counterfactual_n else 0.0
            ),
            "counterfactual_n": counterfactual_n,
            "examples": failure_examples,
        },
        "per_agent": summary_per_agent,
    }


# Plots

def _plot_role_adherence(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    rates = [summary["per_agent"][a]["role_adherence_rate"] for a in AGENT_NAMES]
    rates_correct = [
        summary["per_agent"][a]["role_adherence_when_run_correct"] for a in AGENT_NAMES
    ]
    rates_wrong = [
        summary["per_agent"][a]["role_adherence_when_run_wrong"] for a in AGENT_NAMES
    ]
    x = np.arange(len(AGENT_NAMES))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width, rates, width, label="overall", color=STYLE.primary)
    ax.bar(x, rates_correct, width, label="run correct", color=STYLE.success)
    ax.bar(x + width, rates_wrong, width, label="run wrong", color=STYLE.error)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("role-adherence rate")
    ax.set_ylim(0, 1)
    ax.set_title("Per-agent role adherence (judge verdict)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_quality_histogram(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(QUALITY_LEVELS))
    width = 0.15
    for i, agent in enumerate(AGENT_NAMES):
        vals = [
            summary["per_agent"][agent]["argumentative_quality"][q] for q in QUALITY_LEVELS
        ]
        ax.bar(x + (i - 2) * width, vals, width, label=agent)
    ax.set_xticks(x)
    ax.set_xticklabels(QUALITY_LEVELS)
    ax.set_ylabel("count")
    ax.set_title("Argumentative quality (judge verdict)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_score_distribution(
    summary: dict, key: str, title: str, ylabel: str, out_path: Path
) -> None:
    """Per-agent bar chart of mean ± stdev for a [0,1] quantitative score."""
    setup_plot_style()
    means = []
    stds = []
    ns = []
    for a in AGENT_NAMES:
        st = summary["per_agent"][a].get(key) or {}
        means.append(st.get("mean") if st.get("mean") is not None else 0.0)
        stds.append(st.get("stdev") if st.get("stdev") is not None else 0.0)
        ns.append(st.get("n") or 0)
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=STYLE.primary, edgecolor="black", alpha=0.85)
    for i, n in enumerate(ns):
        ax.text(i, (means[i] or 0) + (stds[i] or 0) + 0.02,
                f"n={n}", ha="center", fontsize=8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_failure_reasons(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    by_reason = (summary.get("tournament_failure") or {}).get("by_reason") or {}
    # Stable order
    reasons = list(FAILURE_REASONS)
    counts = [by_reason.get(r, 0) for r in reasons]
    if sum(counts) == 0:
        # Empty fallback
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "no tournament failures recorded",
                ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    colors = [
        STYLE.error, STYLE.warning, STYLE.neutral,
        STYLE.accent, STYLE.primary, STYLE.success, STYLE.secondary,
    ]
    ax.bar(reasons, counts, color=colors[: len(reasons)], edgecolor="black")
    ax.set_ylabel("count")
    ax.set_title("Tournament failure attribution (truth in pool but lost)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_overall_quality_distribution(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    st = summary.get("overall_quality_score") or {}
    n = st.get("n") or 0
    fig, ax = plt.subplots(figsize=(7, 4))
    if n == 0:
        ax.text(0.5, 0.5, "no overall_quality_score recorded",
                ha="center", va="center")
        ax.set_axis_off()
    else:
        # Render as a simple summary text box with mean/median/stdev
        msg = (
            f"n = {n}\n"
            f"mean   = {st.get('mean'):.3f}\n"
            f"median = {st.get('median'):.3f}\n"
            f"stdev  = {st.get('stdev'):.3f}\n"
            f"min    = {st.get('min'):.3f}\n"
            f"max    = {st.get('max'):.3f}"
        )
        ax.text(0.05, 0.5, msg, family="monospace", fontsize=11,
                va="center", ha="left",
                bbox={"facecolor": "#F4F4F4", "edgecolor": "#BBBBBB"})
        ax.set_axis_off()
    ax.set_title("Overall quality score (run-level)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(out_dir: Path, results_dir: Path | None = None) -> dict:
    judge_dir = out_dir / "judge"
    if not judge_dir.is_dir():
        raise FileNotFoundError(f"no judge directory at {judge_dir}")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = _aggregate(judge_dir, results_dir)

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
    _plot_failure_reasons(summary, plots_dir / "judge_failure_reasons.png")
    _plot_overall_quality_distribution(summary, plots_dir / "judge_overall_quality.png")

    out_file = out_dir / "judge_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[judge_aggregate] wrote {out_file}")
    print(
        f"[judge_aggregate] verdicts={summary['n_with_verdict']}/"
        f"{summary['n_total_judge_files']}  "
        f"constructive={summary['constructive_synthesis_rate']:.1%}"
    )
    oq = summary.get("overall_quality_score") or {}
    if oq.get("n"):
        print(
            f"[judge_aggregate] overall_quality mean={oq['mean']:.3f} "
            f"median={oq['median']:.3f} n={oq['n']}"
        )
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
