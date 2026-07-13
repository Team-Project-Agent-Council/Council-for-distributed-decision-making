"""Aggregate per-image judge JSONs into a single summary.

Reads every <out>/judge/<image_id>.json produced by judge.py (one file per
image, all five agents inside each verdict as dict[agent_name, value]) and
collapses them into judge_summary.json.

Legacy per-agent files (<image_id>_<agent>.json from before the pro-image
refactor) are detected via the ``agent_name`` key in the payload and skipped.

Outputs:
  judge_summary.json
  plots/judge_role_adherence.png
  plots/judge_hallucination.png
  plots/judge_question_relevance.png
  plots/judge_response_update.png
  plots/judge_strategy.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_hubspoke._style import STYLE, setup_plot_style
from eval_hubspoke.loader import AGENT_NAMES


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

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


# Scores that are only meaningful when the agent was actually questioned.
_QUESTION_ONLY_SCORE_KEYS = {
    "question_relevance_score",
    "response_update_quality",
}


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _aggregate(judge_dir: Path) -> dict:
    per_agent: dict[str, dict] = {
        a: {
            "n": 0,
            "role_adherence_true": 0,
            "hallucination_scores": [],
            "visual_consistency_scores": [],
            "confidence_calibration_scores": [],
            "question_relevance_scores": [],
            "response_update_scores": [],
            "n_questioned": 0,
            "n_addressed_question": 0,
            "n_dodged_question": 0,
            "hallucination_examples_raw": [],  # list of {image_id, example}
        }
        for a in AGENT_NAMES
    }

    judge_strategy_scores: list[float] = []
    judge_synthesis_scores: list[float] = []
    convergence_scores: list[float] = []
    convergence_by_rounds: dict[int, list[float]] = {}
    n_total = 0
    n_ok = 0
    n_errors: dict[str, int] = {}

    for path in sorted(judge_dir.glob("*.json")):
        n_total += 1
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception:
            n_errors["read_error"] = n_errors.get("read_error", 0) + 1
            continue

        # Skip legacy pro-agent files (pre-refactor: <id>_<agent>.json).
        if "agent_name" in payload:
            n_errors["legacy_skipped"] = n_errors.get("legacy_skipped", 0) + 1
            continue

        if "error" in payload:
            err = payload["error"]
            key = "parse_error" if "parse" in err else ("llm_error" if "llm" in err else "other_error")
            n_errors[key] = n_errors.get(key, 0) + 1
            continue

        verdict = payload.get("verdict") or {}
        if not verdict:
            n_errors["missing_verdict"] = n_errors.get("missing_verdict", 0) + 1
            continue

        n_ok += 1
        image_id = payload.get("image_id", path.stem)
        was_questioned_map = payload.get("was_questioned") or {}
        discussion_rounds = payload.get("discussion_rounds")

        # Per-agent scoring (pull from dict-shaped verdict fields)
        for agent in AGENT_NAMES:
            bucket = per_agent[agent]
            bucket["n"] += 1
            this_agent_questioned = bool(was_questioned_map.get(agent))

            ra_map = verdict.get("role_adherence") or {}
            if ra_map.get(agent):
                bucket["role_adherence_true"] += 1

            for score_key, bucket_key in [
                ("hallucination_score", "hallucination_scores"),
                ("visual_consistency_score", "visual_consistency_scores"),
                ("confidence_calibration_score", "confidence_calibration_scores"),
                ("question_relevance_score", "question_relevance_scores"),
                ("response_update_quality", "response_update_scores"),
            ]:
                if score_key in _QUESTION_ONLY_SCORE_KEYS and not this_agent_questioned:
                    continue
                val = (verdict.get(score_key) or {}).get(agent)
                if isinstance(val, (int, float)):
                    bucket[bucket_key].append(float(val))

            if this_agent_questioned:
                bucket["n_questioned"] += 1
                addressed = (verdict.get("targeted_agent_addressed_question") or {}).get(agent)
                if addressed is True:
                    bucket["n_addressed_question"] += 1
                elif addressed is False:
                    bucket["n_dodged_question"] += 1

            examples = (verdict.get("hallucination_examples") or {}).get(agent) or []
            if isinstance(examples, list) and examples:
                existing = bucket["hallucination_examples_raw"]
                for ex in examples:
                    if isinstance(ex, str) and ex.strip() and len(existing) < 10:
                        existing.append({"image_id": image_id, "example": ex.strip()})

        # Image-level scalars (one per image)
        strat = verdict.get("judge_question_strategy_score")
        synth = verdict.get("judge_synthesis_quality")
        conv = verdict.get("discussion_convergence_score")
        if isinstance(strat, (int, float)):
            judge_strategy_scores.append(float(strat))
        if isinstance(synth, (int, float)):
            judge_synthesis_scores.append(float(synth))
        if isinstance(conv, (int, float)):
            convergence_scores.append(float(conv))
            if isinstance(discussion_rounds, int):
                convergence_by_rounds.setdefault(discussion_rounds, []).append(float(conv))

    # Build summary
    summary_per_agent: dict[str, dict] = {}
    for agent, b in per_agent.items():
        n = b["n"]
        summary_per_agent[agent] = {
            "n": n,
            "role_adherence_rate": (b["role_adherence_true"] / n) if n else 0.0,
            "hallucination_score": _stats(b["hallucination_scores"]),
            "visual_consistency_score": _stats(b["visual_consistency_scores"]),
            "confidence_calibration_score": _stats(b["confidence_calibration_scores"]),
            "question_relevance_score": _stats(b["question_relevance_scores"]),
            "response_update_quality": _stats(b["response_update_scores"]),
            "n_questioned": b["n_questioned"],
            "n_addressed_question": b["n_addressed_question"],
            "n_dodged_question": b["n_dodged_question"],
            "address_rate": (
                b["n_addressed_question"] / b["n_questioned"]
                if b.get("n_questioned") else None
            ),
            "hallucination_examples": b["hallucination_examples_raw"][:10],
        }

    return {
        "n_total_judge_files": n_total,
        "n_with_verdict": n_ok,
        "errors": n_errors,
        "per_agent": summary_per_agent,
        "judge_strategy_score": _stats(judge_strategy_scores),
        "judge_synthesis_quality": _stats(judge_synthesis_scores),
        "discussion_convergence_score": _stats(convergence_scores),
        "_judge_strategy_scores": judge_strategy_scores,
        "_convergence_by_rounds": convergence_by_rounds,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_per_agent_rate(
    summary: dict, score_key: str, title: str, ylabel: str, out_path: Path,
    color: str | None = None,
) -> None:
    setup_plot_style()
    per = summary.get("per_agent", {})
    means = []
    stds = []
    for a in AGENT_NAMES:
        st = per.get(a, {}).get(score_key) or {}
        means.append(st.get("mean") if st.get("mean") is not None else 0.0)
        stds.append(st.get("stdev") if st.get("stdev") is not None else 0.0)
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x, means, yerr=stds, capsize=4,
           color=color or STYLE.primary, edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_role_adherence(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    per = summary.get("per_agent", {})
    rates = [per.get(a, {}).get("role_adherence_rate", 0.0) for a in AGENT_NAMES]
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x, rates, color=STYLE.success, edgecolor="black", alpha=0.85)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("Role adherence rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-agent role adherence (Hub-and-Spoke judge)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_strategy_histogram(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    scores = summary.get("_judge_strategy_scores", [])
    fig, ax = plt.subplots(figsize=(7, 4))
    if scores:
        ax.hist(scores, bins=20, range=(0, 1), color=STYLE.primary,
                edgecolor="black", alpha=0.85)
        mean = float(np.mean(scores))
        ax.axvline(mean, color=STYLE.error, linestyle="--", label=f"mean={mean:.2f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    ax.set_xlabel("judge_question_strategy_score")
    ax.set_ylabel("count")
    ax.set_title("Judge question strategy score distribution per image")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_convergence_by_rounds(by_rounds: dict, out_path: Path) -> None:
    setup_plot_style()
    round_keys = sorted(by_rounds.keys())
    if not round_keys:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Convergence score vs. discussion rounds")
        fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(len(round_keys)))
    data = [by_rounds[r] for r in round_keys]

    ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True,
               boxprops=dict(facecolor=STYLE.primary, alpha=0.5),
               medianprops=dict(color=STYLE.error, linewidth=2),
               whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
               flierprops=dict(marker="o", markersize=4, alpha=0.5))

    rng = np.random.default_rng(42)
    for i, scores in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(scores))
        ax.scatter([i + j for j in jitter], scores, alpha=0.45, s=18,
                   color=STYLE.neutral, zorder=3)
        ax.text(i, -0.1, f"n={len(scores)}", ha="center", fontsize=8, color="#555555")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{r} round{'s' if r != 1 else ''}" for r in round_keys])
    ax.set_xlabel("Number of discussion rounds")
    ax.set_ylabel("discussion_convergence_score")
    ax.set_ylim(-0.15, 1.15)
    ax.set_title("Discussion convergence score vs. number of hub rounds")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(out_dir: Path) -> dict:
    judge_dir = out_dir / "judge"
    if not judge_dir.is_dir():
        raise FileNotFoundError(f"No judge directory at {judge_dir}")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = _aggregate(judge_dir)

    _plot_role_adherence(summary, plots_dir / "judge_role_adherence.png")
    _plot_per_agent_rate(
        summary, "hallucination_score",
        "Per-agent hallucination score (0=clean, 1=severe)",
        "mean hallucination score",
        plots_dir / "judge_hallucination.png",
        color=STYLE.error,
    )
    _plot_per_agent_rate(
        summary, "question_relevance_score",
        "Per-agent question relevance score",
        "mean question relevance",
        plots_dir / "judge_question_relevance.png",
        color=STYLE.warning,
    )
    _plot_per_agent_rate(
        summary, "response_update_quality",
        "Per-agent response update quality",
        "mean response update quality",
        plots_dir / "judge_response_update.png",
        color=STYLE.neutral,
    )
    _plot_strategy_histogram(summary, plots_dir / "judge_strategy.png")
    _plot_convergence_by_rounds(
        summary.get("_convergence_by_rounds", {}),
        plots_dir / "convergence_by_rounds.png",
    )

    # Strip private fields before serialization
    serializable = {k: v for k, v in summary.items() if not k.startswith("_")}
    out_file = out_dir / "judge_summary.json"
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[judge_aggregate] wrote {out_file}")
    print(
        f"[judge_aggregate] verdicts={summary['n_with_verdict']}/"
        f"{summary['n_total_judge_files']}"
    )
    strat = summary.get("judge_strategy_score") or {}
    if strat.get("n"):
        print(f"[judge_aggregate] judge_strategy mean={strat['mean']:.3f} n={strat['n']}")
    conv = summary.get("discussion_convergence_score") or {}
    if conv.get("n"):
        print(f"[judge_aggregate] discussion_convergence  mean={conv['mean']:.3f}  median={conv.get('median', 0):.3f}  n={conv['n']}")
    for agent in AGENT_NAMES:
        a = serializable["per_agent"].get(agent, {})
        hall = a.get("hallucination_score") or {}
        ra = a.get("role_adherence_rate", 0.0)
        hall_mean = hall.get("mean")
        hall_str = f"{hall_mean:.2f}" if hall_mean is not None else "n/a"
        print(f"[judge_aggregate] {agent:11s} role={ra:.1%}  hall={hall_str}  n={a.get('n', 0)}")
    return serializable
