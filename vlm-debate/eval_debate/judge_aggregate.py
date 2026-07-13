"""Aggregate per-image judge JSONs into judge_summary.json.

Reads every <out>/judge/<image_id>.json produced by judge.py (one file per
image, all five agents inside each verdict as dict[agent_name, value]) and
collapses them into judge_summary.json.

Legacy per-agent files (<image_id>_<agent>.json from before the pro-image
refactor) are detected via the ``agent_name`` key in the payload and skipped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_debate._style import STYLE, setup_plot_style
from eval_debate.loader import AGENT_NAMES


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "stdev": None}
    n = len(vals)
    mean = sum(vals) / n
    median = float(np.median(vals))
    stdev = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
    return {"n": n, "mean": round(mean, 4), "median": round(median, 4),
            "stdev": round(stdev, 4)}


def _empty_bucket() -> dict:
    return {
        "n": 0, "n_debated": 0, "n_role_ok": 0,
        "hallucination_scores": [], "visual_scores": [], "calibration_scores": [],
        "argument_quality_scores": [], "revision_justification_scores": [],
        "debate_contribution_scores": [],
        "hallucination_examples": [],
    }


# Scores that are only meaningful when the agent actually debated.
_DEBATE_ONLY_SCORE_KEYS = {
    "argument_quality_score",
    "revision_justification_score",
    "debate_contribution_score",
}


def _aggregate(judge_dir: Path) -> dict:
    per_agent: dict[str, dict] = {a: _empty_bucket() for a in AGENT_NAMES}

    moderator_scores: list[float] = []
    synthesis_scores: list[float] = []
    synthesis_by_debate: dict[bool, list[float]] = {True: [], False: []}

    n_total = 0
    n_with_verdict = 0
    n_errors: dict[str, int] = {}

    for jf in sorted(judge_dir.glob("*.json")):
        try:
            payload = json.loads(jf.read_text())
        except Exception:
            n_errors["read_error"] = n_errors.get("read_error", 0) + 1
            continue
        n_total += 1

        # Skip legacy pro-agent files (pre-refactor: <id>_<agent>.json or <id>_judge.json).
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

        n_with_verdict += 1
        image_id = payload.get("image_id", jf.stem)
        debate_happened = bool(payload.get("debate_happened", False))
        agent_debated_map = payload.get("agent_debated") or {}

        # Image-level scalars (one per image)
        mod = verdict.get("moderator_pairing_quality_score")
        syn = verdict.get("judge_synthesis_quality")
        if isinstance(mod, (int, float)):
            moderator_scores.append(float(mod))
        if isinstance(syn, (int, float)):
            synthesis_scores.append(float(syn))
            synthesis_by_debate[debate_happened].append(float(syn))

        # Per-agent (pull from dict-shaped verdict fields)
        for agent in AGENT_NAMES:
            bucket = per_agent[agent]
            bucket["n"] += 1
            this_agent_debated = bool(agent_debated_map.get(agent, False))
            if this_agent_debated:
                bucket["n_debated"] += 1

            ra_map = verdict.get("role_adherence") or {}
            if ra_map.get(agent) is True:
                bucket["n_role_ok"] += 1

            for score_key, list_key in [
                ("hallucination_score", "hallucination_scores"),
                ("visual_consistency_score", "visual_scores"),
                ("confidence_calibration_score", "calibration_scores"),
                ("argument_quality_score", "argument_quality_scores"),
                ("revision_justification_score", "revision_justification_scores"),
                ("debate_contribution_score", "debate_contribution_scores"),
            ]:
                if score_key in _DEBATE_ONLY_SCORE_KEYS and not this_agent_debated:
                    continue
                v = (verdict.get(score_key) or {}).get(agent)
                if isinstance(v, (int, float)):
                    bucket[list_key].append(float(v))

            hall_score = (verdict.get("hallucination_score") or {}).get(agent)
            examples = (verdict.get("hallucination_examples") or {}).get(agent) or []
            if isinstance(examples, list) and examples:
                for ex in examples[:3]:
                    if isinstance(ex, str) and ex.strip() and len(bucket["hallucination_examples"]) < 10:
                        bucket["hallucination_examples"].append({
                            "image_id": image_id,
                            "score": float(hall_score) if isinstance(hall_score, (int, float)) else None,
                            "example": ex.strip(),
                        })

    summary_per_agent = {}
    for agent, b in per_agent.items():
        n = b["n"]
        summary_per_agent[agent] = {
            "n": n,
            "n_debated": b["n_debated"],
            "role_adherence_rate": (b["n_role_ok"] / n) if n else 0.0,
            "hallucination_score": _stats(b["hallucination_scores"]),
            "visual_consistency_score": _stats(b["visual_scores"]),
            "confidence_calibration_score": _stats(b["calibration_scores"]),
            "argument_quality_score": _stats(b["argument_quality_scores"]),
            "revision_justification_score": _stats(b["revision_justification_scores"]),
            "debate_contribution_score": _stats(b["debate_contribution_scores"]),
            "hallucination_examples": b["hallucination_examples"],
        }

    return {
        "n_total_judge_files": n_total,
        "n_with_verdict": n_with_verdict,
        "errors": n_errors,
        "per_agent": summary_per_agent,
        "moderator_pairing_quality_score": _stats(moderator_scores),
        "judge_synthesis_quality": _stats(synthesis_scores),
        "_synthesis_by_debate": synthesis_by_debate,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_bar_per_agent(summary: dict, score_key: str, title: str,
                        color: str, out_path: Path) -> None:
    setup_plot_style()
    means = []
    for a in AGENT_NAMES:
        st = (summary["per_agent"].get(a) or {}).get(score_key) or {}
        means.append(st.get("mean") or 0.0)
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, means, color=color, alpha=0.85, edgecolor="black")
    for i, v in enumerate(means):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("mean score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_synthesis_by_debate(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    by_debate = summary.get("_synthesis_by_debate", {})
    data_no = by_debate.get(False, [])
    data_yes = by_debate.get(True, [])

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = [0, 1]
    data = [data_no, data_yes]
    labels = [f"No debate\n(n={len(data_no)})", f"Debate occurred\n(n={len(data_yes)})"]

    if any(data):
        ax.boxplot(
            data, positions=positions, widths=0.4, patch_artist=True,
            boxprops=dict(facecolor=STYLE.primary, alpha=0.5),
            medianprops=dict(color=STYLE.error, linewidth=2),
        )
        rng = np.random.default_rng(42)
        for i, scores in enumerate(data):
            jitter = rng.uniform(-0.1, 0.1, size=len(scores))
            ax.scatter([i + j for j in jitter], scores,
                       alpha=0.4, s=15, color=STYLE.neutral, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("judge_synthesis_quality")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("Judge synthesis quality: debate vs. no-debate images")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    judge_dir = out_dir / "judge"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    if not judge_dir.exists():
        print(f"[judge_aggregate] no judge/ dir found at {judge_dir}")
        return {}

    summary = _aggregate(judge_dir)

    public = {k: v for k, v in summary.items() if not k.startswith("_")}
    with open(out_dir / "judge_summary.json", "w") as f:
        json.dump(public, f, indent=2)
    print(f"[judge_aggregate] wrote judge_summary.json  "
          f"n_files={summary['n_total_judge_files']}  "
          f"n_with_verdict={summary['n_with_verdict']}  "
          f"errors={summary['errors']}")

    _plot_bar_per_agent(summary, "hallucination_score",
                        "Per-agent hallucination score (0=clean, lower=better)",
                        STYLE.error, plots_dir / "judge_hallucination.png")
    _plot_bar_per_agent(summary, "argument_quality_score",
                        "Per-agent argument quality in debate (higher=better)",
                        STYLE.primary, plots_dir / "judge_argument_quality.png")
    _plot_bar_per_agent(summary, "revision_justification_score",
                        "Per-agent revision justification quality (higher=better)",
                        STYLE.warning, plots_dir / "judge_revision_justification.png")
    _plot_bar_per_agent(summary, "debate_contribution_score",
                        "Per-agent debate contribution, new evidence surfaced (higher=better)",
                        STYLE.success, plots_dir / "judge_debate_contribution.png")

    setup_plot_style()
    rates = [
        (summary["per_agent"].get(a) or {}).get("role_adherence_rate", 0.0)
        for a in AGENT_NAMES
    ]
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, rates, color=STYLE.success, alpha=0.85)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylim(0, 1.1); ax.set_ylabel("role adherence rate")
    ax.set_title("Per-agent role adherence")
    fig.tight_layout(); fig.savefig(plots_dir / "judge_role_adherence.png"); plt.close(fig)

    _plot_synthesis_by_debate(summary, plots_dir / "judge_synthesis_by_debate.png")

    for agent in AGENT_NAMES:
        a = (summary["per_agent"] or {}).get(agent, {})
        hall = (a.get("hallucination_score") or {}).get("mean") or float("nan")
        arg = (a.get("argument_quality_score") or {}).get("mean") or float("nan")
        rev = (a.get("revision_justification_score") or {}).get("mean") or float("nan")
        print(f"[judge_aggregate] {agent:11s}  "
              f"role={a.get('role_adherence_rate', 0):.1%}  "
              f"hall={hall:.2f}  arg_q={arg:.2f}  rev_j={rev:.2f}  "
              f"n={a.get('n', 0)}  n_debated={a.get('n_debated', 0)}")

    mod = public.get("moderator_pairing_quality_score") or {}
    syn = public.get("judge_synthesis_quality") or {}
    if mod.get("n"):
        print(f"[judge_aggregate] moderator_pairing  "
              f"mean={mod['mean']:.3f}  median={mod['median']:.3f}  n={mod['n']}")
    if syn.get("n"):
        print(f"[judge_aggregate] judge_synthesis     "
              f"mean={syn['mean']:.3f}  median={syn['median']:.3f}  n={syn['n']}")

    return public
