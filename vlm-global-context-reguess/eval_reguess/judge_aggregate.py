"""Aggregate per-image judge JSONs into a single summary.

Reads every <out>/judge/<image_id>.json produced by judge.py (one file per
image, all five agents inside each verdict as dict[agent_name, value]) and
collapses them into judge_summary.json.

Legacy per-agent files (<image_id>_<agent>.json from before the pro-image
refactor) are detected via the ``agent`` key in the payload and skipped.

Outputs:
  judge_summary.json
  plots/judge_role_adherence.png
  plots/judge_hallucination.png
  plots/judge_round2_improvement.png
  plots/judge_synthesis.png
  plots/judge_role_leakage.png
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
from eval_reguess.loader import AGENT_NAMES


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


def _empty_per_agent() -> dict[str, dict]:
    return {
        a: {
            "n": 0,
            "role_adherence_true": 0,
            "hallucination_scores": [],
            "visual_consistency_scores": [],
            "confidence_calibration_scores": [],
            "round2_improvement_scores": [],
            "hallucination_examples": [],
            "role_leakage_scores": [],
            "role_leakage_notes_examples": [],
        }
        for a in AGENT_NAMES
    }


def _aggregate(judge_dir: Path) -> dict:
    per_agent = _empty_per_agent()
    n_total = 0
    n_ok = 0
    n_errors: Counter = Counter()
    judge_synthesis_scores: list[float] = []
    round2_all_scores: list[float] = []

    for path in sorted(judge_dir.glob("*.json")):
        n_total += 1
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception:
            n_errors["read_error"] += 1
            continue

        # Skip legacy pro-agent files (pre-refactor: <id>_<agent>.json or <id>_judge.json).
        if "agent" in payload:
            n_errors["legacy_skipped"] += 1
            continue

        if "error" in payload:
            n_errors["llm_or_parse_error"] += 1
            continue

        verdict = payload.get("verdict") or {}
        if not verdict:
            n_errors["missing_verdict"] += 1
            continue

        n_ok += 1
        image_id = payload.get("image_id") or path.stem

        # Image-level: judge synthesis quality (one per image)
        jsq = verdict.get("judge_synthesis_quality")
        if isinstance(jsq, (int, float)):
            judge_synthesis_scores.append(float(jsq))

        # Per-agent: pull from dict-shaped verdict fields
        for agent in AGENT_NAMES:
            bucket = per_agent[agent]
            bucket["n"] += 1

            ra_map = verdict.get("role_adherence") or {}
            if ra_map.get(agent) is True:
                bucket["role_adherence_true"] += 1

            for score_key, list_key in [
                ("hallucination_score", "hallucination_scores"),
                ("visual_consistency_score", "visual_consistency_scores"),
                ("confidence_calibration_score", "confidence_calibration_scores"),
                ("round2_improvement", "round2_improvement_scores"),
            ]:
                v = (verdict.get(score_key) or {}).get(agent)
                if isinstance(v, (int, float)):
                    bucket[list_key].append(float(v))
                    if score_key == "round2_improvement":
                        round2_all_scores.append(float(v))

            # Hallucination examples (cap 10 per agent)
            hall_score = (verdict.get("hallucination_score") or {}).get(agent)
            examples = (verdict.get("hallucination_examples") or {}).get(agent) or []
            if isinstance(examples, list) and examples and len(bucket["hallucination_examples"]) < 10:
                for s in examples[:3]:
                    if isinstance(s, str) and s.strip():
                        bucket["hallucination_examples"].append({
                            "image_id": image_id,
                            "example": s.strip(),
                            "score": float(hall_score) if isinstance(hall_score, (int, float)) else None,
                        })
                        if len(bucket["hallucination_examples"]) >= 10:
                            break

            # Role leakage
            leakage = (verdict.get("role_leakage_score") or {}).get(agent)
            if isinstance(leakage, (int, float)):
                bucket["role_leakage_scores"].append(float(leakage))
            leakage_note = (verdict.get("role_leakage_notes") or {}).get(agent) or ""
            if (isinstance(leakage, (int, float)) and leakage > 0.0
                    and isinstance(leakage_note, str) and leakage_note.strip()
                    and leakage_note.strip().lower() not in ("none", "n/a - evaluating judge, not a specialist agent")
                    and len(bucket["role_leakage_notes_examples"]) < 10):
                bucket["role_leakage_notes_examples"].append({
                    "image_id": image_id,
                    "note": leakage_note.strip(),
                    "score": float(leakage),
                })

    # Build per-agent summary
    summary_per_agent: dict[str, dict] = {}
    for agent, b in per_agent.items():
        n = b["n"]
        summary_per_agent[agent] = {
            "n": n,
            "role_adherence_rate": (b["role_adherence_true"] / n) if n else 0.0,
            "hallucination_score": _stats(b["hallucination_scores"]),
            "visual_consistency_score": _stats(b["visual_consistency_scores"]),
            "confidence_calibration_score": _stats(b["confidence_calibration_scores"]),
            "round2_improvement": _stats(b["round2_improvement_scores"]),
            "hallucination_examples": b["hallucination_examples"],
            "role_leakage_score": _stats(b["role_leakage_scores"]),
            "role_leakage_notes_examples": b["role_leakage_notes_examples"],
        }

    # Overall round2 improvement (across all agents)
    mean_r2 = float(np.mean(round2_all_scores)) if round2_all_scores else None

    return {
        "n_total": n_total,
        "n_with_verdict": n_ok,
        "errors": dict(n_errors),
        "mean_round2_improvement": mean_r2,
        "judge_synthesis_quality": _stats(judge_synthesis_scores),
        "per_agent": summary_per_agent,
    }


# ── Plots ──────────────────────────────────────────────────────────────────────


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


def _plot_hallucination(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    means = []
    for a in AGENT_NAMES:
        st = summary["per_agent"][a]["hallucination_score"]
        means.append(st.get("mean") if st.get("mean") is not None else 0.0)
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, means, color=STYLE.error, edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("mean hallucination score (0=clean, 1=severe)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-agent hallucination score (lower is better)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_round2_improvement(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    means = []
    for a in AGENT_NAMES:
        st = summary["per_agent"][a]["round2_improvement"]
        means.append(st.get("mean") if st.get("mean") is not None else 0.0)
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, means, color=STYLE.success, edgecolor="black", alpha=0.85)
    ax.axhline(0.5, color=STYLE.warning, linestyle="--", linewidth=1.5,
               label="0.5 = acknowledged but didn't change")
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("mean round2_improvement (0-1)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-agent Round 2 improvement (1=genuine synthesis)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_synthesis_histogram(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    st = summary.get("judge_synthesis_quality") or {}
    n = st.get("n") or 0
    fig, ax = plt.subplots(figsize=(7, 4))
    if n == 0:
        ax.text(0.5, 0.5, "no judge_synthesis_quality scores recorded",
                ha="center", va="center")
        ax.set_axis_off()
    else:
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
    ax.set_title("Judge synthesis quality distribution")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_role_leakage(summary: dict, out_path: Path) -> None:
    setup_plot_style()
    means = []
    for a in AGENT_NAMES:
        st = (summary.get("per_agent") or {}).get(a, {}).get("role_leakage_score") or {}
        means.append(st.get("mean") or 0.0)
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, means, color=STYLE.error, edgecolor="black", alpha=0.85)
    for i, v in enumerate(means):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("mean role_leakage_score (0=clean, 1=heavy leakage)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-agent role leakage in Round 2 (lower is better)")
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
    _plot_hallucination(summary, plots_dir / "judge_hallucination.png")
    _plot_round2_improvement(summary, plots_dir / "judge_round2_improvement.png")
    _plot_synthesis_histogram(summary, plots_dir / "judge_synthesis.png")
    _plot_role_leakage(summary, plots_dir / "judge_role_leakage.png")

    out_file = out_dir / "judge_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[judge_aggregate] wrote {out_file}")
    print(
        f"[judge_aggregate] verdicts={summary['n_with_verdict']}/{summary['n_total']}  "
        f"mean_r2_improvement={summary.get('mean_round2_improvement')}"
    )
    jsq = summary.get("judge_synthesis_quality") or {}
    if jsq.get("n"):
        print(
            f"[judge_aggregate] judge_synthesis mean={jsq['mean']:.3f} "
            f"median={jsq['median']:.3f} n={jsq['n']}"
        )
    for agent in AGENT_NAMES:
        a = summary["per_agent"][agent]
        r2 = a["round2_improvement"]
        hall = a["hallucination_score"]
        r2_mean = r2.get("mean") if r2.get("mean") is not None else float("nan")
        hall_mean = hall.get("mean") if hall.get("mean") is not None else float("nan")
        leakage_st = a.get("role_leakage_score") or {}
        leakage_mean = leakage_st.get("mean") if leakage_st.get("mean") is not None else float("nan")
        print(
            f"[judge_aggregate] {agent:11s} role={a['role_adherence_rate']:.1%}  "
            f"r2_imp={r2_mean:.2f}  hall={hall_mean:.2f}  leakage={leakage_mean:.2f}  n={a['n']}"
        )
    return summary
