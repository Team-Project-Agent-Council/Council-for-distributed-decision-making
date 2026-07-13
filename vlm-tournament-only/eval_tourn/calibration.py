"""Confidence calibration: per-agent Brier, ECE, and per-label hit-rate.

Each specialist agent annotates **every candidate** in its list (not just the
top-1) with one of: ``high`` / ``medium`` / ``low`` / ``speculative``.

These per-candidate labels are then fed to the tournament judge as evidence,
so the question that matters is:

    "When this agent assigns label X to a country, in what fraction of those
     (image, country, label) tuples was that country the ground truth?"

That is ``P(truth | label, agent)``, a per-label hit-rate. A well-calibrated
agent has monotonically falling hit-rates: high > medium > low > speculative.

Brier and ECE are kept *for the top-1 only* (one (p, outcome) pair per image
per agent), they remain useful as a single-number summary of how well the
agent's most confident pick lines up with its self-assessed confidence.

Module is purely deterministic, no LLM call.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_tourn._style import STYLE, setup_plot_style
from eval_tourn.loader import (
    AGENT_NAMES,
    RunRecord,
    countries_match,
    load_run,
    top1_country,
)


_CONFIDENCE_TO_P: dict[str, float] = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.3,
    "speculative": 0.1,
}
_LABELS = ("high", "medium", "low", "speculative")


def _confidence_p(label: str | None) -> float | None:
    if not label:
        return None
    return _CONFIDENCE_TO_P.get(label.strip().lower())


def _norm_label(label: str | None) -> str | None:
    if not label:
        return None
    s = label.strip().lower()
    return s if s in _LABELS else None


# --- Top-1 calibration (Brier / ECE) -----------------------------------------

def _agent_top1_rows(records: list[RunRecord], agent: str) -> list[tuple[float, bool]]:
    """Per-agent (p, is_correct) tuples for the top-1 of each image."""
    out: list[tuple[float, bool]] = []
    for r in records:
        a = (r.assessments or {}).get(agent)
        if not a:
            continue
        cands = a.get("candidates") or []
        if not cands:
            continue
        p = _confidence_p(cands[0].get("confidence"))
        if p is None:
            continue
        top1 = top1_country(a)
        is_correct = bool(top1 and countries_match(top1, r.truth_country_code))
        out.append((p, is_correct))
    return out


def _brier(rows: list[tuple[float, bool]]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([(p - (1.0 if c else 0.0)) ** 2 for p, c in rows]))


def _ece(rows: list[tuple[float, bool]], n_bins: int = 5) -> float:
    if not rows:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(rows)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            members = [(p, c) for p, c in rows if lo <= p <= hi]
        else:
            members = [(p, c) for p, c in rows if lo <= p < hi]
        if not members:
            continue
        mean_p = float(np.mean([p for p, _ in members]))
        acc = float(np.mean([1.0 if c else 0.0 for _, c in members]))
        ece += (len(members) / n) * abs(mean_p - acc)
    return float(ece)


# --- Per-label hit-rate ------------------------------------------------------

def _agent_label_hits(records: list[RunRecord], agent: str) -> dict[str, dict]:
    """Per-label hit-rate over **all candidates** the agent annotates.

    For each (image, candidate_country, label) the agent emits, record whether
    that country equals the ground truth. Returns ``{label: {n, hits, rate}}``.
    """
    out: dict[str, dict] = {lab: {"n": 0, "hits": 0} for lab in _LABELS}
    for r in records:
        a = (r.assessments or {}).get(agent)
        if not a:
            continue
        for cand in (a.get("candidates") or []):
            lab = _norm_label(cand.get("confidence"))
            if not lab:
                continue
            country = cand.get("country")
            if not country:
                continue
            is_truth = countries_match(country, r.truth_country_code)
            out[lab]["n"] += 1
            if is_truth:
                out[lab]["hits"] += 1
    for lab in _LABELS:
        n = out[lab]["n"]
        out[lab]["rate"] = (out[lab]["hits"] / n) if n else None
    return out


# --- Aggregation -------------------------------------------------------------

def compute(records: list[RunRecord]) -> dict:
    per_agent: dict[str, dict] = {}
    all_top1: list[tuple[float, bool]] = []
    pooled_label_hits = {lab: {"n": 0, "hits": 0} for lab in _LABELS}

    for agent in AGENT_NAMES:
        top1_rows = _agent_top1_rows(records, agent)
        all_top1.extend(top1_rows)
        label_hits = _agent_label_hits(records, agent)
        for lab in _LABELS:
            pooled_label_hits[lab]["n"] += label_hits[lab]["n"]
            pooled_label_hits[lab]["hits"] += label_hits[lab]["hits"]
        per_agent[agent] = {
            "n_top1": len(top1_rows),
            "brier": _brier(top1_rows),
            "ece": _ece(top1_rows),
            "label_hit_rate": label_hits,
        }

    for lab in _LABELS:
        n = pooled_label_hits[lab]["n"]
        pooled_label_hits[lab]["rate"] = (pooled_label_hits[lab]["hits"] / n) if n else None

    return {
        "confidence_to_p": _CONFIDENCE_TO_P,
        "labels": list(_LABELS),
        "per_agent": per_agent,
        "average": {
            "n_top1": len(all_top1),
            "brier": _brier(all_top1),
            "ece": _ece(all_top1),
            "label_hit_rate": pooled_label_hits,
        },
    }


# --- Plot --------------------------------------------------------------------

def plot_label_hit_rate(metrics: dict, out_path: Path) -> None:
    """Grouped bar chart: per-label hit-rate, one cluster per agent."""
    setup_plot_style()
    agents = list(AGENT_NAMES) + ["average"]
    labels = list(_LABELS)
    n_agents = len(agents)
    n_labels = len(labels)
    label_colors = {
        "high": STYLE.success,
        "medium": STYLE.primary,
        "low": STYLE.warning,
        "speculative": STYLE.neutral,
    }

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bar_w = 0.8 / n_labels
    x = np.arange(n_agents)

    for j, lab in enumerate(labels):
        rates = []
        ns = []
        for agent in agents:
            blk = (metrics["average"] if agent == "average"
                   else metrics["per_agent"][agent])
            lab_blk = (blk.get("label_hit_rate") or {}).get(lab) or {}
            rate = lab_blk.get("rate")
            rates.append(rate if rate is not None else 0.0)
            ns.append(lab_blk.get("n", 0))
        offset = (j - (n_labels - 1) / 2) * bar_w
        bars = ax.bar(x + offset, rates, bar_w, label=lab,
                      color=label_colors[lab], edgecolor="black", linewidth=0.5)
        for bar, rate, n in zip(bars, rates, ns):
            if n == 0:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{rate:.0%}\nn={n}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("P(country = truth | label)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Per-label hit-rate, how often does a labeled candidate match the ground truth?"
    )
    ax.legend(title="confidence label", loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --- Entry -------------------------------------------------------------------

def run(results_dir: Path, gt_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    records = load_run(results_dir, gt_csv)
    metrics = compute(records)

    plot_label_hit_rate(metrics, plots_dir / "calibration_label_hit_rate.png")

    out_file = out_dir / "calibration_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    avg = metrics["average"]
    print(
        f"[calibration] wrote {out_file}  avg Brier={avg['brier']:.3f}  "
        f"avg ECE={avg['ece']:.3f}  n_top1={avg['n_top1']}"
    )
    for a in AGENT_NAMES:
        m = metrics["per_agent"][a]
        lab_str = "  ".join(
            f"{lab}={(m['label_hit_rate'][lab]['rate'] or 0):.0%}/n={m['label_hit_rate'][lab]['n']}"
            for lab in _LABELS
        )
        print(f"[calibration] {a:11s} Brier={m['brier']:.3f}  ECE={m['ece']:.3f}  "
              f"n_top1={m['n_top1']}  |  {lab_str}")
    return metrics
