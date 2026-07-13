"""Pipeline funnel metrics for VLM Council, Progressive Narrowing.

Computes deterministically (no LLM needed) from existing result.json files:

  1. Pipeline funnel (S0..S4): truth-survival rate at each stage with Wilson
     95% confidence intervals and auto-identified bottleneck.

     S0, Truth in any agent's initial top-K candidates
     S1, Confirmed region matches truth's region
     S2, Truth in country-round candidates
          (Path A: identical to S1; Path B: truth in constrained agent top-K)
     S3, Truth in country hypothesis pool (presented to evaluate step)
     S4, Final prediction matches truth (= country accuracy)

  2. Oracle ceilings: counterfactual accuracy if region / hypothesis pool
     were perfect, plus majority-vote baseline.

  3. Agreement-vs-accuracy: final accuracy by how many of the 5 agents
     agreed on the same top-1 country.

  4. Path A vs. Path B comparison.

  5. Severity breakdown: near-miss / same-region-wrong / wrong-region.

Writes ``funnel_metrics.json`` and plots under ``plots/``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval._style import STYLE, setup_plot_style
from eval.loader import (
    AGENT_NAMES,
    RunRecord,
    countries_match,
    load_run,
    top1_country,
    topk_countries,
)


# Wilson CI

def wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate_block(k: int, n: int) -> dict:
    if n == 0:
        return {"n": 0, "correct": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    lo, hi = wilson_ci(k, n)
    return {"n": n, "correct": k, "rate": k / n, "ci_low": lo, "ci_high": hi}


# Region helpers

def _truth_region(record: RunRecord) -> str | None:
    """Return the truth country's world region, lowercased."""
    try:
        from vlm_council.regions import country_to_region
        r = country_to_region(record.truth_country_name) or country_to_region(
            record.truth_country_code
        )
        return r.lower() if r else None
    except Exception:
        return None


def _regions_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return a.strip().lower() == b.strip().lower()


# Stage classifiers

def _all_topk(round_dict: dict | None) -> list[str]:
    if not round_dict:
        return []
    out: list[str] = []
    for agent in AGENT_NAMES:
        out.extend(topk_countries(round_dict.get(agent), k=99))
    return out


def _truth_in(candidates: list[str], truth_code: str) -> bool:
    return any(countries_match(c, truth_code) for c in candidates if c)


def s0_initial_recall(r: RunRecord) -> bool:
    """Truth appears in any agent's initial candidates."""
    return _truth_in(_all_topk(r.assessments), r.truth_country_code)


def s1_region_correct(r: RunRecord) -> bool:
    """Confirmed/proposed region contains truth's region."""
    truth_region = _truth_region(r)
    if truth_region is None:
        return False
    if _regions_match(r.confirmed_region, truth_region):
        return True
    # Also accept if truth region is in proposed_regions (not yet decided)
    for p in r.proposed_regions:
        if _regions_match(p, truth_region):
            return True
    return False


def s2_country_round_recall(r: RunRecord) -> bool:
    """Truth in country-round candidates.

    Path A: region was correct by consensus (= S1), no second round exists.
    Path B: truth must appear in the constrained country assessments.
    """
    if r.path == "B":
        return _truth_in(_all_topk(r.country_assessments), r.truth_country_code)
    # Path A: no separate country-round, so survival == S1
    return s1_region_correct(r)


def s3_in_hypothesis_pool(r: RunRecord) -> bool:
    """Truth is in the country hypothesis pool (active_hypotheses)."""
    return r.truth_in_hypothesis_pool


def s4_final_correct(r: RunRecord) -> bool:
    return r.is_correct


_STAGE_FNS = [s0_initial_recall, s1_region_correct, s2_country_round_recall,
              s3_in_hypothesis_pool, s4_final_correct]

_STAGE_META = [
    ("S0", "Truth in any agent's initial top-K"),
    ("S1", "Confirmed/proposed region matches truth's region"),
    ("S2", "Truth in country-round top-K (Path B) / region OK (Path A)"),
    ("S3", "Truth in country hypothesis pool"),
    ("S4", "Final prediction matches truth"),
]


# Funnel

def compute_funnel(records: list[RunRecord]) -> dict:
    n = len(records)
    survived = [[fn(r) for r in records] for fn in _STAGE_FNS]
    counts = [sum(s) for s in survived]

    # Cumulative: truth survived all stages 0..i.
    # S4 (final correct) is defined as is_correct regardless of upstream
    # survival to preserve exact match with geo_metrics.country_accuracy.
    cumulative: list[int] = []
    accum = [True] * n
    for i, s in enumerate(survived):
        if i < len(survived) - 1:
            accum = [a and b for a, b in zip(accum, s)]
            cumulative.append(sum(accum))
        else:
            cumulative.append(counts[i])  # S4 = is_correct directly

    stages = []
    for i, ((code, desc), c, c_cum) in enumerate(zip(_STAGE_META, counts, cumulative)):
        conditional_n = cumulative[i - 1] if i > 0 else n
        conditional_k = min(c_cum, conditional_n)
        stages.append({
            "code": code,
            "description": desc,
            "stage_only_rate": _rate_block(c, n),
            "cumulative_survival": _rate_block(c_cum, n),
            "conditional_on_prev": _rate_block(conditional_k, conditional_n),
        })

    # Bottleneck: stage with lowest conditional_on_prev (i ≥ 1)
    bottleneck = None
    worst = min(
        (s for s in stages[1:] if s["conditional_on_prev"]["n"] > 0),
        key=lambda s: s["conditional_on_prev"]["rate"],
        default=None,
    )
    if worst:
        bottleneck = {
            "stage_code": worst["code"],
            "description": worst["description"],
            "conditional_rate": worst["conditional_on_prev"]["rate"],
            "ci_low": worst["conditional_on_prev"]["ci_low"],
            "ci_high": worst["conditional_on_prev"]["ci_high"],
        }

    return {
        "n_total": n,
        "stages": stages,
        "bottleneck": bottleneck,
        "monotone": all(
            cumulative[i] >= cumulative[i + 1] for i in range(len(cumulative) - 2)
        ),
    }


# Oracle ceilings

def compute_oracle_ceilings(records: list[RunRecord]) -> dict:
    n = len(records)
    if n == 0:
        return {"n_total": 0}

    actual = sum(1 for r in records if s4_final_correct(r))

    # Oracle region: force correct region. Downstream: truth must have appeared
    # in agents' candidates (S0 or S2 Path B), and must end up in hypothesis pool.
    # Approximation: correct iff truth was in any initial or country-round top-K.
    def oracle_region_one(r: RunRecord) -> bool:
        return s0_initial_recall(r) or (
            r.path == "B" and _truth_in(_all_topk(r.country_assessments), r.truth_country_code)
        )

    oracle_region = sum(1 for r in records if oracle_region_one(r))

    # Oracle hypothesis pool: force truth into hypothesis pool.
    # Correct iff truth was named by any agent at any point.
    def oracle_pool_one(r: RunRecord) -> bool:
        return (
            s3_in_hypothesis_pool(r)
            or s0_initial_recall(r)
            or (r.path == "B" and _truth_in(_all_topk(r.country_assessments), r.truth_country_code))
        )

    oracle_pool = sum(1 for r in records if oracle_pool_one(r))

    # Oracle decision: truth is in pool AND the judge picks it (perfect judge).
    # = truth in hypothesis pool (S3).
    oracle_decision = sum(1 for r in records if s3_in_hypothesis_pool(r))

    # Majority-vote baseline: most common top-1 across 5 initial agents.
    def majority_vote_one(r: RunRecord) -> bool:
        votes: Counter = Counter()
        for agent in AGENT_NAMES:
            t = top1_country(r.assessments.get(agent))
            if t:
                votes[t] += 1
        if not votes:
            return False
        winner, _ = votes.most_common(1)[0]
        return countries_match(winner, r.truth_country_code)

    majority = sum(1 for r in records if majority_vote_one(r))

    return {
        "n_total": n,
        "actual": _rate_block(actual, n),
        "majority_vote_baseline": _rate_block(majority, n),
        "oracle_region": _rate_block(oracle_region, n),
        "oracle_pool": _rate_block(oracle_pool, n),
        "oracle_decision": _rate_block(oracle_decision, n),
    }


# Agreement vs accuracy

def agreement_curve(records: list[RunRecord]) -> dict:
    bins: dict[int, list[bool]] = {i: [] for i in range(1, 6)}
    for r in records:
        votes: Counter = Counter()
        for agent in AGENT_NAMES:
            t = top1_country(r.assessments.get(agent))
            if t:
                votes[t] += 1
        if not votes:
            continue
        modal_count = votes.most_common(1)[0][1]
        bins[modal_count].append(r.is_correct)

    return {
        "levels": [
            {"agreement": f"{level}/5", **_rate_block(sum(bins[level]), len(bins[level]))}
            for level in (5, 4, 3, 2, 1)
        ]
    }


# Path A vs B

def path_comparison(records: list[RunRecord]) -> dict:
    def stats(subset: list[RunRecord]) -> dict:
        n = len(subset)
        if n == 0:
            return {"n": 0}
        correct = sum(1 for r in subset if r.is_correct)
        haversines = [r.haversine_km for r in subset if r.haversine_km is not None]
        in_pool = sum(1 for r in subset if s3_in_hypothesis_pool(r))
        seconds = [r.total_seconds for r in subset if r.total_seconds]
        return {
            **_rate_block(correct, n),
            "median_haversine_km": float(np.median(haversines)) if haversines else None,
            "truth_in_pool_rate": in_pool / n if n else 0.0,
            "truth_in_pool_n": in_pool,
            "mean_total_seconds": float(np.mean(seconds)) if seconds else None,
        }

    return {
        "path_a": stats([r for r in records if r.path == "A"]),
        "path_b": stats([r for r in records if r.path == "B"]),
    }


# Severity

def severity_breakdown(records: list[RunRecord]) -> dict:
    near_miss, same_region_wrong, wrong_region_wrong = [], [], []
    for r in records:
        if r.is_correct:
            continue
        truth_r = _truth_region(r)
        pred_r_raw = None
        try:
            from vlm_council.regions import country_to_region
            pred_r_raw = country_to_region(r.pred_country)
            pred_r = pred_r_raw.lower() if pred_r_raw else None
        except Exception:
            pred_r = None
        hav = r.haversine_km if r.haversine_km is not None else float("inf")
        entry = {
            "image_id": r.image_id,
            "truth": r.truth_country_name,
            "pred": r.pred_country,
            "haversine_km": round(hav) if hav != float("inf") else None,
        }
        if hav < 500:
            near_miss.append(entry)
        if truth_r and pred_r and truth_r == pred_r:
            same_region_wrong.append(entry)
        elif truth_r and pred_r and truth_r != pred_r:
            wrong_region_wrong.append(entry)

    n_total = len(records)
    n_wrong = sum(1 for r in records if not r.is_correct)
    return {
        "n_total": n_total,
        "n_wrong": n_wrong,
        "near_miss_count": len(near_miss),
        "same_region_wrong_count": len(same_region_wrong),
        "wrong_region_count": len(wrong_region_wrong),
        "near_miss_examples": near_miss[:15],
        "same_region_wrong_examples": same_region_wrong[:15],
        "wrong_region_examples": wrong_region_wrong[:15],
    }


# Plots

def plot_funnel(funnel: dict, out_path: Path) -> None:
    setup_plot_style()
    stages = funnel.get("stages", [])
    if not stages:
        return
    labels = [f"{s['code']}: {s['description'][:50]}" for s in stages]
    rates = [s["cumulative_survival"]["rate"] for s in stages]
    los = [s["cumulative_survival"]["ci_low"] for s in stages]
    his = [s["cumulative_survival"]["ci_high"] for s in stages]

    fig, ax = plt.subplots(figsize=(10, 4))
    y = np.arange(len(labels))
    err_low = [max(0.0, r - lo) for r, lo in zip(rates, los)]
    err_high = [max(0.0, hi - r) for r, hi in zip(rates, his)]
    ax.barh(y, rates, color=STYLE.primary, alpha=0.85)
    ax.errorbar(rates, y,
                xerr=[err_low, err_high],
                fmt="none", ecolor="black", capsize=3)
    for i, (r, s) in enumerate(zip(rates, stages)):
        n = s["cumulative_survival"]["n"]
        k = s["cumulative_survival"]["correct"]
        ax.text(min(r + 0.02, 0.97), i, f"{k}/{n} = {r:.1%}",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("cumulative truth-survival (Wilson 95% CI)")
    ax.set_title("Pipeline funnel, where truth gets lost")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_oracle_bars(ceilings: dict, out_path: Path) -> None:
    setup_plot_style()
    keys = ["actual", "majority_vote_baseline", "oracle_region",
            "oracle_pool", "oracle_decision"]
    labels = ["Actual", "Majority\nbaseline", "Oracle\nregion",
              "Oracle\npool", "Oracle\ndecision"]
    rates = [(ceilings.get(k) or {}).get("rate", 0.0) for k in keys]
    los = [(ceilings.get(k) or {}).get("ci_low", 0.0) for k in keys]
    his = [(ceilings.get(k) or {}).get("ci_high", 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    colors = [STYLE.primary, STYLE.neutral, STYLE.success, STYLE.warning, STYLE.error]
    err_low = [max(0.0, r - lo) for r, lo in zip(rates, los)]
    err_high = [max(0.0, hi - r) for r, hi in zip(rates, his)]
    ax.bar(x, rates, color=colors, alpha=0.85)
    ax.errorbar(x, rates,
                yerr=[err_low, err_high],
                fmt="none", ecolor="black", capsize=3)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.015, f"{r:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("end-to-end accuracy")
    ax.set_title("Oracle ceilings, counterfactual accuracy if a stage were perfect")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_agreement_curve(curve: dict, out_path: Path) -> None:
    setup_plot_style()
    levels = curve.get("levels", [])
    if not levels:
        return
    labels = [lv["agreement"] for lv in levels]
    rates = [lv["rate"] for lv in levels]
    ns = [lv["n"] for lv in levels]
    los = [lv["ci_low"] for lv in levels]
    his = [lv["ci_high"] for lv in levels]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, rates, color=STYLE.success, alpha=0.85)
    err_low = [max(0.0, r - lo) for r, lo in zip(rates, los)]
    err_high = [max(0.0, hi - r) for r, hi in zip(rates, his)]
    ax.errorbar(x, rates,
                yerr=[err_low, err_high],
                fmt="none", ecolor="black", capsize=3)
    for i, (r, n) in enumerate(zip(rates, ns)):
        ax.text(i, r + 0.015, f"{r:.1%}\n(n={n})", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("agents agreeing on top-1 country (initial round)")
    ax.set_ylabel("final accuracy")
    ax.set_title("Agreement vs. accuracy")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Entrypoint

def run(results_dir: Path, gt_csv: Path, out_dir: Path) -> dict:
    records = load_run(results_dir, gt_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    funnel = compute_funnel(records)
    ceilings = compute_oracle_ceilings(records)
    agree = agreement_curve(records)
    paths = path_comparison(records)
    severity = severity_breakdown(records)

    summary = {
        "funnel": funnel,
        "oracle_ceilings": ceilings,
        "agreement_curve": agree,
        "path_comparison": paths,
        "severity": severity,
    }

    plot_funnel(funnel, plots_dir / "funnel.png")
    plot_oracle_bars(ceilings, plots_dir / "oracle_ceilings.png")
    plot_agreement_curve(agree, plots_dir / "agreement_curve.png")

    out_file = out_dir / "funnel_metrics.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[funnel] wrote {out_file}")
    print(
        f"[funnel] n={funnel['n_total']}  monotone={funnel['monotone']}  "
        f"actual={ceilings['actual']['rate']:.1%}  "
        f"oracle_pool={ceilings['oracle_pool']['rate']:.1%}"
    )
    if funnel.get("bottleneck"):
        b = funnel["bottleneck"]
        print(f"[funnel] bottleneck: {b['stage_code']}, {b['description']} "
              f"(conditional rate {b['conditional_rate']:.1%})")
    return summary
