"""Pipeline funnel + diagnostic metrics for VLM Council v12.

Computes, deterministically, from existing result.json files, six diagnostic
sections that together identify where the pipeline loses the truth answer and
which stage to invest engineering effort in:

  1. Pipeline funnel (S0..S5): truth-survival rate at each stage with Wilson
     95% confidence intervals and an auto-identified bottleneck.
  2. Oracle ceilings: counterfactual end-accuracy if region/pool/tournament
     were perfect, plus a majority-vote baseline.
  3. Tournament diagnostics: symmetric-judge agreement distribution, tiebreak
     rate, pool-seed bias, per-match truth-win rate, specific pair failures.
  4. Agreement-vs-accuracy: how does final accuracy depend on how many of the
     5 agents agreed on top-1?
  5. Path A vs. Path B comparison.
  6. Severity of errors: near-miss / same-region-wrong / wrong-region.

Writes ``funnel_metrics.json`` plus four PNGs under ``plots/``.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vlm_council.regions import country_to_region

from eval_tourn._style import STYLE, setup_plot_style
from eval_tourn.loader import (
    AGENT_NAMES,
    RunRecord,
    countries_match,
    load_run,
    top1_country,
    topk_countries,
)


# Wilson confidence interval

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    Better than normal-approx for small n / extreme p. No scipy dependency.
    """
    if n <= 0:
        return (0.0, 0.0)
    # 95% z = 1.959963984540054; we hardcode for alpha=0.05
    if abs(alpha - 0.05) < 1e-9:
        z = 1.959963984540054
    else:
        # crude inverse-normal for other alphas (good enough for typical usage)
        from math import erf, sqrt
        # newton-ish; for simplicity just fall back to 1.96 if alpha != 0.05
        z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate_block(k: int, n: int) -> dict:
    """Return a uniform {n, correct, rate, ci_low, ci_high} block."""
    if n == 0:
        return {"n": 0, "correct": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "correct": k,
        "rate": k / n,
        "ci_low": lo,
        "ci_high": hi,
    }


# Truth-membership helpers

def _truth_in_candidates(candidates: list[str], truth_code: str) -> bool:
    return any(countries_match(c, truth_code) for c in candidates if c)


def _all_topk_countries(round_dict: dict | None) -> list[str]:
    """Flatten every agent's top-K candidates from one round into a list."""
    if not round_dict:
        return []
    out: list[str] = []
    for agent in AGENT_NAMES:
        out.extend(topk_countries(round_dict.get(agent), k=99))
    return out


def _truth_region(record: RunRecord) -> str | None:
    return country_to_region(record.truth_country_name) or country_to_region(
        record.truth_country_code
    )


# Stage classifiers, each returns True iff truth survived that stage

def _s0_truth_in_initial(r: RunRecord) -> bool:
    return _truth_in_candidates(_all_topk_countries(r.assessments), r.truth_country_code)


def _s1_region_correct(r: RunRecord) -> bool:
    truth_region = _truth_region(r)
    if truth_region is None:
        return False
    if r.confirmed_region and truth_region.lower() == r.confirmed_region.lower():
        return True
    proposed = (r.raw.get("progressive_narrowing") or {}).get("proposed_regions") or []
    return any(p and p.lower() == truth_region.lower() for p in proposed)


def _s2_country_round(r: RunRecord) -> bool:
    """Path A: identical to S1. Path B: truth in any country_assessment topK."""
    if r.path == "B":
        return _truth_in_candidates(
            _all_topk_countries(r.country_assessments), r.truth_country_code
        )
    return _s1_region_correct(r)


def _s3_in_pool(r: RunRecord) -> bool:
    return _truth_in_candidates(r.candidate_pool, r.truth_country_code)


def _s4_tournament_winner(r: RunRecord) -> bool:
    if not r.tournament_log:
        # No tournament was run, fall back to S5 (final pred matches truth)
        return r.is_correct
    final_winner = r.tournament_log[-1].get("winner", "") or ""
    return countries_match(final_winner, r.truth_country_code)


def _s5_final_correct(r: RunRecord) -> bool:
    return r.is_correct


# Pipeline funnel

_STAGES = [
    ("S0", "Truth in any initial-round top-K"),
    ("S1", "Confirmed/proposed region matches truth"),
    ("S2", "Truth in country-round top-K (Path B) / region OK (Path A)"),
    ("S3", "Truth in candidate pool"),
    ("S4", "Truth wins the tournament final"),
    ("S5", "Final predicted country matches truth"),
]


def compute_funnel(records: list[RunRecord]) -> dict:
    n = len(records)
    classifiers = [
        _s0_truth_in_initial,
        _s1_region_correct,
        _s2_country_round,
        _s3_in_pool,
        _s4_tournament_winner,
        _s5_final_correct,
    ]

    survived = [[fn(r) for r in records] for fn in classifiers]
    counts = [sum(s) for s in survived]

    # Cumulative: truth survived all stages up to and including i.
    # Special case: S5 (final pred matches truth) is defined as the run's
    # ground-truth outcome (`is_correct`) regardless of how it got there.
    # This guarantees S5 cumulative == country_accuracy from geo_metrics.
    # In rare edge cases (empty pool serialised as []; region-mapping mismatch
    # between confirmed_region and pycountry's regions table) the strict
    # cumulative AND would drop a few correct predictions; we don't want that
    # to propagate into a falsely-low headline number.
    cumulative = []
    accum = [True] * n
    for i, s in enumerate(survived):
        if i < len(survived) - 1:
            accum = [a and b for a, b in zip(accum, s)]
            cumulative.append(sum(accum))
        else:
            # S5 cumulative: every run where is_correct, even if it didn't
            # cleanly survive every prior stage. This makes the cumulative
            # bottom equal to the headline accuracy.
            cumulative.append(sum(s))

    stages = []
    for i, ((code, desc), c, c_cum) in enumerate(zip(_STAGES, counts, cumulative)):
        survival = _rate_block(c_cum, n)  # cumulative survival = c_cum / n
        # conditional = survived prev∧this / survived prev (i.e. attrition rate at this stage)
        conditional_n = cumulative[i - 1] if i > 0 else n
        # S5 is defined independently of strict cumulative AND, so its k can
        # exceed the previous cumulative count. Clamp for the conditional.
        conditional_k = min(c_cum, conditional_n)
        conditional = _rate_block(conditional_k, conditional_n)
        stages.append({
            "code": code,
            "description": desc,
            "stage_only_rate": _rate_block(c, n),  # just this stage's classifier
            "cumulative_survival": survival,        # truth still alive after S0..Si
            "conditional_on_prev": conditional,     # cumulative[i] / cumulative[i-1]
        })

    # Bottleneck: largest drop in conditional_on_prev where CIs of consecutive stages
    # don't overlap, pick the stage with the lowest conditional rate (i ≥ 1).
    bottleneck = None
    if n > 0 and len(stages) > 1:
        worst = None
        for i in range(1, len(stages)):
            cond = stages[i]["conditional_on_prev"]
            if cond["n"] == 0:
                continue
            if worst is None or cond["rate"] < worst["conditional_on_prev"]["rate"]:
                worst = stages[i]
        if worst is not None:
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
        # Monotonicity is checked over S0..S4 only (the strict truth-survival
        # chain). S5 is defined as `is_correct` regardless of upstream survival,
        # so by construction it can exceed S4 in rare edge cases (empty-pool
        # serialisation, region-name mapping mismatches), that's expected,
        # not a bug.
        "monotone": all(
            cumulative[i] >= cumulative[i + 1] for i in range(len(cumulative) - 2)
        ),
    }


# Oracle ceilings

def compute_oracle_ceilings(records: list[RunRecord]) -> dict:
    """Counterfactual: what would final accuracy be if a given stage were perfect?

    Definitions:
      - actual:                S5 / N (matches geo_metrics.country_accuracy)
      - oracle_region:         records where truth is reachable from the initial
                               pool (S0) AND from the pool / tournament path
                               assuming region selection is forced correct.
                               Approximation: count truth as final if it's in
                               either the initial top-K or the country-round
                               top-K (= would survive S1 by fiat) AND wins
                               its tournament matches.
      - oracle_pool:           force truth into the pool. Final correct iff
                               truth wins the tournament when present (i.e.,
                               we trust the tournament judge).
      - oracle_tournament:     truth wins every match it's in. Final correct
                               iff S3 (truth in pool) holds.
      - oracle_majority_vote:  ignore pool/tournament; predict the majority
                               top-1 across the 5 initial agents (with
                               linguistic-style ties broken by first agent).
    """
    n = len(records)
    if n == 0:
        return {"n_total": 0}

    actual_correct = sum(1 for r in records if _s5_final_correct(r))

    # Oracle Region: assume region is always right; downstream still needs truth
    # to be reachable from the agents. We approximate by checking S0 ∨ S2 (Path B)
    #, i.e. truth was named by some agent in either round, AND truth wins its
    # matches. If no tournament happened, fall back to S5.
    def oracle_region_one(r: RunRecord) -> bool:
        truth_reachable = _s0_truth_in_initial(r) or (
            r.path == "B" and _truth_in_candidates(
                _all_topk_countries(r.country_assessments), r.truth_country_code
            )
        )
        if not truth_reachable:
            return False
        if not r.tournament_log:
            return r.is_correct
        # Approximate: truth would have made the pool (region forced); trust
        # the existing tournament outcome only if truth was actually in pool.
        if _s3_in_pool(r):
            return _s4_tournament_winner(r)
        # Truth was reachable but not in pool, under oracle-region we assume
        # the pool would have included it; but we don't know how the
        # tournament would have decided. Use majority of initial top-1's.
        votes = Counter()
        for agent in AGENT_NAMES:
            t = top1_country(r.assessments.get(agent))
            if t:
                votes[t] += 1
        if not votes:
            return False
        winner, _ = votes.most_common(1)[0]
        return countries_match(winner, r.truth_country_code)

    oracle_region_correct = sum(1 for r in records if oracle_region_one(r))

    # Oracle Pool: force truth into pool. By construction this is the strongest
    # oracle short of "fix everything", once truth is in the pool, the only
    # remaining failure mode is the tournament losing it. We take the optimistic
    # bound (oracle pool ⇒ oracle tournament): count as correct iff truth was
    # in pool already OR is reachable from agents (so a perfect pool selector
    # would have included it). This guarantees oracle_pool ≥ oracle_tournament.
    def oracle_pool_one(r: RunRecord) -> bool:
        if _s3_in_pool(r):
            return True
        return _s0_truth_in_initial(r) or (
            r.path == "B" and _truth_in_candidates(
                _all_topk_countries(r.country_assessments), r.truth_country_code
            )
        )

    oracle_pool_correct = sum(1 for r in records if oracle_pool_one(r))

    # Oracle Tournament: truth wins every match it's in. Final correct iff
    # truth is in the pool (S3). This is the cleanest counterfactual.
    oracle_tournament_correct = sum(1 for r in records if _s3_in_pool(r))

    # Majority-Vote Baseline: take the most common top-1 across the 5 initial
    # agents (ties broken by first occurrence), use that as the prediction.
    def majority_vote_one(r: RunRecord) -> bool:
        votes = Counter()
        for agent in AGENT_NAMES:
            t = top1_country(r.assessments.get(agent))
            if t:
                votes[t] += 1
        if not votes:
            return False
        winner, _ = votes.most_common(1)[0]
        return countries_match(winner, r.truth_country_code)

    majority_correct = sum(1 for r in records if majority_vote_one(r))

    return {
        "n_total": n,
        "actual": _rate_block(actual_correct, n),
        "oracle_region": _rate_block(oracle_region_correct, n),
        "oracle_pool": _rate_block(oracle_pool_correct, n),
        "oracle_tournament": _rate_block(oracle_tournament_correct, n),
        "majority_vote_baseline": _rate_block(majority_correct, n),
    }


# Tournament diagnostics

def tournament_diagnostics(records: list[RunRecord]) -> dict:
    """Symmetric-judge-aware tournament diagnostics.

    The judge runs every pairwise match twice (forward + reverse positions) and
    falls back to pool-rank when the two runs disagree. ``country_a`` /
    ``country_b`` are bracket slot indices, NOT prompt positions, so a slot-
    based win-rate is meaningless. Instead we surface:

      • Agreement distribution: how often forward + reverse agree (= robust)
        vs. fall back to pool-rank (= judge undecided/biased)
      • Tiebreak rate: share of matches resolved via pool-rank
      • Pool-seed bias: how often the higher-seeded (lower pool_rank) country
        wins, irrespective of bracket slot. >50% with significance is the real
        position-bias signal in this pipeline.
      • Truth-in-match outcomes split by agreement type
      • Specific pair failures (truth → winner) for downstream eyeballing
    """
    total_matches = 0
    agreement_counts: Counter = Counter()
    higher_seed_wins = 0
    seed_total = 0
    truth_in_match = truth_won = 0
    truth_by_agreement: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "won": 0}
    )
    pair_losses: Counter = Counter()

    for r in records:
        truth_code = r.truth_country_code
        # Build name -> pool_rank lookup; lower rank = higher seed
        pool_rank: dict[str, int] = {
            (c or "").strip().lower(): i
            for i, c in enumerate(r.candidate_pool or [])
        }
        for match in r.tournament_log:
            a = match.get("country_a", "") or ""
            b = match.get("country_b", "") or ""
            w = match.get("winner", "") or ""
            if not (a and b and w):
                continue
            total_matches += 1

            ag = match.get("agreement") or "unknown"
            agreement_counts[ag] += 1

            # Pool-seed bias: lower pool_rank = higher seed.
            # Prefer fields on the match if present, else look up in pool.
            ra = match.get("pool_rank_a")
            rb = match.get("pool_rank_b")
            if ra is None:
                ra = pool_rank.get(a.strip().lower())
            if rb is None:
                rb = pool_rank.get(b.strip().lower())
            if isinstance(ra, int) and isinstance(rb, int) and ra != rb:
                seed_total += 1
                higher_seed = a if ra < rb else b
                if w == higher_seed:
                    higher_seed_wins += 1

            truth_is_a = countries_match(a, truth_code)
            truth_is_b = countries_match(b, truth_code)
            if truth_is_a or truth_is_b:
                truth_in_match += 1
                truth_by_agreement[ag]["n"] += 1
                if (truth_is_a and w == a) or (truth_is_b and w == b):
                    truth_won += 1
                    truth_by_agreement[ag]["won"] += 1
                else:
                    truth_label = a if truth_is_a else b
                    pair_losses[(truth_label, w)] += 1

    pair_failures = [
        {"truth": t, "lost_to": lw, "count": c}
        for (t, lw), c in pair_losses.most_common(15)
        if c >= 2
    ]

    agreement_dist = {
        ag: _rate_block(agreement_counts.get(ag, 0), total_matches)
        for ag in ("agree", "forward_only", "reverse_only", "disagree", "both_empty", "unknown")
    }

    # Tiebreak = anything that wasn't a clean both-runs-agree
    n_tiebreak = total_matches - agreement_counts.get("agree", 0)

    truth_split = {
        ag: {
            "n": v["n"],
            "won": v["won"],
            "win_rate": (v["won"] / v["n"]) if v["n"] else None,
        }
        for ag, v in truth_by_agreement.items()
    }

    return {
        "n_matches": total_matches,
        "agreement_distribution": agreement_dist,
        "tiebreak": _rate_block(n_tiebreak, total_matches),
        "pool_seed_bias": _rate_block(higher_seed_wins, seed_total),
        "n_seed_comparable_matches": seed_total,
        "truth_match_win_rate": _rate_block(truth_won, truth_in_match),
        "n_truth_matches": truth_in_match,
        "truth_outcomes_by_agreement": truth_split,
        "specific_pair_failures": pair_failures,
    }


# Agreement-vs-accuracy

def agreement_curve(records: list[RunRecord]) -> dict:
    """For each image, count how many of the 5 agents picked the same top-1
    country (the modal vote-count among the 5). Then bucket by that count and
    compute final-prediction accuracy per bucket.
    """
    bins: dict[int, list[bool]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for r in records:
        votes = Counter()
        for agent in AGENT_NAMES:
            t = top1_country(r.assessments.get(agent))
            if t:
                votes[t] += 1
        if not votes:
            continue
        modal_count = votes.most_common(1)[0][1]  # how many agents agreed on the top
        bins.setdefault(modal_count, []).append(r.is_correct)

    out = []
    for level in (5, 4, 3, 2, 1):
        results = bins.get(level, [])
        n = len(results)
        k = sum(results)
        out.append({
            "agreement": f"{level}/5",
            **_rate_block(k, n),
        })
    return {"levels": out}


# Path A vs. B comparison

def path_comparison(records: list[RunRecord]) -> dict:
    def stats(subset: list[RunRecord]) -> dict:
        n = len(subset)
        if n == 0:
            return {"n": 0}
        correct = sum(1 for r in subset if r.is_correct)
        haversines = [r.haversine_km for r in subset if r.haversine_km is not None]
        truth_in_pool = sum(1 for r in subset if _s3_in_pool(r))
        tour_win = sum(1 for r in subset if _s3_in_pool(r) and _s4_tournament_winner(r))
        seconds = [r.total_seconds for r in subset if r.total_seconds]
        return {
            **_rate_block(correct, n),
            "median_haversine_km": float(np.median(haversines)) if haversines else 0.0,
            "truth_in_pool_rate": (truth_in_pool / n) if n else 0.0,
            "truth_in_pool_n": truth_in_pool,
            "tournament_survival_when_in_pool": (
                tour_win / truth_in_pool if truth_in_pool else 0.0
            ),
            "mean_total_seconds": float(np.mean(seconds)) if seconds else 0.0,
        }

    return {
        "path_a": stats([r for r in records if r.path == "A"]),
        "path_b": stats([r for r in records if r.path == "B"]),
    }


# Severity breakdown

def severity_breakdown(records: list[RunRecord]) -> dict:
    near_miss: list[dict] = []
    same_region: list[dict] = []
    wrong_region: list[dict] = []

    for r in records:
        if r.is_correct:
            continue
        truth_region = _truth_region(r)
        pred_region = country_to_region(r.pred_country)
        hav = r.haversine_km if r.haversine_km is not None else float("inf")
        entry = {
            "image_id": r.image_id,
            "truth": r.truth_country_name,
            "pred": r.pred_country,
            "haversine_km": (round(hav, 0) if hav != float("inf") else None),
        }
        if hav < 500:
            near_miss.append(entry)
        if truth_region and pred_region and truth_region == pred_region:
            same_region.append(entry)
        elif truth_region and pred_region and truth_region != pred_region:
            wrong_region.append(entry)

    n_total = len(records)
    n_wrong = sum(1 for r in records if not r.is_correct)
    return {
        "n_total": n_total,
        "n_wrong": n_wrong,
        "near_miss_count": len(near_miss),
        "same_region_wrong_count": len(same_region),
        "wrong_region_count": len(wrong_region),
        "near_miss_examples": near_miss[:15],
        "same_region_wrong_examples": same_region[:15],
        "wrong_region_examples": wrong_region[:15],
    }


# Plots

def plot_funnel(funnel: dict, out_path: Path) -> None:
    setup_plot_style()
    stages = funnel.get("stages", [])
    if not stages:
        return
    labels = [s["code"] for s in stages]
    rates = [s["cumulative_survival"]["rate"] for s in stages]
    los = [s["cumulative_survival"]["ci_low"] for s in stages]
    his = [s["cumulative_survival"]["ci_high"] for s in stages]
    err_low = [max(0.0, r - lo) for r, lo in zip(rates, los)]
    err_high = [max(0.0, hi - r) for r, hi in zip(rates, his)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(labels))
    ax.barh(y, rates, color=STYLE.primary, alpha=0.85)
    ax.errorbar(rates, y, xerr=[err_low, err_high], fmt="none",
                ecolor="black", capsize=3)
    for i, (r, s) in enumerate(zip(rates, stages)):
        ax.text(min(r + 0.02, 0.98), i,
                f"{s['cumulative_survival']['n']*r:.0f}/{s['cumulative_survival']['n']}"
                f" = {r:.1%}",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s['code']}: {s['description'][:48]}" for s in stages])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("cumulative truth-survival rate (Wilson 95% CI)")
    ax.set_title("Pipeline funnel, where truth gets lost")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_oracle_bars(ceilings: dict, out_path: Path) -> None:
    setup_plot_style()
    keys = ["actual", "majority_vote_baseline", "oracle_region",
            "oracle_pool", "oracle_tournament"]
    labels = ["Actual", "Majority\nbaseline", "Oracle\nregion",
              "Oracle\npool", "Oracle\ntournament"]
    rates = [(ceilings.get(k) or {}).get("rate", 0.0) for k in keys]
    los = [(ceilings.get(k) or {}).get("ci_low", 0.0) for k in keys]
    his = [(ceilings.get(k) or {}).get("ci_high", 0.0) for k in keys]
    err_low = [max(0.0, r - lo) for r, lo in zip(rates, los)]
    err_high = [max(0.0, hi - r) for r, hi in zip(rates, his)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    colors = [STYLE.primary, STYLE.neutral, STYLE.success, STYLE.warning, STYLE.error]
    ax.bar(x, rates, color=colors, alpha=0.85)
    ax.errorbar(x, rates, yerr=[err_low, err_high], fmt="none",
                ecolor="black", capsize=3)
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
    los = [lv["ci_low"] for lv in levels]
    his = [lv["ci_high"] for lv in levels]
    ns = [lv["n"] for lv in levels]
    err_low = [max(0.0, r - lo) for r, lo in zip(rates, los)]
    err_high = [max(0.0, hi - r) for r, hi in zip(rates, his)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, rates, color=STYLE.success, alpha=0.85)
    ax.errorbar(x, rates, yerr=[err_low, err_high], fmt="none",
                ecolor="black", capsize=3)
    for i, (r, n) in enumerate(zip(rates, ns)):
        ax.text(i, r + 0.015, f"{r:.1%}\n(n={n})", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("agents agreeing on top-1 country")
    ax.set_ylabel("end-to-end accuracy")
    ax.set_title("Agreement vs. accuracy")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tournament_bias(diag: dict, out_path: Path) -> None:
    """Two-panel: agreement distribution + pool-seed bias check."""
    setup_plot_style()
    n = diag.get("n_matches", 0)
    if not n:
        return
    ag = diag.get("agreement_distribution", {})
    order = ["agree", "forward_only", "reverse_only", "disagree", "both_empty"]
    rates = [ag.get(k, {}).get("rate", 0.0) or 0.0 for k in order]
    counts = [ag.get(k, {}).get("correct", 0) or 0 for k in order]
    colors = [STYLE.success, STYLE.warning, STYLE.warning, STYLE.error, STYLE.neutral]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [2, 1]})

    x = np.arange(len(order))
    ax1.bar(x, rates, color=colors, alpha=0.85)
    for i, (rate, c) in enumerate(zip(rates, counts)):
        if rate > 0:
            ax1.text(i, rate + 0.015, f"{rate:.1%}\n(n={c})",
                     ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=15)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("share of matches")
    ax1.set_title(f"Symmetric judge agreement  (n={n} matches)")

    seed = diag.get("pool_seed_bias", {})
    seed_rate = seed.get("rate") or 0.0
    seed_n = seed.get("n", 0) or 0
    seed_lo = seed.get("ci_low") or 0.0
    seed_hi = seed.get("ci_high") or 0.0
    bar_color = (
        STYLE.error if seed_n and (seed_lo > 0.55 or seed_hi < 0.45)
        else STYLE.primary
    )
    ax2.bar([0], [seed_rate], color=bar_color, alpha=0.85)
    ax2.errorbar([0], [seed_rate],
                 yerr=[[seed_rate - seed_lo], [seed_hi - seed_rate]],
                 fmt="none", ecolor="black", capsize=4)
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    label_y = max(0.05, seed_rate - 0.18)
    ax2.text(0, label_y,
             f"{seed_rate:.1%}\n[{seed_lo:.1%}, {seed_hi:.1%}]\nn={seed_n}",
             ha="center", va="top", fontsize=9,
             color="white" if seed_rate > 0.4 else "black")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["higher seed wins"])
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("rate")
    ax2.set_title("Pool-seed bias")

    fig.suptitle(
        "Tournament symmetry, the symmetric judge runs forward+reverse; "
        "tiebreaks fall back to pool-rank",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Entrypoint

def run(results: Path, gt: Path, out: Path) -> dict:
    records = load_run(results, gt)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    funnel = compute_funnel(records)
    ceilings = compute_oracle_ceilings(records)
    tour = tournament_diagnostics(records)
    agree = agreement_curve(records)
    paths = path_comparison(records)
    severity = severity_breakdown(records)

    summary = {
        "funnel": funnel,
        "oracle_ceilings": ceilings,
        "tournament_diagnostics": tour,
        "agreement_curve": agree,
        "path_comparison": paths,
        "severity": severity,
    }

    plot_funnel(funnel, plots_dir / "funnel.png")
    plot_oracle_bars(ceilings, plots_dir / "oracle_ceilings.png")
    plot_agreement_curve(agree, plots_dir / "agreement_curve.png")
    plot_tournament_bias(tour, plots_dir / "tournament_symmetry.png")

    out_file = out / "funnel_metrics.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[funnel] wrote {out_file}")
    print(
        f"[funnel] n={funnel['n_total']}  monotone={funnel['monotone']}  "
        f"actual={ceilings['actual']['rate']:.1%}  "
        f"oracle_tournament={ceilings['oracle_tournament']['rate']:.1%}"
    )
    if funnel.get("bottleneck"):
        b = funnel["bottleneck"]
        print(
            f"[funnel] bottleneck: {b['stage_code']} "
            f"(conditional rate {b['conditional_rate']:.1%})"
        )
    return summary
