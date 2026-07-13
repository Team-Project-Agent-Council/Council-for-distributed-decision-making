"""RAG-utility diagnostic for VLM Council v12.

How often does retrieval get called, how many refs survive verification,
and how often does the presence/absence of refs actually swing a tournament
match? Three nested questions:

  1. Was retrieval invoked at all?, proxy: ``len(rag_refs_seen) > 0`` in the
     persisted state. The pipeline always *tries* to fetch refs once a pool
     is non-trivial, so this is mostly a "did at least one ref survive
     filter+verify?" signal.

  2. When refs were available, how was the budget split?, refs per country
     (from ``rag_refs_seen``), unique countries with verified refs, max-per-
     country, total verified.

  3. When refs were asymmetric in a match, did they decide it?, for each
     tournament match where one side has verified refs and the other has
     zero, count whether the ref-side won. Cross with ground truth: did
     ref-asymmetry steer the bracket *toward* truth or *away from* it?

Note: pre-v12.2 result.json files don't store per-match ref counts, only
aggregate ``rag_refs_seen``. We approximate per-match availability by the
total count for each country across the whole image. That's a slight
over-estimate (a country may have had refs in semi but not in final), but
the directional conclusions ("ref-asymmetry → ref-side wins X% of the time")
are robust.

Writes ``rag_metrics.json`` plus ``plots/rag_utility.png``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_tourn._style import STYLE, setup_plot_style
from eval_tourn.loader import RunRecord, countries_match, load_run


def _refs_per_country(record: RunRecord) -> Counter:
    """Count verified refs per country for this image."""
    seen = (record.raw or {}).get("rag_refs_seen") or []
    cnt: Counter = Counter()
    for entry in seen:
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            country = str(entry[0]).strip()
            if country:
                cnt[country.lower()] += 1
    return cnt


def _norm(country: str) -> str:
    return (country or "").strip().lower()


def compute(records: list[RunRecord]) -> dict:
    n_total = len(records)
    n_with_pool = 0
    n_with_any_refs = 0          # at least one ref survived
    n_with_no_refs = 0            # pool ≥ 2 but zero refs got through
    total_refs = 0
    refs_per_image: list[int] = []
    countries_with_refs_per_image: list[int] = []

    # Match-level: ref asymmetry vs. outcome
    n_matches_total = 0
    n_matches_both_have_refs = 0
    n_matches_neither_has_refs = 0
    n_matches_asymmetric = 0
    n_asym_ref_side_won = 0
    n_asym_ref_side_was_truth = 0
    n_asym_ref_side_was_truth_and_won = 0
    n_asym_no_ref_side_was_truth = 0
    n_asym_no_ref_side_won = 0  # the side without refs still won

    examples_ref_side_won_truth: list[dict] = []
    examples_ref_side_won_wrong: list[dict] = []
    examples_no_ref_side_won_anyway: list[dict] = []

    for r in records:
        pool = list(r.candidate_pool or [])
        if len(pool) < 2:
            continue
        n_with_pool += 1

        per_country = _refs_per_country(r)
        n_refs = sum(per_country.values())
        total_refs += n_refs
        refs_per_image.append(n_refs)
        countries_with_refs_per_image.append(len(per_country))

        if n_refs > 0:
            n_with_any_refs += 1
        else:
            n_with_no_refs += 1

        # Match-level
        for m in (r.tournament_log or []):
            country_a = m.get("country_a")
            country_b = m.get("country_b")
            winner = m.get("winner")
            if not (country_a and country_b and winner):
                continue
            n_matches_total += 1

            refs_a = per_country.get(_norm(country_a), 0)
            refs_b = per_country.get(_norm(country_b), 0)

            if refs_a > 0 and refs_b > 0:
                n_matches_both_have_refs += 1
                continue
            if refs_a == 0 and refs_b == 0:
                n_matches_neither_has_refs += 1
                continue

            n_matches_asymmetric += 1
            ref_side = country_a if refs_a > 0 else country_b
            no_ref_side = country_b if refs_a > 0 else country_a
            ref_side_won = (winner.strip().lower() == ref_side.strip().lower())

            truth_code = r.truth_country_code
            ref_side_is_truth = countries_match(ref_side, truth_code)
            no_ref_side_is_truth = countries_match(no_ref_side, truth_code)

            if ref_side_won:
                n_asym_ref_side_won += 1
            else:
                n_asym_no_ref_side_won += 1
                if len(examples_no_ref_side_won_anyway) < 8:
                    examples_no_ref_side_won_anyway.append({
                        "image_id": r.image_id,
                        "match": m.get("round_label"),
                        "ref_side": ref_side,
                        "no_ref_side": no_ref_side,
                        "winner": winner,
                        "truth": r.truth_country_name,
                    })

            if ref_side_is_truth:
                n_asym_ref_side_was_truth += 1
                if ref_side_won:
                    n_asym_ref_side_was_truth_and_won += 1
                    if len(examples_ref_side_won_truth) < 8:
                        examples_ref_side_won_truth.append({
                            "image_id": r.image_id,
                            "match": m.get("round_label"),
                            "winner": winner,
                            "truth": r.truth_country_name,
                        })
            elif no_ref_side_is_truth:
                n_asym_no_ref_side_was_truth += 1
                if ref_side_won and len(examples_ref_side_won_wrong) < 8:
                    examples_ref_side_won_wrong.append({
                        "image_id": r.image_id,
                        "match": m.get("round_label"),
                        "ref_side_winner": ref_side,
                        "truth_was": r.truth_country_name,
                    })

    rate = lambda n, d: (n / d) if d else None  # noqa: E731

    return {
        # Coverage
        "n_total": n_total,
        "n_with_pool": n_with_pool,
        "n_with_any_refs": n_with_any_refs,
        "n_with_no_refs": n_with_no_refs,
        "ref_coverage_rate": rate(n_with_any_refs, n_with_pool),
        "total_refs_across_dataset": total_refs,
        "mean_refs_per_image": (sum(refs_per_image) / len(refs_per_image)) if refs_per_image else 0,
        "mean_countries_with_refs_per_image": (
            sum(countries_with_refs_per_image) / len(countries_with_refs_per_image)
        ) if countries_with_refs_per_image else 0,

        # Matches
        "n_matches_total": n_matches_total,
        "n_matches_both_have_refs": n_matches_both_have_refs,
        "n_matches_neither_has_refs": n_matches_neither_has_refs,
        "n_matches_asymmetric": n_matches_asymmetric,
        "asymmetric_match_share": rate(n_matches_asymmetric, n_matches_total),

        # Outcome of asymmetric matches, does having refs predict winning?
        "asym_ref_side_win_rate": rate(n_asym_ref_side_won, n_matches_asymmetric),
        "n_asym_ref_side_won": n_asym_ref_side_won,
        "n_asym_no_ref_side_won": n_asym_no_ref_side_won,

        # Crossed with truth: did the ref-side actually deserve to win?
        "n_asym_ref_side_was_truth": n_asym_ref_side_was_truth,
        "n_asym_ref_side_was_truth_and_won": n_asym_ref_side_was_truth_and_won,
        "n_asym_no_ref_side_was_truth": n_asym_no_ref_side_was_truth,
        # Of all asymmetric matches where ref-side won: how often was that correct?
        "asym_ref_side_won_correctly_rate": rate(
            n_asym_ref_side_was_truth_and_won, n_asym_ref_side_won
        ),

        "examples_ref_side_won_truth": examples_ref_side_won_truth,
        "examples_ref_side_won_wrong": examples_ref_side_won_wrong,
        "examples_no_ref_side_won_anyway": examples_no_ref_side_won_anyway,
    }


def plot(metrics: dict, out_path: Path) -> bool:
    if not metrics or not metrics.get("n_with_pool"):
        return False
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_cov, ax_match, ax_asym = axes

    # 1. Coverage: how many images had any verified refs?
    cov_labels = ["any verified\nrefs", "no verified\nrefs"]
    cov_vals = [metrics["n_with_any_refs"], metrics["n_with_no_refs"]]
    ax_cov.bar(cov_labels, cov_vals,
               color=[STYLE.success, STYLE.neutral])
    ax_cov.set_ylabel(f"# images (of {metrics['n_with_pool']} with non-trivial pool)")
    ax_cov.set_title("RAG coverage")
    for i, v in enumerate(cov_vals):
        ax_cov.text(i, v + 0.4, str(v), ha="center", fontsize=9)
    ax_cov.grid(True, axis="y", alpha=0.3)

    # 2. Matches: ref distribution across matches
    n_total_matches = metrics["n_matches_total"]
    match_labels = ["both have\nrefs", "asymmetric", "neither has\nrefs"]
    match_vals = [
        metrics["n_matches_both_have_refs"],
        metrics["n_matches_asymmetric"],
        metrics["n_matches_neither_has_refs"],
    ]
    ax_match.bar(match_labels, match_vals,
                 color=[STYLE.primary, STYLE.warning, STYLE.neutral])
    ax_match.set_ylabel(f"# matches (of {n_total_matches} total)")
    ax_match.set_title("Per-match ref availability")
    for i, v in enumerate(match_vals):
        ax_match.text(i, v + 0.4, str(v), ha="center", fontsize=9)
    ax_match.grid(True, axis="y", alpha=0.3)

    # 3. Asymmetric matches: does ref-side win? was it the truth-side?
    n_asym = metrics["n_matches_asymmetric"]
    if n_asym:
        won_correctly = metrics["n_asym_ref_side_was_truth_and_won"]
        won_wrong = metrics["n_asym_ref_side_won"] - won_correctly
        no_ref_side_won = metrics["n_asym_no_ref_side_won"]
        labels = ["ref-side won\n(= truth)",
                  "ref-side won\n(≠ truth)",
                  "no-ref side\nwon"]
        vals = [won_correctly, won_wrong, no_ref_side_won]
        colors = [STYLE.success, STYLE.error, STYLE.neutral]
        ax_asym.bar(labels, vals, color=colors)
        ax_asym.set_ylabel("# asymmetric matches")
        ax_asym.set_title(
            f"Outcome | asymmetric  (n={n_asym}, "
            f"ref-side wins {metrics['asym_ref_side_win_rate']:.0%})"
        )
        for i, v in enumerate(vals):
            ax_asym.text(i, v + 0.2, str(v), ha="center", fontsize=9)
        ax_asym.grid(True, axis="y", alpha=0.3)
    else:
        ax_asym.set_axis_off()
        ax_asym.text(0.5, 0.5, "No asymmetric matches\nin this run.",
                     ha="center", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def run(results_dir: Path, gt_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = load_run(results_dir, gt_csv)
    metrics = compute(records)
    metrics["schema_version"] = "2.0"

    (out_dir / "rag_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot(metrics, plots_dir / "rag_utility.png")
    return metrics
