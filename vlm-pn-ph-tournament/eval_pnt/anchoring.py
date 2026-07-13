"""Anchoring-bias diagnostic for VLM Council v12.

How often does the tournament *just confirm* the pre-tournament top-seed,
versus actually re-rank candidates? When the tournament agrees with the
seeding, the bracket added no signal, the answer was already implied by
the specialists' confidence aggregation. When it disagrees, the bracket
genuinely re-ranked. This module measures both, and crosses them with
ground truth to quantify when anchoring helped, hurt, or was neutral.

Definitions (per image with a non-trivial pool of ≥ 2 candidates):

  pool_top1            = candidate_pool[0], the highest-seeded country
                         (specialist confidence sum, ties broken by pool order).
  tournament_winner    = the country that won the final match (or sole
                         survivor if the bracket was a walkover).
  anchored             = (tournament_winner == pool_top1).
  anchoring_correct    = anchored AND winner matches truth.
  anchoring_wrong      = anchored AND winner does not match truth.
  re_ranked            = NOT anchored.
  re_ranking_helped    = re_ranked AND new winner matches truth (and pool_top1
                         did not).
  re_ranking_hurt      = re_ranked AND pool_top1 matched truth but the new
                         winner does not.
  re_ranking_neutral   = re_ranked AND neither winner nor pool_top1 match
                         truth (truth was missed regardless).

Writes ``anchoring_metrics.json`` plus ``plots/anchoring.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_pnt._style import STYLE, setup_plot_style
from eval_pnt.loader import RunRecord, countries_match, load_run


def _winner(record: RunRecord) -> str | None:
    log = record.tournament_log or []
    if not log:
        # Walkover: the prediction itself is the de-facto winner if the
        # pool had exactly one survivor.
        if record.candidate_pool and len(record.candidate_pool) == 1:
            return record.candidate_pool[0]
        return None
    # Prefer the explicit "final" round; fall back to the last entry.
    for m in reversed(log):
        if m.get("round_label") == "final":
            return m.get("winner")
    return log[-1].get("winner")


def compute(records: list[RunRecord]) -> dict:
    n_total = len(records)
    n_with_pool = 0
    n_trivial_pool = 0  # pool size 0 or 1, no anchoring possible

    n_anchored = 0
    n_anchored_correct = 0
    n_anchored_wrong = 0

    n_re_ranked = 0
    n_re_rank_helped = 0
    n_re_rank_hurt = 0
    n_re_rank_neutral = 0

    examples_helped: list[dict] = []
    examples_hurt: list[dict] = []
    examples_anchored_wrong: list[dict] = []

    for r in records:
        pool = list(r.candidate_pool or [])
        if len(pool) <= 1:
            n_trivial_pool += 1
            continue
        n_with_pool += 1

        top_seed = pool[0]
        winner = _winner(r)
        if not winner:
            continue

        anchored = winner.strip().lower() == top_seed.strip().lower()

        truth = r.truth_country_code or r.truth_country_name
        winner_correct = countries_match(winner, truth)
        seed_correct = countries_match(top_seed, truth)

        if anchored:
            n_anchored += 1
            if winner_correct:
                n_anchored_correct += 1
            else:
                n_anchored_wrong += 1
                if len(examples_anchored_wrong) < 8:
                    examples_anchored_wrong.append({
                        "image_id": r.image_id,
                        "pool": pool,
                        "anchored_to": top_seed,
                        "truth": r.truth_country_name,
                    })
        else:
            n_re_ranked += 1
            if winner_correct and not seed_correct:
                n_re_rank_helped += 1
                if len(examples_helped) < 8:
                    examples_helped.append({
                        "image_id": r.image_id,
                        "pool": pool,
                        "seed_top1": top_seed,
                        "tournament_winner": winner,
                        "truth": r.truth_country_name,
                    })
            elif seed_correct and not winner_correct:
                n_re_rank_hurt += 1
                if len(examples_hurt) < 8:
                    examples_hurt.append({
                        "image_id": r.image_id,
                        "pool": pool,
                        "seed_top1": top_seed,
                        "tournament_winner": winner,
                        "truth": r.truth_country_name,
                    })
            else:
                n_re_rank_neutral += 1

    rate = lambda num, denom: (num / denom) if denom else None  # noqa: E731

    return {
        "n_total": n_total,
        "n_with_pool": n_with_pool,
        "n_trivial_pool": n_trivial_pool,
        "anchoring_rate": rate(n_anchored, n_with_pool),
        "n_anchored": n_anchored,
        "n_anchored_correct": n_anchored_correct,
        "n_anchored_wrong": n_anchored_wrong,
        "anchoring_correctness": rate(n_anchored_correct, n_anchored),
        "n_re_ranked": n_re_ranked,
        "re_ranking_rate": rate(n_re_ranked, n_with_pool),
        "n_re_rank_helped": n_re_rank_helped,
        "n_re_rank_hurt": n_re_rank_hurt,
        "n_re_rank_neutral": n_re_rank_neutral,
        "re_rank_net_help": n_re_rank_helped - n_re_rank_hurt,
        "examples_helped": examples_helped,
        "examples_hurt": examples_hurt,
        "examples_anchored_wrong": examples_anchored_wrong,
    }


def plot(metrics: dict, out_path: Path) -> bool:
    if not metrics or not metrics.get("n_with_pool"):
        return False
    setup_plot_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: anchored vs. re-ranked stacked
    n_pool = metrics["n_with_pool"]
    anchored_correct = metrics["n_anchored_correct"]
    anchored_wrong = metrics["n_anchored_wrong"]
    rerank_help = metrics["n_re_rank_helped"]
    rerank_hurt = metrics["n_re_rank_hurt"]
    rerank_neutral = metrics["n_re_rank_neutral"]

    ax_l.barh(["Anchored\n(winner = top seed)"], [anchored_correct],
              color=STYLE.success, label="confirmed truth")
    ax_l.barh(["Anchored\n(winner = top seed)"], [anchored_wrong],
              left=[anchored_correct], color=STYLE.error, label="anchored on wrong country")
    ax_l.barh(["Re-ranked\n(winner ≠ top seed)"], [rerank_help],
              color=STYLE.success)
    ax_l.barh(["Re-ranked\n(winner ≠ top seed)"], [rerank_hurt],
              left=[rerank_help], color=STYLE.error)
    ax_l.barh(["Re-ranked\n(winner ≠ top seed)"], [rerank_neutral],
              left=[rerank_help + rerank_hurt], color=STYLE.neutral,
              label="neutral (truth missed either way)")
    ax_l.set_xlim(0, n_pool)
    ax_l.set_xlabel(f"# images (of {n_pool} with non-trivial pool)")
    ax_l.set_title("Anchoring outcomes")
    ax_l.legend(loc="lower right", fontsize=8)
    ax_l.grid(True, axis="x", alpha=0.3)

    # Right: net effect of re-ranking on accuracy
    cats = ["Helped\n(rerank → truth)", "Hurt\n(rerank away from truth)", "Neutral"]
    vals = [rerank_help, rerank_hurt, rerank_neutral]
    colors = [STYLE.success, STYLE.error, STYLE.neutral]
    ax_r.bar(cats, vals, color=colors)
    ax_r.set_ylabel("# images")
    ax_r.set_title(
        f"Re-ranking effect  (net help = {rerank_help - rerank_hurt:+d})",
    )
    ax_r.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax_r.text(i, v + 0.1, str(v), ha="center", fontsize=9)

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

    (out_dir / "anchoring_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot(metrics, plots_dir / "anchoring.png")
    return metrics
