"""Cross-agent influence metrics.

Three angles:

1. **Revision (Path B only)**, diff each agent's initial top-1 against its
   country-round top-1. Buckets: productive (wrong→right), destructive
   (right→wrong), preserved-correct, preserved-wrong.

2. **Persuasion matrix**, when agent A's initial top-1 == final result and
   agent B's initial top-1 ≠ final, did B's country-round prediction move
   toward A's? Aggregated into a 5×5 ``persuader → persuaded`` matrix.

3. **Hypothesis & tournament influence**, for each hypothesis, count
   agent support; flag cases where the minority view won and the majority
   was overridden. For each tournament match, count how often each agent's
   initial top-1 corresponded to the winner vs. the loser.

Outputs: ``influence.json`` plus a ``persuasion_matrix.png`` heatmap.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vlm_council.evaluate import _normalize_country
from eval_tourn.loader import AGENT_NAMES, RunRecord, countries_match, load_run, top1_country


SUPPORT = {"strongly_support", "support"}
CONTRADICT = {"contradicts", "strongly_contradicts"}


# Revision analysis (Path B)

def _revision_buckets(records: list[RunRecord]) -> dict[str, dict]:
    """Per-agent counts of how the country round revised the initial round.

    Only Path-B records are analyzed (Path A skips the country round entirely).
    """
    out: dict[str, dict] = {}
    path_b = [r for r in records if r.path == "B"]

    for agent in AGENT_NAMES:
        productive = destructive = preserved_correct = preserved_wrong = 0
        agreed_changed_top1 = 0
        n = 0

        for r in path_b:
            initial = (r.assessments or {}).get(agent)
            country = (r.country_assessments or {}).get(agent)
            if not initial or not country:
                continue

            t_initial = top1_country(initial)
            t_country = top1_country(country)
            if not t_initial or not t_country:
                continue
            n += 1

            init_correct = countries_match(t_initial, r.truth_country_code)
            ctry_correct = countries_match(t_country, r.truth_country_code)

            if t_initial != t_country:
                agreed_changed_top1 += 1
                if not init_correct and ctry_correct:
                    productive += 1
                elif init_correct and not ctry_correct:
                    destructive += 1
                elif init_correct and ctry_correct:
                    preserved_correct += 1
                else:
                    preserved_wrong += 1
            else:
                if init_correct:
                    preserved_correct += 1
                else:
                    preserved_wrong += 1

        out[agent] = {
            "n_with_both_rounds": n,
            "changed_top1": agreed_changed_top1,
            "productive_revisions": productive,    # wrong → correct
            "destructive_revisions": destructive,  # correct → wrong
            "preserved_correct": preserved_correct,
            "preserved_wrong": preserved_wrong,
            "net_productive": productive - destructive,
        }
    return out


# Persuasion matrix

def _persuasion_matrix(records: list[RunRecord]) -> dict:
    """Detect when one agent's view 'pulled' another agent's country-round.

    For Path-B records:
      - Find the set of "persuader" agents whose initial top-1 == final result.
      - Find the set of "moved" agents whose initial top-1 ≠ final but whose
        country-round top-1 == final.
      - For every (persuader, moved) pair, increment the matrix.
    """
    matrix: dict[tuple[str, str], int] = defaultdict(int)
    moved_total: Counter = Counter()
    persuader_total: Counter = Counter()
    convergence_correct = 0
    convergence_incorrect = 0

    for r in records:
        if r.path != "B":
            continue
        final = r.pred_country
        if not final:
            continue

        persuaders: list[str] = []
        moved: list[tuple[str, bool]] = []  # (agent, ended_correct)

        for agent in AGENT_NAMES:
            initial = (r.assessments or {}).get(agent)
            country = (r.country_assessments or {}).get(agent)
            t_initial = top1_country(initial) if initial else None
            t_country = top1_country(country) if country else None

            if t_initial and t_initial == final:
                persuaders.append(agent)

            if (
                t_initial and t_country
                and t_initial != final
                and t_country == final
            ):
                ended_correct = countries_match(final, r.truth_country_code)
                moved.append((agent, ended_correct))

        for p in persuaders:
            persuader_total[p] += 1
            for m, ok in moved:
                if m == p:
                    continue
                matrix[(p, m)] += 1
                moved_total[m] += 1
                if ok:
                    convergence_correct += 1
                else:
                    convergence_incorrect += 1

    return {
        "matrix": {f"{p}->{m}": n for (p, m), n in sorted(matrix.items())},
        "persuader_totals": dict(persuader_total),
        "moved_totals": dict(moved_total),
        "convergence_correct": convergence_correct,
        "convergence_incorrect": convergence_incorrect,
        "_matrix": matrix,  # raw; stripped before serializing
    }


# Hypothesis-level minority/majority

def _hypothesis_consensus(records: list[RunRecord]) -> dict:
    """For each (image, hypothesis_id), see whether the eventual final result
    contradicted majority sentiment.

    A hypothesis "won" if it matches the final country prediction (case-insensitive
    on the value embedded in ``hypothesis_id``).
    """
    minority_won = 0
    majority_overridden = 0
    no_clear_majority = 0
    n_hypotheses_total = 0

    for r in records:
        per_hyp_support: dict[str, Counter] = defaultdict(Counter)
        per_hyp_value: dict[str, str] = {}

        for ev in r.hypothesis_evaluations:
            hid = ev.get("hypothesis_id", "") or ""
            if not hid:
                continue
            conf = (ev.get("confidence") or "").lower()
            if conf in SUPPORT:
                per_hyp_support[hid]["support"] += 1
            elif conf in CONTRADICT:
                per_hyp_support[hid]["contradict"] += 1
            else:
                per_hyp_support[hid]["neutral"] += 1
            if hid not in per_hyp_value:
                if hid.startswith("country_"):
                    per_hyp_value[hid] = _normalize_country(hid[len("country_"):].replace("_", " "))
                elif hid.startswith("region_"):
                    per_hyp_value[hid] = hid[len("region_"):].replace("_", " ").lower()
                else:
                    per_hyp_value[hid] = hid

        for hid, counts in per_hyp_support.items():
            n_hypotheses_total += 1
            value = per_hyp_value.get(hid, "")
            won = (
                value
                and value.startswith("country")
                and countries_match(value.replace("country ", ""), r.truth_country_code)
            )
            sup = counts["support"]
            con = counts["contradict"]
            if sup >= 4 and not won:
                majority_overridden += 1
            elif con >= 4 and won:
                minority_won += 1
            elif sup == con:
                no_clear_majority += 1

    return {
        "n_hypotheses_total": n_hypotheses_total,
        "minority_right_majority_wrong": minority_won,
        "majority_supported_but_lost": majority_overridden,
        "no_clear_majority": no_clear_majority,
    }


# Tournament-level

def _tournament_alignment(records: list[RunRecord]) -> dict:
    """For each tournament match, did each agent's initial top-1 align with the
    winner or the loser?
    """
    per_agent = {a: {"with_winner": 0, "with_loser": 0, "neither": 0} for a in AGENT_NAMES}
    n_matches = 0

    for r in records:
        if not r.tournament_log:
            continue
        for match in r.tournament_log:
            winner = _normalize_country(match.get("winner", "") or "")
            a = _normalize_country(match.get("country_a", "") or "")
            b = _normalize_country(match.get("country_b", "") or "")
            if not winner:
                continue
            loser = b if winner == a else a
            n_matches += 1
            for agent in AGENT_NAMES:
                t = top1_country((r.assessments or {}).get(agent))
                if not t:
                    continue
                if t == winner:
                    per_agent[agent]["with_winner"] += 1
                elif t == loser:
                    per_agent[agent]["with_loser"] += 1
                else:
                    per_agent[agent]["neither"] += 1

    return {
        "n_matches": n_matches,
        "per_agent_alignment": per_agent,
    }


# Plot

def plot_persuasion_matrix(matrix: dict[tuple[str, str], int], out_path: Path) -> None:
    n = len(AGENT_NAMES)
    grid = np.zeros((n, n), dtype=int)
    for (p, m), v in matrix.items():
        if p in AGENT_NAMES and m in AGENT_NAMES:
            grid[AGENT_NAMES.index(p), AGENT_NAMES.index(m)] = v

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, cmap="YlOrRd")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(AGENT_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(AGENT_NAMES)
    ax.set_xlabel("moved agent")
    ax.set_ylabel("persuader")
    ax.set_title("Persuasion matrix (Path B)\nrow's initial = final, col flipped to it")
    for i in range(n):
        for j in range(n):
            if grid[i, j]:
                ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                        color="black" if grid[i, j] < grid.max() * 0.6 else "white")
    fig.colorbar(im, ax=ax, label="# images")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compute(records: list[RunRecord]) -> dict:
    persuasion = _persuasion_matrix(records)
    raw_matrix = persuasion.pop("_matrix")
    return {
        "n_total": len(records),
        "n_path_b": sum(1 for r in records if r.path == "B"),
        "revisions": _revision_buckets(records),
        "persuasion": persuasion,
        "_persuasion_matrix_raw": raw_matrix,
        "hypothesis_consensus": _hypothesis_consensus(records),
        "tournament_alignment": _tournament_alignment(records),
    }


def run(results_dir: Path, gt_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    records = load_run(results_dir, gt_csv)
    metrics = compute(records)
    plot_persuasion_matrix(metrics["_persuasion_matrix_raw"], plots_dir / "persuasion_matrix.png")

    serializable = {k: v for k, v in metrics.items() if not k.startswith("_")}
    out_file = out_dir / "influence.json"
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[influence] wrote {out_file}")
    for agent in AGENT_NAMES:
        rev = metrics["revisions"][agent]
        print(f"[influence] {agent:11s} productive={rev['productive_revisions']:2d}  "
              f"destructive={rev['destructive_revisions']:2d}  "
              f"net={rev['net_productive']:+d}  (n={rev['n_with_both_rounds']})")
    return serializable
