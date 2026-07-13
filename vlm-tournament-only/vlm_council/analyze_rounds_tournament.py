"""Analyze the Tournament-only architecture.

Topology per image:

    5 agents independent assessments
         ↓
    Candidate pool assembly (4-6 countries drawn from agents)
         ↓
    Parallel hypothesis evaluation (country-only; no region hypotheses)
         ↓
    Bracket seeding (top 4 of the pool, seeded 0..3)
         ↓
    Semi-1 (seed 0 vs seed 3)  ->  winner
    Semi-2 (seed 1 vs seed 2)  ->  winner
    Final  (semi-1 winner vs semi-2 winner)  ->  champion == final answer

Compared to v12, the tournament-only approach:
  * has NO region gate (progressive_narrowing.path == "TOURNAMENT_ONLY",
    no confirmed_region, no runner-up),
  * has NO per-agent country reassessment (country_assessments is empty),
  * has NO road-side driving filter,
  * always runs the full 4-country bracket (no walkovers, no final-only),
  * judge just returns the tournament champion verbatim.

So the natural evaluation gates are much shorter:
    initial plurality -> pool -> bracket (top-4) -> seed 0 -> champion.

The single most interesting question is: without the region gate,
does the tournament-only approach still recover GT via the pool + HE
mechanism, or does it collapse to "top-seed always wins" and expose
whatever ranking the pool-builder produces?

Usage:
    python3 -m vlm_council.analyze_rounds_tournament results_tournament_500/
    python3 -m vlm_council.analyze_rounds_tournament \\
        results_tournament_500/ georc_locations.csv
"""

from __future__ import annotations

import json
import math
import textwrap
from collections import Counter
from pathlib import Path

from vlm_council.evaluate import (
    _NAME_TO_CODE,
    _countries_match,
    _extract_coordinates,
    _extract_country,
    _load_ground_truth,
    _normalize_country,
)


AGENT_NAMES = ["linguistic", "landscape", "botanics", "regulatory", "meta"]

# Stance -> numeric score. `low` and `speculative` are "not enough evidence"
# buckets that we treat as neutral. `medium` / `high` occasionally leak in
# from the initial-assessment vocabulary - same treatment.
STANCE_SCORE = {
    "strongly_support": 2,
    "support": 1,
    "neutral": 0,
    "low": 0,
    "speculative": 0,
    "medium": 0,
    "high": 0,
    "contradicts": -1,
    "strongly_contradicts": -2,
}
STANCE_ORDER = [
    "strongly_support", "support", "neutral", "low", "speculative",
    "contradicts", "strongly_contradicts",
]
STANCE_SHORT = {
    "strongly_support": "s_supp",
    "support": "supp",
    "neutral": "neut",
    "low": "low",
    "speculative": "spec",
    "contradicts": "contra",
    "strongly_contradicts": "s_contra",
}

KM_PER_DEG = 111.0


# ── Data loading ─────────────────────────────────────────────────────────


def _load_results(results_dir: Path) -> list[dict]:
    out: list[dict] = []
    for img_dir in sorted(results_dir.iterdir()):
        rf = img_dir / "result.json"
        if not rf.exists():
            continue
        try:
            with open(rf) as f:
                data = json.load(f)
            if data.get("error"):
                continue
            data["_name"] = img_dir.name
            out.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return out


# ── Small extractors ─────────────────────────────────────────────────────


def _top_country_of(assessment: dict) -> str | None:
    cands = assessment.get("candidates", [])
    if not cands or not isinstance(cands[0], dict):
        return None
    c = cands[0].get("country", "").strip().rstrip(".")
    return c or None


def _top_confidence_of(assessment: dict) -> str | None:
    cands = assessment.get("candidates", [])
    if not cands or not isinstance(cands[0], dict):
        return None
    return cands[0].get("confidence", "").strip().lower() or None


def _initial_pick(r: dict, agent: str) -> str | None:
    return _top_country_of(r.get("assessments", {}).get(agent, {}) or {})


def _initial_confidence(r: dict, agent: str) -> str | None:
    return _top_confidence_of(r.get("assessments", {}).get(agent, {}) or {})


def _all_agent_candidates(r: dict, agent: str) -> list[str]:
    """Full ranked list of countries an agent proposed initially."""
    cands = (r.get("assessments", {}).get(agent, {}) or {}).get("candidates") or []
    out = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        name = (c.get("country") or "").strip().rstrip(".")
        if name:
            out.append(name)
    return out


def _matches(country: str | None, gt_code: str) -> bool:
    if not country:
        return False
    return _countries_match(country, gt_code)


def _country_to_code(country: str | None) -> str | None:
    if not country:
        return None
    return _NAME_TO_CODE.get(_normalize_country(country))


def _same_country(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    ca, cb = _country_to_code(a), _country_to_code(b)
    if ca and cb:
        return ca == cb
    return _normalize_country(a) == _normalize_country(b)


def _plurality(picks: dict[str, str | None]) -> tuple[str | None, int, int]:
    votes: Counter = Counter()
    for c in picks.values():
        if c:
            votes[c.lower()] += 1
    if not votes:
        return None, 0, 0
    top, cnt = votes.most_common(1)[0]
    return top, cnt, sum(votes.values())


def _initial_plurality(r: dict) -> tuple[str | None, int, int]:
    return _plurality({a: _initial_pick(r, a) for a in AGENT_NAMES})


def _candidate_pool(r: dict) -> list[str]:
    return list(r.get("candidate_pool") or [])


def _tournament(r: dict) -> list[dict]:
    return list(r.get("tournament_log") or [])


def _bracket_countries(r: dict) -> list[str]:
    """The 4 countries that actually appear in the bracket (deduped),
    ordered by their tournament seed (0..3)."""
    tl = _tournament(r)
    seed_to_country: dict[int, str] = {}
    for m in tl:
        for side in ("a", "b"):
            c = m.get(f"country_{side}")
            rk = m.get(f"pool_rank_{side}")
            if c is None or rk is None:
                continue
            seed_to_country.setdefault(rk, c)
    return [seed_to_country[k] for k in sorted(seed_to_country)]


def _tournament_champion(r: dict) -> str | None:
    tl = _tournament(r)
    if not tl:
        return None
    for m in tl:
        if m.get("round_label") == "final":
            return m.get("winner")
    return tl[-1].get("winner")


def _semi_winners(r: dict) -> list[str]:
    return [m.get("winner") for m in _tournament(r)
            if (m.get("round_label") or "").startswith("semi")
            and m.get("winner")]


def _final_country(r: dict) -> str | None:
    return _extract_country(r.get("country_result", "") or "")


# ── Formatting ───────────────────────────────────────────────────────────


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def _hsep() -> None:
    print("-" * 74)


def _section(title: str) -> None:
    _hsep()
    print(f"  {title}")
    _hsep()


def _blurb(text: str) -> None:
    for para in text.strip().split("\n\n"):
        print(textwrap.fill(
            " ".join(para.split()), width=70,
            initial_indent="  ", subsequent_indent="  ",
        ))
    print()


# ── Sections (no GT) ─────────────────────────────────────────────────────


def _overview(results: list[dict]) -> None:
    n = len(results)
    print("=" * 74)
    print("  VLM Council - Tournament-only Analysis")
    print("=" * 74)
    print(f"  Total images: {n}")
    print()

    _section("PIPELINE OVERVIEW")
    _blurb(
        "The tournament-only approach strips out PN region gating, "
        "per-agent country reassessment, and road-side filtering - "
        "leaving just initial assessments -> candidate pool -> country "
        "hypothesis eval -> 4-seed bracket -> champion. The bracket is "
        "always a full 3-match tree (semi-1, semi-2, final). We check "
        "here that the runs conform to that shape, and how much pool-"
        "vs-bracket slack there is (a pool of 5-6 candidates means the "
        "seeding step dropped one or two countries before bracketing)."
    )
    n_tourn_only = sum(1 for r in results
                       if (r.get("progressive_narrowing", {}) or {}).get("path") == "TOURNAMENT_ONLY")
    print(f"  Runs marked as TOURNAMENT_ONLY path: {n_tourn_only}/{n} "
          f"({n_tourn_only / n * 100:.1f}%)")

    shapes = Counter()
    for r in results:
        labels = tuple(m.get("round_label") for m in _tournament(r))
        shapes[labels] += 1
    print(f"  Bracket shapes:")
    for k, v in shapes.most_common():
        label = " -> ".join(k) if k else "(empty)"
        print(f"    {label:<40} {v:>4}  ({v / n * 100:.1f}%)")
    print()

    pool_sizes = [len(_candidate_pool(r)) for r in results]
    bracket_sizes = [len(_bracket_countries(r)) for r in results]
    if pool_sizes:
        print(f"  Candidate-pool size:  min {min(pool_sizes)}, "
              f"max {max(pool_sizes)}, mean {_mean(pool_sizes):.2f}, "
              f"median {_median(pool_sizes):.0f}")
    if bracket_sizes:
        print(f"  Bracket size:         min {min(bracket_sizes)}, "
              f"max {max(bracket_sizes)}, mean {_mean(bracket_sizes):.2f}")
    slack = [ps - bs for ps, bs in zip(pool_sizes, bracket_sizes)]
    if slack:
        print(f"  Pool ↔ bracket slack (pool-size - bracket-size): "
              f"mean {_mean(slack):.2f}, max {max(slack)}")
    print()

    n_he = [len(r.get("hypothesis_evaluations") or []) for r in results]
    if n_he:
        print(f"  Hypothesis evaluations per image: "
              f"mean {_mean(n_he):.1f}, median {_median(n_he):.0f}, "
              f"max {max(n_he)}")
    print()


def _pool_vs_bracket(results: list[dict]) -> None:
    _section("POOL -> BRACKET SEEDING (which pool candidates make the tree)")
    _blurb(
        "Given a pool of 4-6 countries, only 4 enter the bracket. This "
        "section characterises what the seeding step drops. "
        "'Bracket == pool[:4]' asks whether the bracket is simply the "
        "first four of the pool array (i.e. the pool is already a "
        "ranked list). If not, the seeding uses information beyond the "
        "pool order - most likely the aggregated hypothesis-eval "
        "scores. The seed-0 country identity table shows which agent's "
        "top pick tends to become the top seed."
    )
    n = len(results)
    same_as_top4 = 0
    n_pool_beyond4 = 0
    for r in results:
        pool = _candidate_pool(r)
        bracket = _bracket_countries(r)
        if len(pool) > 4:
            n_pool_beyond4 += 1
        if len(bracket) == 4 and len(pool) >= 4:
            top4 = [c.lower() for c in pool[:4]]
            b4 = [c.lower() for c in bracket]
            if sorted(top4) == sorted(b4):
                same_as_top4 += 1
    print(f"  Images where the bracket == pool[:4] as a set: "
          f"{same_as_top4}/{n} ({same_as_top4 / n * 100:.1f}%)")
    print(f"  Images with pool larger than 4 (some pool entries did NOT "
          f"make the bracket): {n_pool_beyond4}/{n} "
          f"({n_pool_beyond4 / n * 100:.1f}%)")
    print()

    # Seed-0 agent-origin: which agent (by initial top pick) most often
    # matches the eventual seed-0 country.
    seed0_from_agent: Counter = Counter()
    seed0_from_plurality = 0
    seed0_total = 0
    for r in results:
        bracket = _bracket_countries(r)
        if not bracket:
            continue
        seed0 = bracket[0]
        seed0_total += 1
        for a in AGENT_NAMES:
            if _same_country(_initial_pick(r, a), seed0):
                seed0_from_agent[a] += 1
        ti, ci, _ = _initial_plurality(r)
        if ti and ci >= 2 and _same_country(ti, seed0):
            seed0_from_plurality += 1
    if seed0_total:
        print(f"  Seed-0 country origin (across {seed0_total} images):")
        print(f"    matches the initial plurality (≥2/5): "
              f"{seed0_from_plurality}/{seed0_total} "
              f"({seed0_from_plurality / seed0_total * 100:.1f}%)")
        print(f"    matches an individual agent's initial top pick:")
        for a in AGENT_NAMES:
            c = seed0_from_agent[a]
            print(f"      {a:<12} {c:>4}/{seed0_total} "
                  f"({c / seed0_total * 100:>5.1f}%)")
    print()


def _hypothesis_stance_distribution(results: list[dict]) -> None:
    _section("HYPOTHESIS STANCE DISTRIBUTION (per agent, country hyps only)")
    _blurb(
        "Every hypothesis evaluated in this pipeline is a country "
        "hypothesis (there are no region hypotheses because there is "
        "no region gate). This table is the agent's behavioural "
        "signature: assertive agents pile up support/strongly_support, "
        "sceptical agents produce many contradicts/strongly_contradicts, "
        "narrow-signal agents sit on neutral/low/speculative. Compare "
        "this shape against the v12 stance distribution to see how "
        "removing the region step changes agent commitment patterns."
    )
    per_agent: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}
    for r in results:
        for h in r.get("hypothesis_evaluations") or []:
            a = h.get("agent_name")
            if a not in per_agent:
                continue
            per_agent[a][(h.get("confidence") or "").lower()] += 1

    header_cells = " ".join(f"{STANCE_SHORT[s]:>13}" for s in STANCE_ORDER)
    print(f"  {'Agent':<12} {'Total':>6}  {header_cells}")
    print(f"  {'─' * 12} {'─' * 6}  "
          + " ".join("─" * 13 for _ in STANCE_ORDER))
    for a in AGENT_NAMES:
        c = per_agent[a]
        tot = sum(c.values())
        cells = " ".join(
            f"{c.get(s, 0):>6} ({c.get(s, 0) / tot * 100 if tot else 0:>4.1f}%)"
            for s in STANCE_ORDER
        )
        print(f"  {a:<12} {tot:>6}  {cells}")
    print()
    print("  Stance scoring: strongly_support=+2 support=+1 neutral/low/spec=0 "
          "contradicts=-1 strongly_contradicts=-2")
    print()


def _tournament_dynamics(results: list[dict]) -> None:
    _section("TOURNAMENT DYNAMICS (matches, upsets, judge overrides)")
    _blurb(
        "Match-level behaviour of the 4-seed bracket. `Agreement` = "
        "the specialists defending each candidate and the judge all "
        "reached the same verdict. `Disagreement` = judge overruled a "
        "specialist. `Upset` = a lower-seeded country beat a higher "
        "seeded one - i.e. the tournament revised the seeding. In this "
        "pipeline the tournament IS the whole answer, so upsets are "
        "where the mechanism actually does work beyond the pool builder; "
        "later GT sections check whether the upsets are informative."
    )
    matches = 0
    agree = disagree = 0
    upsets = 0
    per_round_upsets: Counter = Counter()
    per_round_disagree: Counter = Counter()
    per_round_total: Counter = Counter()
    finals_played = 0
    winner_seed_hist: Counter = Counter()
    for r in results:
        for m in _tournament(r):
            matches += 1
            rl = m.get("round_label") or "?"
            per_round_total[rl] += 1
            ag = m.get("agreement")
            if ag == "agree":
                agree += 1
            elif ag == "disagree":
                disagree += 1
                per_round_disagree[rl] += 1
            ra, rb = m.get("pool_rank_a"), m.get("pool_rank_b")
            w = m.get("winner")
            ca = m.get("country_a")
            if ra is None or rb is None or not w or not ca:
                continue
            w_seed = ra if _same_country(w, ca) else rb
            l_seed = rb if _same_country(w, ca) else ra
            if w_seed > l_seed:
                upsets += 1
                per_round_upsets[rl] += 1
            if rl == "final":
                finals_played += 1
                winner_seed_hist[w_seed] += 1

    print(f"  Total matches: {matches}")
    if matches:
        print(f"    Agreement (judge == specialists): {agree}/{matches} "
              f"({agree / matches * 100:.1f}%)")
        print(f"    Disagreement (judge overruled):   {disagree}/{matches} "
              f"({disagree / matches * 100:.1f}%)")
        print(f"    Upsets (lower seed won):          {upsets}/{matches} "
              f"({upsets / matches * 100:.1f}%)")
        print()
        print(f"  {'Round':<10} {'N':>5} {'Upsets':>7} {'Upset %':>8} "
              f"{'Disagree':>9} {'Disagree %':>11}")
        print(f"  {'─' * 10} {'─' * 5} {'─' * 7} {'─' * 8} "
              f"{'─' * 9} {'─' * 11}")
        for rl in ("semi-1", "semi-2", "final"):
            t = per_round_total.get(rl, 0)
            u = per_round_upsets.get(rl, 0)
            d = per_round_disagree.get(rl, 0)
            if t:
                print(f"  {rl:<10} {t:>5} {u:>7} {u / t * 100:>7.1f}% "
                      f"{d:>9} {d / t * 100:>10.1f}%")
    print()
    if finals_played:
        print(f"  Finals played: {finals_played}")
        print(f"    Champion by original seed:")
        for k in sorted(winner_seed_hist):
            v = winner_seed_hist[k]
            print(f"      seed {k}: {v:>4}  ({v / finals_played * 100:>5.1f}%)")
    print()


def _initial_to_champion_shift(results: list[dict]) -> None:
    _section("INITIAL PLURALITY -> CHAMPION (does the tournament ratify or revise?)")
    _blurb(
        "Compares each image's initial plurality (≥2 agents naming the "
        "same country up front) with the tournament champion. A high "
        "match rate means the tournament mostly rubber-stamps the pre-"
        "tournament consensus; a lower one means the bracket + HE "
        "actively revise the initial answer. Whether revisions are "
        "productive can only be told with GT (see later sections)."
    )
    n = len(results)
    same = diff = no_init = 0
    for r in results:
        ti, ci, _ = _initial_plurality(r)
        champ = _tournament_champion(r)
        if not ti or ci < 2:
            no_init += 1
            continue
        if _same_country(ti, champ):
            same += 1
        else:
            diff += 1
    if same + diff:
        print(f"  Images with initial plurality (≥2/5): {same + diff}/{n}")
        print(f"    Champion == initial plurality: {same}/{same + diff} "
              f"({same / (same + diff) * 100:.1f}%)")
        print(f"    Champion  ≠ initial plurality: {diff}/{same + diff} "
              f"({diff / (same + diff) * 100:.1f}%)")
    print(f"    No initial plurality:          {no_init}/{n}")
    print()


def _timing(results: list[dict]) -> None:
    _section("TIMING")
    _blurb(
        "Wall-clock cost per image. Since every image runs the full 4-"
        "seed bracket, timing variance mostly reflects agent latency "
        "and hypothesis-eval volume, not pipeline shape (as it did in "
        "v12 where walkovers were much cheaper)."
    )
    ts = [r.get("timing", {}).get("total_seconds") for r in results]
    ts = [t for t in ts if t is not None]
    if not ts:
        print("  (no timing data)")
        print()
        return
    print(f"  Overall: avg {_mean(ts):.1f}s, median {_median(ts):.1f}s, "
          f"min {min(ts):.1f}s, max {max(ts):.1f}s")
    print(f"  Total compute: {sum(ts):.0f}s ({sum(ts) / 60:.1f} min)")
    print()


# ── GT-based sections ────────────────────────────────────────────────────


def _pipeline_accuracy_ladder(results: list[dict], gt: dict) -> None:
    _section("PIPELINE ACCURACY LADDER (each gate, on GT)")
    _blurb(
        "The tournament-only pipeline has just four gates that can "
        "drop GT: pool inclusion, bracket seeding (top-4 of pool), "
        "reaching the final, and winning the final. Once a gate drops "
        "GT, no downstream stage can recover it. Compare the survival "
        "rates side-by-side to see which gate is the primary "
        "bottleneck - for tournament-only this is almost always the "
        "pool builder, since there's no region gate above it."
    )
    n = 0
    initial_plur_correct = 0
    initial_top_any_agent = 0    # GT was any agent's top pick
    gt_in_pool = 0
    gt_in_pool_top3 = 0
    gt_at_pool_top = 0
    gt_in_bracket = 0
    gt_at_seed0 = 0
    gt_in_bracket_top2 = 0       # GT seeded 0 or 1
    gt_reached_final = 0
    gt_won_final = 0
    final_correct = 0

    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        n += 1
        gt_code = truth["country_code"]

        if any(_matches(_initial_pick(r, a), gt_code) for a in AGENT_NAMES):
            initial_top_any_agent += 1
        ti, ci, _ = _initial_plurality(r)
        if ti and ci >= 2 and _matches(ti, gt_code):
            initial_plur_correct += 1

        pool = _candidate_pool(r)
        if any(_matches(c, gt_code) for c in pool):
            gt_in_pool += 1
        if any(_matches(c, gt_code) for c in pool[:3]):
            gt_in_pool_top3 += 1
        if pool and _matches(pool[0], gt_code):
            gt_at_pool_top += 1

        bracket = _bracket_countries(r)
        if any(_matches(c, gt_code) for c in bracket):
            gt_in_bracket += 1
        if len(bracket) >= 1 and _matches(bracket[0], gt_code):
            gt_at_seed0 += 1
        if len(bracket) >= 2 and (_matches(bracket[0], gt_code)
                                  or _matches(bracket[1], gt_code)):
            gt_in_bracket_top2 += 1

        tl = _tournament(r)
        if tl:
            final_match = tl[-1]
            if (_matches(final_match.get("country_a"), gt_code)
                    or _matches(final_match.get("country_b"), gt_code)):
                gt_reached_final += 1
            if _matches(final_match.get("winner"), gt_code):
                gt_won_final += 1

        fin = _final_country(r)
        if fin and _matches(fin, gt_code):
            final_correct += 1

    if n == 0:
        print("  (no GT overlap)")
        print()
        return

    def row(label, x):
        print(f"  {label:<48} {x:>4}/{n}  ({x / n * 100:>5.1f}%)")

    print(f"  Images with GT: {n}")
    print()
    row("GT was some agent's initial top pick", initial_top_any_agent)
    row("Initial plurality (≥2/5) on GT", initial_plur_correct)
    row("GT ∈ candidate_pool (any rank)", gt_in_pool)
    row("GT ∈ candidate_pool top-3", gt_in_pool_top3)
    row("GT == pool[0]", gt_at_pool_top)
    row("GT ∈ bracket (top-4 seeded)", gt_in_bracket)
    row("GT ∈ bracket top-2 seed (seed 0 or 1)", gt_in_bracket_top2)
    row("GT == seed 0 (bracket top seed)", gt_at_seed0)
    row("GT reached tournament final", gt_reached_final)
    row("GT won tournament (== champion)", gt_won_final)
    row("Final judge country == GT", final_correct)
    print()

    def conv(from_x, to_x, from_label, to_label):
        if from_x == 0:
            print(f"    {from_label} -> {to_label}: n/a")
            return
        print(f"    {from_label} -> {to_label}: "
              f"{to_x}/{from_x} ({to_x / from_x * 100:.1f}%)")
    print("  Gate-to-gate survival of GT:")
    conv(gt_in_pool, gt_in_bracket, "in pool", "in bracket")
    conv(gt_in_bracket, gt_reached_final, "in bracket", "reached final")
    conv(gt_reached_final, gt_won_final, "reached final", "won tournament")
    conv(gt_won_final, final_correct, "won tournament", "final judge = GT")
    print()


def _seed_vs_gt_analysis(results: list[dict], gt: dict) -> None:
    _section("SEED FIDELITY - does the seeding align with truth?")
    _blurb(
        "For every image where GT made it into the bracket, at which "
        "seed did it land? A well-calibrated seeding puts GT at seed 0, "
        "where it plays only two matches to win. The seed-to-outcome "
        "table shows how often GT wins by seed: if seed 3 GT still wins "
        "half its images, the tournament corrects mis-seedings; if "
        "seed 3 GT almost always loses, the seeding basically decides "
        "the answer."
    )
    seed_of_gt: Counter = Counter()
    seed_of_gt_won: Counter = Counter()
    n_with_gt = 0
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        gt_code = truth["country_code"]
        bracket = _bracket_countries(r)
        gt_seed = None
        for i, c in enumerate(bracket):
            if _matches(c, gt_code):
                gt_seed = i
                break
        if gt_seed is None:
            continue
        n_with_gt += 1
        seed_of_gt[gt_seed] += 1
        champ = _tournament_champion(r)
        if _matches(champ, gt_code):
            seed_of_gt_won[gt_seed] += 1

    if n_with_gt == 0:
        print("  (no GT ever made it into a bracket)")
        print()
        return
    print(f"  Images where GT is in the bracket: {n_with_gt}")
    print()
    print(f"  {'GT seed':>8} {'N':>5} {'Won':>5} {'Win rate':>10}")
    print(f"  {'─' * 8} {'─' * 5} {'─' * 5} {'─' * 10}")
    for k in (0, 1, 2, 3):
        v = seed_of_gt[k]
        w = seed_of_gt_won[k]
        if v == 0:
            continue
        print(f"  {k:>8} {v:>5} {w:>5} {w / v * 100:>9.1f}%")
    total_won = sum(seed_of_gt_won.values())
    print()
    print(f"  Overall: when GT is in the bracket, "
          f"tournament picks it {total_won}/{n_with_gt} "
          f"({total_won / n_with_gt * 100:.1f}%) of the time.")
    print()


def _hypothesis_calibration(results: list[dict], gt: dict) -> None:
    _section("HYPOTHESIS-STANCE CALIBRATION (GT country hyp vs the rest)")
    _blurb(
        "For each image we sum stance scores per country hypothesis "
        "and rank them. If HE is well-calibrated, the GT country "
        "hypothesis ranks #1 with a positive score. Because tournament-"
        "only has no region hypotheses, this table is a pure country-"
        "level probe. Compare 'ranked #1' (a strict criterion) with "
        "'score > 0' (a floor: at least net-supported)."
    )
    n_total = 0
    n_gt_hyp = 0
    gt_scores = []
    gt_ranks = []
    top1 = top3 = pos = neg = 0
    per_agent_gt: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}

    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        n_total += 1
        gt_code = truth["country_code"]
        gt_name = truth["country_name"]

        # Aggregate stance score per hypothesis
        agg: dict[str, int] = {}
        raw_by_hyp: dict[str, list] = {}
        for h in r.get("hypothesis_evaluations") or []:
            hid = h.get("hypothesis_id")
            if not hid or not hid.startswith("country_"):
                continue
            s = STANCE_SCORE.get((h.get("confidence") or "").lower(), 0)
            agg[hid] = agg.get(hid, 0) + s
            raw_by_hyp.setdefault(hid, []).append(h)
        if not agg:
            continue

        # Find GT hypothesis
        gt_hid = None
        for hid in agg:
            country_name = hid[len("country_"):].replace("_", " ")
            if _same_country(country_name, gt_name):
                gt_hid = hid
                break
        if not gt_hid:
            continue
        n_gt_hyp += 1
        score = agg[gt_hid]
        gt_scores.append(score)
        ranked = sorted(agg.items(), key=lambda kv: -kv[1])
        rank = next(i for i, (h, _) in enumerate(ranked, 1) if h == gt_hid)
        gt_ranks.append(rank)
        if rank == 1:
            top1 += 1
        if rank <= 3:
            top3 += 1
        if score > 0:
            pos += 1
        elif score < 0:
            neg += 1
        for h in raw_by_hyp.get(gt_hid, []):
            a = h.get("agent_name")
            if a in per_agent_gt:
                per_agent_gt[a][(h.get("confidence") or "").lower()] += 1

    print(f"  Images with GT: {n_total}")
    if n_gt_hyp == 0:
        print("  (GT country was never evaluated as a hypothesis)")
        print()
        return
    print(f"  Images where GT country was evaluated as a hypothesis: "
          f"{n_gt_hyp}/{n_total} ({n_gt_hyp / n_total * 100:.1f}%)")
    print(f"    Ranked #1 by aggregate stance:   {top1}/{n_gt_hyp} "
          f"({top1 / n_gt_hyp * 100:.1f}%)")
    print(f"    Ranked in top 3:                 {top3}/{n_gt_hyp} "
          f"({top3 / n_gt_hyp * 100:.1f}%)")
    print(f"    Aggregate score > 0 (net supported):    {pos}/{n_gt_hyp} "
          f"({pos / n_gt_hyp * 100:.1f}%)")
    print(f"    Aggregate score < 0 (net contradicted): {neg}/{n_gt_hyp} "
          f"({neg / n_gt_hyp * 100:.1f}%)")
    print(f"    Mean rank of GT hyp:            {_mean(gt_ranks):.2f} "
          f"(median {_median(gt_ranks):.0f})")
    print(f"    Mean aggregate score of GT hyp: {_mean(gt_scores):+.2f}")
    print()
    print("  Per-agent stance on the GT country hypothesis:")
    print(f"    {'Agent':<12} {'N':>5} "
          + " ".join(f"{STANCE_SHORT[s]:>9}" for s in STANCE_ORDER))
    print(f"    {'─' * 12} {'─' * 5} " + " ".join("─" * 9 for _ in STANCE_ORDER))
    for a in AGENT_NAMES:
        c = per_agent_gt[a]
        tot = sum(c.values())
        if tot == 0:
            continue
        cells = " ".join(f"{c.get(s, 0):>9}" for s in STANCE_ORDER)
        print(f"    {a:<12} {tot:>5} {cells}")
    print()


def _match_level_gt(results: list[dict], gt: dict) -> None:
    _section("MATCH-LEVEL OUTCOMES vs GT")
    _blurb(
        "For each 1v1 match: was the winning country GT, the losing "
        "country GT, or neither? 'Toward truth' = winner was GT; "
        "'Away from truth' = GT was in the match but lost - a wasted "
        "chance for the pipeline; 'Both wrong' = GT never entered the "
        "match at all (a pool-level failure). Decisive-match accuracy "
        "isolates only matches where GT was on exactly one side."
    )
    matches = toward = away = both_wrong = 0
    upsets_total = upsets_toward = upsets_away = 0
    per_round: dict[str, dict] = {}
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        gt_code = truth["country_code"]
        for m in _tournament(r):
            matches += 1
            rl = m.get("round_label") or "?"
            per_round.setdefault(rl, {"n": 0, "toward": 0, "away": 0,
                                     "both_wrong": 0})
            per_round[rl]["n"] += 1
            ca, cb = m.get("country_a"), m.get("country_b")
            w = m.get("winner")
            ok_a = _matches(ca, gt_code)
            ok_b = _matches(cb, gt_code)
            ok_w = _matches(w, gt_code)
            if not ok_a and not ok_b:
                both_wrong += 1
                per_round[rl]["both_wrong"] += 1
            elif ok_w:
                toward += 1
                per_round[rl]["toward"] += 1
            else:
                away += 1
                per_round[rl]["away"] += 1

            ra, rb = m.get("pool_rank_a"), m.get("pool_rank_b")
            if ra is None or rb is None or not w or not ca:
                continue
            w_seed = ra if _same_country(w, ca) else rb
            l_seed = rb if _same_country(w, ca) else ra
            if w_seed > l_seed:
                upsets_total += 1
                if ok_w:
                    upsets_toward += 1
                elif ok_a or ok_b:
                    upsets_away += 1

    if matches == 0:
        print("  (no matches with GT)")
        print()
        return
    print(f"  Total matches (GT-anchored images): {matches}")
    print(f"    Toward truth (winner == GT):          {toward}/{matches} "
          f"({toward / matches * 100:.1f}%)")
    print(f"    Away from truth (GT was in match, lost): {away}/{matches} "
          f"({away / matches * 100:.1f}%)")
    print(f"    Both wrong (GT not in match):         {both_wrong}/{matches} "
          f"({both_wrong / matches * 100:.1f}%)")
    decisive = toward + away
    if decisive:
        print(f"    Decisive matches (GT on exactly one side): "
              f"{toward}/{decisive} ({toward / decisive * 100:.1f}%) "
              f"toward truth")
    print()
    print(f"  {'Round':<10} {'N':>5} {'Toward':>7} {'Away':>6} {'Both wrong':>11}")
    print(f"  {'─' * 10} {'─' * 5} {'─' * 7} {'─' * 6} {'─' * 11}")
    for rl in ("semi-1", "semi-2", "final"):
        d = per_round.get(rl)
        if not d or d["n"] == 0:
            continue
        n_r = d["n"]
        print(f"  {rl:<10} {n_r:>5} "
              f"{d['toward']:>7} ({d['toward'] / n_r * 100:>4.1f}%) "
              f"{d['away']:>4} ({d['away'] / n_r * 100:>4.1f}%) "
              f"{d['both_wrong']:>6} ({d['both_wrong'] / n_r * 100:>4.1f}%)")
    print()
    print(f"  Bracket upsets (lower seed won): {upsets_total}")
    if upsets_total:
        print(f"    Toward GT: {upsets_toward}/{upsets_total} "
              f"({upsets_toward / upsets_total * 100:.1f}%)")
        print(f"    Away from GT: {upsets_away}/{upsets_total} "
              f"({upsets_away / upsets_total * 100:.1f}%)")
        print(f"    (remainder = neither side was GT)")
    print()


def _elimination_analysis(results: list[dict], gt: dict) -> None:
    _section("HOW GT LOSES - semi vs final elimination + top eliminators")
    _blurb(
        "When GT is in the bracket but doesn't win, at which round is "
        "it eliminated, and by which country? A large tail of the same "
        "eliminator suggests a systematic confusion (e.g. Costa Rica "
        "beats Taiwan repeatedly). Semi-eliminations mean GT lost "
        "early; final-eliminations mean GT reached the last match but "
        "was outvoted."
    )
    semi_elims: Counter = Counter()   # eliminator country name
    final_elims: Counter = Counter()
    n_gt_in_bracket = n_gt_lost = 0
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        gt_code = truth["country_code"]
        bracket = _bracket_countries(r)
        if not any(_matches(c, gt_code) for c in bracket):
            continue
        n_gt_in_bracket += 1
        champ = _tournament_champion(r)
        if _matches(champ, gt_code):
            continue
        n_gt_lost += 1
        for m in _tournament(r):
            ca, cb = m.get("country_a"), m.get("country_b")
            w = m.get("winner")
            rl = m.get("round_label") or ""
            if (_matches(ca, gt_code) and not _matches(w, gt_code)):
                (final_elims if rl == "final" else semi_elims)[w] += 1
            elif (_matches(cb, gt_code) and not _matches(w, gt_code)):
                (final_elims if rl == "final" else semi_elims)[w] += 1
    if n_gt_in_bracket == 0:
        print("  (no GT ever reached the bracket)")
        print()
        return
    print(f"  GT in bracket: {n_gt_in_bracket}    Eliminated: {n_gt_lost}")
    print(f"    Lost in a semi-final: {sum(semi_elims.values())}")
    print(f"    Lost in the final:    {sum(final_elims.values())}")
    print()
    if semi_elims:
        print(f"  Top eliminators in the semis:")
        for k, v in semi_elims.most_common(8):
            print(f"    {k:<32} {v:>4}")
        print()
    if final_elims:
        print(f"  Top eliminators in the final:")
        for k, v in final_elims.most_common(8):
            print(f"    {k:<32} {v:>4}")
        print()


def _per_agent_initial_accuracy(results: list[dict], gt: dict) -> None:
    _section("PER-AGENT INITIAL PICK ACCURACY (only pre-tournament signal)")
    _blurb(
        "In tournament-only there is no per-agent country reassessment, "
        "so an agent's contribution is just its initial top pick. "
        "'Init acc' = share of images where the agent's top pick is GT. "
        "'Also in bracket' = share of images where the agent's top pick "
        "made it into the bracket at all - a measure of how much the "
        "pool-builder listens to this agent. Compare the two: an agent "
        "who is often correct but rarely bracketed is being underused."
    )
    print(f"  {'Agent':<12} {'N':>5} {'Init acc':>12} "
          f"{'Top ∈ bracket':>15} {'Top ∈ pool':>12}")
    print(f"  {'─' * 12} {'─' * 5} {'─' * 12} {'─' * 15} {'─' * 12}")
    for agent in AGENT_NAMES:
        n = ok = in_bracket = in_pool = 0
        for r in results:
            truth = gt.get(r["_name"])
            if not truth:
                continue
            pick = _initial_pick(r, agent)
            if not pick:
                continue
            n += 1
            if _matches(pick, truth["country_code"]):
                ok += 1
            bracket = _bracket_countries(r)
            pool = _candidate_pool(r)
            if any(_same_country(pick, c) for c in bracket):
                in_bracket += 1
            if any(_same_country(pick, c) for c in pool):
                in_pool += 1
        if n == 0:
            print(f"  {agent:<12} {0:>5}  (no data)")
            continue
        print(f"  {agent:<12} {n:>5} {ok}/{n} ({ok / n * 100:>4.1f}%) "
              f"{in_bracket}/{n} ({in_bracket / n * 100:>4.1f}%) "
              f"{in_pool}/{n} ({in_pool / n * 100:>4.1f}%)")
    print()


def _bracket_diversity(results: list[dict], gt: dict) -> None:
    _section("BRACKET DIVERSITY (how far apart are the 4 seeded countries?)")
    _blurb(
        "A well-designed bracket contains alternatives that cover "
        "different plausible answers. We approximate this via the "
        "number of distinct continents/macro-regions among the 4 "
        "seeded countries. A bracket of 4 European countries is "
        "'locally focused' - if it wins, PN-style narrowing was "
        "essentially implicit; if it loses, GT was pushed out very "
        "early. Cross-continental brackets hedge but signal higher "
        "uncertainty about the region."
    )
    # Use the same PN region grouping as the v12 analyzer.
    PN_REGIONS = {
        "Europe": {"al","ad","at","ba","be","bg","by","ch","cy","cz","de","dk","ee","es","fi","fo","fr","gb","uk","gl","gr","hr","hu","ie","is","it","lt","lu","lv","md","me","mk","mt","nl","no","pl","pm","pt","ro","rs","ru","se","si","sk","ua","va","xk"},
        "North America": {"ca","us","mx","pr","vi","gp","mq","aw","cw"},
        "Central America & Caribbean": {"bz","cr","cu","do","gt","hn","ht","jm","ni","pa","sv","tt","ck"},
        "South America": {"ar","bo","br","cl","co","ec","gy","pe","py","sr","uy","ve"},
        "Middle East": {"ae","bh","il","ir","iq","jo","kw","lb","om","ps","qa","sa","sy","tr","ye"},
        "North Africa": {"dz","eg","ly","ma","sd","tn"},
        "Sub-Saharan Africa": {"ao","bf","bi","bj","bw","cd","cf","cg","ci","cm","dj","er","et","ga","gh","gm","gn","gw","ke","lr","ls","mg","ml","mr","mu","mw","mz","na","ne","ng","rw","sl","sn","so","ss","sz","td","tg","tz","ug","yt","re","za","zm","zw"},
        "Central Asia": {"kg","kz","tj","tm","uz","af"},
        "South Asia": {"bd","bt","in","lk","mv","np","pk"},
        "East Asia": {"cn","hk","jp","kr","mn","mo","tw"},
        "Southeast Asia": {"bn","id","kh","la","mm","my","ph","sg","th","tl","vn"},
        "Oceania": {"as","au","fj","gu","mp","nc","nz","pf","pg","to","ws"},
    }
    def region_of(name):
        code = _country_to_code(name)
        if not code:
            return None
        for r, s in PN_REGIONS.items():
            if code in s:
                return r
        return None

    n_regions_hist: Counter = Counter()
    correct_by_diversity: dict[int, list[bool]] = {}
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        bracket = _bracket_countries(r)
        regions = {region_of(c) for c in bracket if region_of(c)}
        k = len(regions)
        n_regions_hist[k] += 1
        champ = _tournament_champion(r)
        correct = bool(champ and _matches(champ, truth["country_code"]))
        correct_by_diversity.setdefault(k, []).append(correct)

    n = sum(n_regions_hist.values())
    if n == 0:
        print("  (no data)")
        print()
        return
    print(f"  {'#Regions in bracket':<22} {'N':>5} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'─' * 22} {'─' * 5} {'─' * 8} {'─' * 10}")
    for k in sorted(n_regions_hist):
        xs = correct_by_diversity.get(k, [])
        c = sum(xs)
        print(f"  {k:<22} {len(xs):>5} {c:>8} {c / len(xs) * 100 if xs else 0:>9.1f}%")
    print()
    print("  A bracket of 1 region = fully-committed (all 4 seeds share a region).")
    print("  A bracket of 4 regions = maximally diversified.")
    print()


def _geographic_bias(results: list[dict], gt: dict) -> None:
    _section("GEOGRAPHIC BIAS (final prediction vs GT)")
    _blurb(
        "Systematic direction of error for the final coordinate. Mean "
        "N-S / W-E offsets in km show whether the pipeline is biased "
        "toward a particular hemisphere. A gap between mean and median "
        "reveals long-tail bias - a few extreme wrong-continent picks "
        "can drag the average without moving the median."
    )
    ns_km, we_km = [], []
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        pred = _extract_coordinates(r.get("country_result", ""))
        if not pred:
            continue
        gt_la, gt_lo = truth["lat"], truth["lng"]
        dlo = pred[1] - gt_lo
        if dlo > 180:
            dlo -= 360
        elif dlo < -180:
            dlo += 360
        ns_km.append((pred[0] - gt_la) * KM_PER_DEG)
        we_km.append(dlo * KM_PER_DEG * math.cos(math.radians(gt_la)))
    if not ns_km:
        print("  (no parsed final coordinates)")
        print()
        return
    print(f"  Images with parsed final coordinates: {len(ns_km)}")
    print(f"  Mean N-S offset: {_mean(ns_km):+.0f} km   "
          f"(median {_median(ns_km):+.0f} km)")
    print(f"    predictions tend {'NORTH' if _mean(ns_km) >= 0 else 'SOUTH'} of GT")
    print(f"  Mean W-E offset: {_mean(we_km):+.0f} km   "
          f"(median {_median(we_km):+.0f} km)")
    print(f"    predictions tend {'EAST' if _mean(we_km) >= 0 else 'WEST'} of GT")
    print()
    print("  Sign convention: +Δlat = north of GT; +Δlng = east of GT.")
    print("  W-E km uses cos(lat_GT).")
    print()


# ── Structured dynamics metrics (machine-readable) ─────────────────────────


def compute_dynamics(results_dir: Path, gt_path: Path | None, out_dir: Path) -> dict:
    """Compute structured tournament dynamics metrics and write dynamics_metrics.json.

    This is the machine-readable counterpart to the printed tournament-only
    analysis, consumed by the ``report`` step so the single per-approach report
    can carry the candidate-pool-to-bracket-seeding dynamics.

    Region gating was removed from this approach, so there is deliberately no
    region / Path-A-B / funnel-S1 content here: the metrics cover the candidate
    pool, the pool to bracket seeding step, the bracket matches, upsets and the
    initial plurality to champion shift, plus the GT view of seed fidelity and
    the gate to gate survival ladder (pool to bracket to final to champion).
    """
    results = _load_results(Path(results_dir))
    gt = _load_ground_truth(Path(gt_path)) if gt_path else {}
    n = len(results)

    # ── Pool and bracket shape ─────────────────────────────────────────────
    pool_sizes = [len(_candidate_pool(r)) for r in results]
    bracket_sizes = [len(_bracket_countries(r)) for r in results]
    slack = [ps - bs for ps, bs in zip(pool_sizes, bracket_sizes)]

    same_as_top4 = 0
    n_pool_beyond4 = 0
    for r in results:
        pool = _candidate_pool(r)
        bracket = _bracket_countries(r)
        if len(pool) > 4:
            n_pool_beyond4 += 1
        if len(bracket) == 4 and len(pool) >= 4:
            if sorted(c.lower() for c in pool[:4]) == sorted(c.lower() for c in bracket):
                same_as_top4 += 1

    # Seed-0 country origin (which agent's initial top pick becomes seed 0)
    seed0_from_agent: Counter = Counter()
    seed0_from_plurality = 0
    seed0_total = 0
    for r in results:
        bracket = _bracket_countries(r)
        if not bracket:
            continue
        seed0 = bracket[0]
        seed0_total += 1
        for a in AGENT_NAMES:
            if _same_country(_initial_pick(r, a), seed0):
                seed0_from_agent[a] += 1
        ti, ci, _ = _initial_plurality(r)
        if ti and ci >= 2 and _same_country(ti, seed0):
            seed0_from_plurality += 1

    # ── Tournament dynamics (match-level, agreement, upsets) ───────────────
    matches = agree = disagree = upsets = 0
    finals_played = 0
    winner_seed_hist: Counter = Counter()
    for r in results:
        for m in _tournament(r):
            matches += 1
            rl = m.get("round_label") or "?"
            ag = m.get("agreement")
            if ag == "agree":
                agree += 1
            elif ag == "disagree":
                disagree += 1
            ra, rb = m.get("pool_rank_a"), m.get("pool_rank_b")
            w = m.get("winner")
            ca = m.get("country_a")
            if ra is None or rb is None or not w or not ca:
                continue
            w_seed = ra if _same_country(w, ca) else rb
            l_seed = rb if _same_country(w, ca) else ra
            if w_seed > l_seed:
                upsets += 1
            if rl == "final":
                finals_played += 1
                winner_seed_hist[w_seed] += 1

    # ── Initial plurality vs champion (ratify or revise) ───────────────────
    plur_same = plur_diff = plur_none = 0
    for r in results:
        ti, ci, _ = _initial_plurality(r)
        champ = _tournament_champion(r)
        if not ti or ci < 2:
            plur_none += 1
            continue
        if _same_country(ti, champ):
            plur_same += 1
        else:
            plur_diff += 1

    # ── GT views: seed fidelity + gate-to-gate survival ────────────────────
    seed_of_gt: Counter = Counter()
    seed_of_gt_won: Counter = Counter()
    n_gt_in_bracket_seed = 0

    gt_n = 0
    gt_in_pool = 0
    gt_in_bracket = 0
    gt_at_seed0 = 0
    gt_reached_final = 0
    gt_won_final = 0

    for r in results:
        truth = gt.get(r.get("_name"))
        if not truth:
            continue
        gt_n += 1
        gt_code = truth["country_code"]

        pool = _candidate_pool(r)
        if any(_matches(c, gt_code) for c in pool):
            gt_in_pool += 1

        bracket = _bracket_countries(r)
        if any(_matches(c, gt_code) for c in bracket):
            gt_in_bracket += 1
        if len(bracket) >= 1 and _matches(bracket[0], gt_code):
            gt_at_seed0 += 1

        gt_seed = None
        for i, c in enumerate(bracket):
            if _matches(c, gt_code):
                gt_seed = i
                break
        if gt_seed is not None:
            n_gt_in_bracket_seed += 1
            seed_of_gt[gt_seed] += 1
            if _matches(_tournament_champion(r), gt_code):
                seed_of_gt_won[gt_seed] += 1

        tl = _tournament(r)
        if tl:
            final_match = tl[-1]
            if (_matches(final_match.get("country_a"), gt_code)
                    or _matches(final_match.get("country_b"), gt_code)):
                gt_reached_final += 1
            if _matches(final_match.get("winner"), gt_code):
                gt_won_final += 1

    def _rate(num: int, den: int) -> float | None:
        return (num / den) if den else None

    seed_fidelity = {}
    for k in (0, 1, 2, 3):
        v = seed_of_gt.get(k, 0)
        if not v:
            continue
        seed_fidelity[str(k)] = {
            "n": v,
            "won": seed_of_gt_won.get(k, 0),
            "win_rate": _rate(seed_of_gt_won.get(k, 0), v),
        }

    metrics = {
        "n_total": n,
        "pool_bracket": {
            "pool_size_mean": _mean(pool_sizes) if pool_sizes else None,
            "pool_size_median": _median(pool_sizes) if pool_sizes else None,
            "pool_size_max": max(pool_sizes) if pool_sizes else None,
            "bracket_size_mean": _mean(bracket_sizes) if bracket_sizes else None,
            "slack_mean": _mean(slack) if slack else None,
            "slack_max": max(slack) if slack else None,
            "n_bracket_equals_pool_top4": same_as_top4,
            "bracket_equals_pool_top4_rate": _rate(same_as_top4, n),
            "n_pool_beyond_4": n_pool_beyond4,
            "pool_beyond_4_rate": _rate(n_pool_beyond4, n),
        },
        "seed0_origin": {
            "n": seed0_total,
            "matches_initial_plurality": seed0_from_plurality,
            "matches_initial_plurality_rate": _rate(seed0_from_plurality, seed0_total),
            "matches_agent_top_pick": {
                a: {
                    "n": seed0_from_agent.get(a, 0),
                    "rate": _rate(seed0_from_agent.get(a, 0), seed0_total),
                }
                for a in AGENT_NAMES
            },
        },
        "tournament": {
            "n_matches": matches,
            "agree": agree,
            "agree_rate": _rate(agree, matches),
            "disagree": disagree,
            "disagree_rate": _rate(disagree, matches),
            "upsets": upsets,
            "upset_rate": _rate(upsets, matches),
            "finals_played": finals_played,
            "champion_by_seed": {
                str(k): {
                    "n": winner_seed_hist.get(k, 0),
                    "rate": _rate(winner_seed_hist.get(k, 0), finals_played),
                }
                for k in sorted(winner_seed_hist)
            },
        },
        "initial_plurality_vs_champion": {
            "n_with_plurality": plur_same + plur_diff,
            "champion_equals_plurality": plur_same,
            "champion_equals_plurality_rate": _rate(plur_same, plur_same + plur_diff),
            "champion_differs": plur_diff,
            "champion_differs_rate": _rate(plur_diff, plur_same + plur_diff),
            "n_no_plurality": plur_none,
        },
        "gt_ladder": {
            "n_with_gt": gt_n,
            "gt_in_pool": gt_in_pool,
            "gt_in_pool_rate": _rate(gt_in_pool, gt_n),
            "gt_in_bracket": gt_in_bracket,
            "gt_in_bracket_rate": _rate(gt_in_bracket, gt_n),
            "gt_at_seed0": gt_at_seed0,
            "gt_at_seed0_rate": _rate(gt_at_seed0, gt_n),
            "gt_reached_final": gt_reached_final,
            "gt_reached_final_rate": _rate(gt_reached_final, gt_n),
            "gt_won_final": gt_won_final,
            "gt_won_final_rate": _rate(gt_won_final, gt_n),
            "survival": {
                "pool_to_bracket": _rate(gt_in_bracket, gt_in_pool),
                "bracket_to_final": _rate(gt_reached_final, gt_in_bracket),
                "final_to_champion": _rate(gt_won_final, gt_reached_final),
            },
        },
        "seed_fidelity": {
            "n_gt_in_bracket": n_gt_in_bracket_seed,
            "by_seed": seed_fidelity,
            "won_when_in_bracket": sum(seed_of_gt_won.values()),
            "won_when_in_bracket_rate": _rate(
                sum(seed_of_gt_won.values()), n_gt_in_bracket_seed
            ),
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dynamics_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[dynamics] wrote {out_file}")
    print(f"[dynamics] n={n} matches={matches} upsets={upsets} "
          f"GT-in-pool={gt_in_pool}/{gt_n} GT-won={gt_won_final}/{gt_n}")
    return metrics


# ── Driver ───────────────────────────────────────────────────────────────


def analyze(results_dir: Path, gt_path: Path | None = None) -> None:
    results = _load_results(results_dir)
    if not results:
        print("No results found.")
        return

    _overview(results)
    _pool_vs_bracket(results)
    _hypothesis_stance_distribution(results)
    _tournament_dynamics(results)
    _initial_to_champion_shift(results)
    _timing(results)

    if gt_path:
        gt = _load_ground_truth(gt_path)
        print("=" * 74)
        print("  GROUND-TRUTH-BASED TOURNAMENT-ONLY ANALYSIS")
        print("=" * 74)
        print()
        _pipeline_accuracy_ladder(results, gt)
        _seed_vs_gt_analysis(results, gt)
        _hypothesis_calibration(results, gt)
        _match_level_gt(results, gt)
        _elimination_analysis(results, gt)
        _per_agent_initial_accuracy(results, gt)
        _bracket_diversity(results, gt)
        _geographic_bias(results, gt)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze the tournament-only architecture"
    )
    parser.add_argument("results_dir", help="Directory with result.json files")
    parser.add_argument("ground_truth", nargs="?", default=None,
                        help="Optional path to georc_locations.csv")
    args = parser.parse_args()
    gt_path = Path(args.ground_truth) if args.ground_truth else None
    analyze(Path(args.results_dir), gt_path)


if __name__ == "__main__":
    main()
