"""Analyze the v12 pipeline: Progressive Narrowing + Parallel Hypothesis
Evaluation + Country Tournament.

Topology per image:

    5 agents independent assessments
         ↓
    Region proposal (each agent proposes ≤N regions)
         ↓
    Region vote  ->  confirmed_region  (+ runner_up_region)
         ↓
    Parallel hypothesis evaluation
        (region_<X> and country_<Y> stances by a subset of agents,
         values: strongly_support | support | neutral |
                 contradicts | strongly_contradicts | low)
         ↓
    Per-agent country re-assessment inside surviving region(s)
         ↓
    Candidate pool (ranked list of countries)
         ↓
    Country tournament (1v1 bracket: semi-1, semi-2, final,
                        or degenerate: semi+final / final-only / walkover)
         ↓
    Judge final country + coordinates

Metrics are tailored to THIS pipeline. In particular:
- there is no "R1/R2" the way hub-and-spoke has,
- the natural evaluation gates are: initial plurality -> PN region ->
  candidate pool -> tournament champion -> final,
- questioning-quality is replaced by *hypothesis stance calibration*
  and *tournament ordering* (does the bracket respect the pool ranking,
  and when it upsets it, is the upset toward GT?).

Usage:
    python -m vlm_council.analyze_rounds_v12 results_v12_pn_500/
    python -m vlm_council.analyze_rounds_v12 \\
        results_v12_pn_500/ georc_locations.csv
"""

from __future__ import annotations

import json
import math
import re
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
CONF_ORDER = {"high": 3, "medium": 2, "low": 1, "speculative": 0}
# Stance -> numeric score for hypothesis evaluation.
# `low` is the "no verdict" bucket the judge emits when the agent lacked
# signal; treated as 0, same as neutral.
STANCE_SCORE = {
    "strongly_support": 2,
    "support": 1,
    "neutral": 0,
    "low": 0,
    "contradicts": -1,
    "strongly_contradicts": -2,
}
STANCE_ORDER = [
    "strongly_support", "support", "neutral", "low",
    "contradicts", "strongly_contradicts",
]
KM_PER_DEG = 111.0


# ── ISO-code -> PN region map (same one the pipeline exposes) ─────────────

PN_REGIONS = {
    "Europe": {
        "al", "ad", "at", "ba", "be", "bg", "by", "ch", "cy", "cz", "de",
        "dk", "ee", "es", "fi", "fo", "fr", "gb", "uk", "gl", "gr", "hr",
        "hu", "ie", "is", "it", "lt", "lu", "lv", "md", "me", "mk", "mt",
        "nl", "no", "pl", "pm", "pt", "ro", "rs", "ru", "se", "si", "sk",
        "ua", "va", "xk",
    },
    "North America": {"ca", "us", "mx", "pr", "vi", "gp", "mq", "aw", "cw"},
    "Central America & Caribbean": {
        "bz", "cr", "cu", "do", "gt", "hn", "ht", "jm", "ni", "pa", "sv",
        "tt", "ck",
    },
    "South America": {
        "ar", "bo", "br", "cl", "co", "ec", "gy", "pe", "py", "sr", "uy", "ve",
    },
    "Middle East": {
        "ae", "bh", "il", "ir", "iq", "jo", "kw", "lb", "om", "ps", "qa",
        "sa", "sy", "tr", "ye",
    },
    "North Africa": {"dz", "eg", "ly", "ma", "sd", "tn"},
    "Sub-Saharan Africa": {
        "ao", "bf", "bi", "bj", "bw", "cd", "cf", "cg", "ci", "cm", "dj",
        "er", "et", "ga", "gh", "gm", "gn", "gw", "ke", "lr", "ls", "mg",
        "ml", "mr", "mu", "mw", "mz", "na", "ne", "ng", "rw", "sl", "sn",
        "so", "ss", "sz", "td", "tg", "tz", "ug", "yt", "re", "za", "zm",
        "zw",
    },
    "Central Asia": {"kg", "kz", "tj", "tm", "uz", "af"},
    "South Asia": {"bd", "bt", "in", "lk", "mv", "np", "pk"},
    "East Asia": {"cn", "hk", "jp", "kr", "mn", "mo", "tw"},
    "Southeast Asia": {"bn", "id", "kh", "la", "mm", "my", "ph", "sg", "th",
                       "tl", "vn"},
    "Oceania": {"as", "au", "fj", "gu", "mp", "nc", "nz", "pf", "pg", "to",
                "ws"},
}

_REGION_KEYS = {re.sub(r"[^a-z]+", "", r.lower()): r for r in PN_REGIONS}


def _canon_region(name: str | None) -> str | None:
    if not name:
        return None
    k = re.sub(r"[^a-z]+", "", name.lower())
    return _REGION_KEYS.get(k)


def _gt_region(gt_code: str) -> str | None:
    for region, codes in PN_REGIONS.items():
        if gt_code in codes:
            return region
    return None


def _country_to_code(country: str | None) -> str | None:
    if not country:
        return None
    return _NAME_TO_CODE.get(_normalize_country(country))


def _country_in_region(country: str | None, region: str | None) -> bool:
    if not country or not region:
        return False
    code = _country_to_code(country)
    if not code:
        return False
    return code in PN_REGIONS.get(region, set())


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


def _ca_pick(r: dict, agent: str) -> str | None:
    return _top_country_of(r.get("country_assessments", {}).get(agent, {}) or {})


def _final_agent_pick(r: dict, agent: str) -> str | None:
    """Final per-agent position: country_assessment (after region confirmed)
    if it exists, else initial assessment."""
    c = _ca_pick(r, agent)
    if c:
        return c
    return _initial_pick(r, agent)


def _matches(country: str | None, gt_code: str) -> bool:
    if not country:
        return False
    return _countries_match(country, gt_code)


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


def _ca_plurality(r: dict) -> tuple[str | None, int, int]:
    return _plurality({a: _ca_pick(r, a) for a in AGENT_NAMES})


def _confirmed_region(r: dict) -> str | None:
    return _canon_region((r.get("progressive_narrowing", {}) or {}).get("confirmed_region"))


def _runner_up_region(r: dict) -> str | None:
    return _canon_region((r.get("progressive_narrowing", {}) or {}).get("runner_up_region"))


def _surviving_regions(r: dict) -> list[str]:
    raw = (r.get("progressive_narrowing", {}) or {}).get("surviving_regions") or []
    out = []
    for x in raw:
        c = _canon_region(x)
        if c and c not in out:
            out.append(c)
    return out


def _proposed_regions(r: dict) -> list[str]:
    raw = (r.get("progressive_narrowing", {}) or {}).get("proposed_regions") or []
    out = []
    for x in raw:
        c = _canon_region(x)
        if c and c not in out:
            out.append(c)
    return out


def _candidate_pool(r: dict) -> list[str]:
    return list(r.get("candidate_pool") or [])


def _tournament(r: dict) -> list[dict]:
    return list(r.get("tournament_log") or [])


def _tournament_shape(r: dict) -> str:
    tl = _tournament(r)
    labels = tuple(x.get("round_label") for x in tl)
    if labels == ("semi-1", "semi-2", "final"):
        return "full-bracket"
    if labels == ("semi", "final"):
        return "3-way (semi+final)"
    if labels == ("final",):
        return "final-only"
    if labels == ():
        return "walkover"
    return "other"


def _tournament_champion(r: dict) -> str | None:
    tl = _tournament(r)
    if tl:
        final = tl[-1]
        w = final.get("winner")
        return w or None
    # Walkover: single-country pool
    pool = _candidate_pool(r)
    if len(pool) == 1:
        return pool[0]
    return None


def _final_country(r: dict) -> str | None:
    return _extract_country(r.get("country_result", "") or "")


# ── Formatting helpers ───────────────────────────────────────────────────


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
    """Print a short indented explanation right under a section header,
    wrapping paragraphs at ~70 chars. Multiple paragraphs separated by
    blank lines."""
    import textwrap
    for para in text.strip().split("\n\n"):
        wrapped = textwrap.fill(
            " ".join(para.split()), width=70,
            initial_indent="  ", subsequent_indent="  ",
        )
        print(wrapped)
    print()


def _pct(a: int, b: int) -> str:
    return f"{a / b * 100:.1f}%" if b else "  n/a"


# ── Sections (no GT) ─────────────────────────────────────────────────────


def _overview(results: list[dict]) -> None:
    n = len(results)
    print("=" * 74)
    print("  VLM Council - v12 (PN + Parallel Hypothesis + Tournament) Analysis")
    print("=" * 74)
    print(f"  Total images: {n}")
    print()

    shapes = Counter(_tournament_shape(r) for r in results)
    pool_sizes = [len(_candidate_pool(r)) for r in results]
    n_hyp = [len(r.get("hypothesis_evaluations") or []) for r in results]
    consensus = sum(1 for r in results
                    if (r.get("progressive_narrowing", {}) or {}).get("region_consensus"))

    _section("PIPELINE OVERVIEW")
    _blurb(
        "High-level shape of the v12 pipeline on this dataset. "
        "`Region-consensus` = agents agreed on one region at the PN vote "
        "(Path A, direct to judge); the rest run the full stack "
        "(Path B: hypothesis eval -> country reassessment -> tournament). "
        "The tournament shape follows from the candidate-pool size: "
        "≥4 -> full bracket, 3 -> semi+final, 2 -> final-only, 1 -> walkover. "
        "A larger share of degenerate shapes means PN eliminated most "
        "alternatives before the tournament even started."
    )
    print(f"  Region-consensus at PN gate (Path A):        "
          f"{consensus}/{n} ({consensus / n * 100:.1f}%)")
    print(f"  Region contested (Path B, runs full stack):  "
          f"{n - consensus}/{n} ({(n - consensus) / n * 100:.1f}%)")
    print()
    print("  Tournament-shape distribution:")
    for k in ("full-bracket", "3-way (semi+final)", "final-only", "walkover", "other"):
        c = shapes.get(k, 0)
        if c == 0 and k == "other":
            continue
        print(f"    {k:<22} {c:>4}  ({c / n * 100:>5.1f}%)")
    print()
    if pool_sizes:
        print(f"  Candidate-pool size:   min {min(pool_sizes)}, "
              f"max {max(pool_sizes)}, mean {_mean(pool_sizes):.2f}, "
              f"median {_median(pool_sizes):.0f}")
    if n_hyp:
        print(f"  Hypothesis evaluations per image: "
              f"mean {_mean(n_hyp):.1f}, median {_median(n_hyp):.0f}, "
              f"max {max(n_hyp)}")
    print()


def _region_proposal_behavior(results: list[dict]) -> None:
    _section("REGION-PROPOSAL BEHAVIOR (PN stage)")
    _blurb(
        "Three funnel stages for every region: how often it was proposed "
        "by at least one agent, how often it survived the PN vote "
        "(top-2 by aggregate score), and how often it was finally "
        "confirmed. A high proposed-count with a low confirmed-count "
        "means the region is a common 'maybe' that rarely wins the vote "
        "(over-guessed). A high confirmed-count relative to proposed "
        "means the region, once floated, tends to stick - a signal that "
        "its evidence is decisive."
    )
    print()
    proposed_hist = Counter()
    surv_hist = Counter()
    confirmed_hist = Counter()
    surv_size = []
    prop_size = []
    for r in results:
        prop_size.append(len(_proposed_regions(r)))
        surv_size.append(len(_surviving_regions(r)))
        for rg in _proposed_regions(r):
            proposed_hist[rg] += 1
        for rg in _surviving_regions(r):
            surv_hist[rg] += 1
        cr = _confirmed_region(r)
        if cr:
            confirmed_hist[cr] += 1
    n = len(results)
    print(f"  Proposed regions per image:   mean {_mean(prop_size):.2f}, "
          f"median {_median(prop_size):.0f}, "
          f"max {max(prop_size) if prop_size else 0}")
    print(f"  Surviving regions per image:  mean {_mean(surv_size):.2f}, "
          f"median {_median(surv_size):.0f}, "
          f"max {max(surv_size) if surv_size else 0}")
    print()
    print(f"  {'Region':<32} {'Proposed':>10} {'Survived':>10} {'Confirmed':>11}")
    print(f"  {'─' * 32} {'─' * 10} {'─' * 10} {'─' * 11}")
    regions = sorted(
        set(proposed_hist) | set(surv_hist) | set(confirmed_hist),
        key=lambda k: (-confirmed_hist.get(k, 0), -proposed_hist.get(k, 0)),
    )
    for rg in regions:
        p = proposed_hist.get(rg, 0)
        s = surv_hist.get(rg, 0)
        c = confirmed_hist.get(rg, 0)
        print(f"  {rg:<32} {p:>10} {s:>10} {c:>11}")
    print()
    print(f"  Proposed  = images where at least one agent proposed this region")
    print(f"  Survived  = region survived the PN vote (top-2 by aggregate score)")
    print(f"  Confirmed = region chosen as `confirmed_region` for this image")
    print()


def _hypothesis_stance_distribution(results: list[dict]) -> None:
    _section("HYPOTHESIS STANCE DISTRIBUTION (per agent, region + country hyps)")
    _blurb(
        "How each agent votes on the hypotheses put in front of it - a "
        "behavioral signature. Agents that stack support/strongly_support "
        "are 'assertive' (they commit even under moderate signal); agents "
        "with a lot of contradicts/strongly_contradicts are 'sceptical' "
        "(they use hard rules to kill candidates); agents with mostly "
        "neutral/low have narrow expertise and stay silent unless they "
        "see their signal. This is orthogonal to accuracy - an assertive "
        "wrong agent and a neutral silent agent both fail to help, but "
        "for opposite reasons."
    )
    print()
    by_agent_stance: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}
    by_agent_type: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}
    for r in results:
        for h in r.get("hypothesis_evaluations") or []:
            a = h.get("agent_name")
            if a not in by_agent_stance:
                continue
            s = (h.get("confidence") or "").lower()
            by_agent_stance[a][s] += 1
            hid = h.get("hypothesis_id", "") or ""
            if hid.startswith("region_"):
                by_agent_type[a]["region"] += 1
            elif hid.startswith("country_"):
                by_agent_type[a]["country"] += 1
            else:
                by_agent_type[a]["other"] += 1

    # Short labels so the header lines up with the cell width.
    short = {
        "strongly_support": "s_supp",
        "support": "supp",
        "neutral": "neut",
        "low": "low",
        "contradicts": "contra",
        "strongly_contradicts": "s_contra",
    }
    hdr = f"  {'Agent':<12} {'Total':>6} {'Region':>7} {'Country':>8} "
    hdr += " ".join(f"{short[s]:>13}" for s in STANCE_ORDER)
    print(hdr)
    print(f"  {'─' * 12} {'─' * 6} {'─' * 7} {'─' * 8} "
          + " ".join("─" * 13 for _ in STANCE_ORDER))
    for a in AGENT_NAMES:
        total = sum(by_agent_stance[a].values())
        rg = by_agent_type[a].get("region", 0)
        co = by_agent_type[a].get("country", 0)
        cells = " ".join(
            f"{by_agent_stance[a].get(s, 0):>6} ({by_agent_stance[a].get(s, 0) / total * 100 if total else 0:>4.1f}%)"
            for s in STANCE_ORDER
        )
        print(f"  {a:<12} {total:>6} {rg:>7} {co:>8} {cells}")
    print()
    print("  Stance meaning (score):")
    print("    strongly_support=+2  support=+1  neutral=0  low=0  "
          "contradicts=-1  strongly_contradicts=-2")
    print("  `low` is the pipeline's 'not enough evidence' bucket "
          "(treated as 0 in scoring).")
    print()


def _candidate_pool_composition(results: list[dict]) -> None:
    _section("CANDIDATE-POOL COMPOSITION")
    _blurb(
        "The pool is the list of countries that enter the tournament. "
        "Its size sets the bracket shape (previous section). "
        "The regional composition tells us whether the pool respects the "
        "PN region decision: countries inside the confirmed_region are "
        "the primary bet, countries from the runner_up_region are a "
        "hedge, and countries outside both regions are 'leaks' - "
        "candidates that survived despite the PN gate deprioritising "
        "their region."
    )
    print()
    n = len(results)
    pool_sizes = Counter(len(_candidate_pool(r)) for r in results)
    print("  Pool-size distribution:")
    print(f"    {'Size':>4} {'Count':>6} {'Share':>7}")
    print(f"    {'─' * 4} {'─' * 6} {'─' * 7}")
    for k in sorted(pool_sizes):
        print(f"    {k:>4} {pool_sizes[k]:>6} {pool_sizes[k] / n * 100:>6.1f}%")
    print()

    # Fraction of pool that lies in the confirmed region vs runner-up vs outside
    in_conf = in_run = outside = 0
    total_pool = 0
    for r in results:
        cr = _confirmed_region(r)
        ru = _runner_up_region(r)
        for c in _candidate_pool(r):
            total_pool += 1
            if _country_in_region(c, cr):
                in_conf += 1
            elif _country_in_region(c, ru):
                in_run += 1
            else:
                outside += 1
    if total_pool:
        print(f"  Pool composition by region ({total_pool} pool slots):")
        print(f"    Inside confirmed_region:  {in_conf}/{total_pool} "
              f"({in_conf / total_pool * 100:.1f}%)")
        print(f"    Inside runner_up_region:  {in_run}/{total_pool} "
              f"({in_run / total_pool * 100:.1f}%)")
        print(f"    Outside both regions:     {outside}/{total_pool} "
              f"({outside / total_pool * 100:.1f}%)")
    print()


def _tournament_dynamics(results: list[dict]) -> None:
    _section("TOURNAMENT DYNAMICS (bracket ordering, upsets, agreement)")
    _blurb(
        "Behavior of the 1v1 country matches. `Agreement` = the two "
        "spec-agents defending each candidate and the judge all named "
        "the same winner (consensus). `Disagreement` = the judge "
        "overruled at least one specialist. `Upset` = a lower-seeded "
        "candidate (worse pool_rank) beat a higher-seeded one - i.e. "
        "the tournament revised the PN ordering. A high top-seed win "
        "rate + low upset rate means the tournament mostly rubber-stamps "
        "the pool ranking; it adds value only when it upsets, and later "
        "sections check whether those upsets move toward or away from GT."
    )
    print()
    match_count = 0
    agree = disagree = 0
    upsets = 0  # winner pool_rank > loser pool_rank
    top_seed_wins_final = 0
    finals_played = 0
    winner_rank_hist = Counter()

    for r in results:
        tl = _tournament(r)
        for m in tl:
            match_count += 1
            ag = m.get("agreement")
            if ag == "agree":
                agree += 1
            elif ag == "disagree":
                disagree += 1
            ra = m.get("pool_rank_a")
            rb = m.get("pool_rank_b")
            w = m.get("winner")
            if ra is None or rb is None or w is None:
                continue
            w_rank = ra if _same_country(w, m.get("country_a")) else rb
            l_rank = rb if _same_country(w, m.get("country_a")) else ra
            if w_rank > l_rank:  # lower-ranked slot beat higher-ranked
                upsets += 1
            if m.get("round_label") == "final":
                finals_played += 1
                winner_rank_hist[w_rank] += 1
                if w_rank == 0:
                    top_seed_wins_final += 1

    print(f"  Total 1v1 matches:                {match_count}")
    if match_count:
        print(f"    Agent agreement (judge==spec):  {agree}/{match_count} "
              f"({agree / match_count * 100:.1f}%)")
        print(f"    Agent disagreement (judge overruled): {disagree}/{match_count} "
              f"({disagree / match_count * 100:.1f}%)")
        print(f"    Bracket upsets "
              f"(lower-seed beat higher-seed): {upsets}/{match_count} "
              f"({upsets / match_count * 100:.1f}%)")
    print()
    if finals_played:
        print(f"  Finals played:                    {finals_played}")
        print(f"    Champion by original pool rank:")
        for k in sorted(winner_rank_hist):
            v = winner_rank_hist[k]
            print(f"      pool_rank {k}: {v:>4} "
                  f"({v / finals_played * 100:>5.1f}%)")
        print(f"    Top seed (pool_rank 0) wins:    {top_seed_wins_final}/"
              f"{finals_played} ({top_seed_wins_final / finals_played * 100:.1f}%)")
    print()


def _initial_to_champion_shift(results: list[dict]) -> None:
    _section("INITIAL PLURALITY -> TOURNAMENT CHAMPION (agreement without GT)")
    _blurb(
        "Compares the pre-PN plurality (≥2 out of 5 agents naming the "
        "same country up front) with the tournament champion. A high "
        "match rate means the pipeline mostly ratifies what the agents "
        "already agreed on initially - the downstream stages didn't "
        "flip the answer. A low match rate would mean PN + tournament "
        "regularly overturn initial consensus, which is only good if "
        "GT-anchored metrics later show it moved toward truth."
    )
    print()
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
    print(f"  Images with initial plurality (≥2/5 same country): {same + diff}/{n}")
    if same + diff:
        print(f"    Champion == initial plurality: {same}/{same + diff} "
              f"({same / (same + diff) * 100:.1f}%)")
        print(f"    Champion  ≠ initial plurality: {diff}/{same + diff} "
              f"({diff / (same + diff) * 100:.1f}%)")
    print(f"    No initial plurality:          {no_init}/{n}")
    print()


def _timing(results: list[dict]) -> None:
    _section("TIMING")
    _blurb(
        "Wall-clock cost per image, split by bracket shape. Full-bracket "
        "images spend the most time (semi + final matches all run); "
        "walkovers are cheapest because the tournament is skipped. If "
        "later sections show high accuracy on cheap shapes and low "
        "accuracy on full-bracket, that means the pipeline is 'over-"
        "thinking' the hard cases without actually resolving them."
    )
    print()
    ts = [r.get("timing", {}).get("total_seconds") for r in results]
    ts = [t for t in ts if t is not None]
    if not ts:
        print("  (no timing data)")
        print()
        return
    print(f"  Overall: avg {_mean(ts):.1f}s, median {_median(ts):.1f}s, "
          f"min {min(ts):.1f}s, max {max(ts):.1f}s")
    print(f"  Total compute: {sum(ts):.0f}s ({sum(ts) / 60:.1f} min)")

    # Split by tournament shape as a rough cost lens
    for shape in ("full-bracket", "3-way (semi+final)", "final-only", "walkover"):
        xs = [t for r, t in zip(results, ts) if _tournament_shape(r) == shape]
        if xs:
            print(f"    {shape:<22} n={len(xs):>4}  avg {_mean(xs):.1f}s  "
                  f"median {_median(xs):.1f}s")
    print()


# ── GT-based sections ────────────────────────────────────────────────────


def _pipeline_accuracy_ladder(results: list[dict], gt: dict) -> None:
    _section("PIPELINE ACCURACY LADDER (each gate, on GT)")
    _blurb(
        "The core diagnostic: at each pipeline gate, is GT still on the "
        "table? A gate that drops GT can never be recovered downstream, "
        "so the numbers strictly decrease (this is a hard upper bound "
        "on final accuracy at each stage). The gate-to-gate survival "
        "rates identify where the pipeline is bleeding the most: a low "
        "survival rate between two stages means that specific stage is "
        "the bottleneck. Compare the ladder values to see whether the "
        "region gate, the pool construction, or the tournament itself "
        "is the primary source of failure."
    )
    n = 0
    gt_in_proposed = 0
    gt_in_surviving = 0
    gt_is_confirmed = 0
    gt_in_confirmed_or_runner = 0
    gt_in_pool = 0
    gt_in_pool_top3 = 0
    gt_at_pool_top = 0
    gt_reached_final = 0
    gt_won_tournament = 0
    final_correct = 0
    initial_plur_correct = 0

    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        n += 1
        gt_code = truth["country_code"]
        gt_name = truth["country_name"]
        gt_reg = _gt_region(gt_code)

        prop = _proposed_regions(r)
        surv = _surviving_regions(r)
        conf = _confirmed_region(r)
        ru = _runner_up_region(r)
        pool = _candidate_pool(r)
        tl = _tournament(r)
        champ = _tournament_champion(r)
        fin = _final_country(r)
        ti, ci, _ = _initial_plurality(r)

        if gt_reg and gt_reg in prop:
            gt_in_proposed += 1
        if gt_reg and gt_reg in surv:
            gt_in_surviving += 1
        if gt_reg and gt_reg == conf:
            gt_is_confirmed += 1
        if gt_reg and gt_reg in (conf, ru):
            gt_in_confirmed_or_runner += 1
        if any(_matches(c, gt_code) for c in pool):
            gt_in_pool += 1
        if any(_matches(c, gt_code) for c in pool[:3]):
            gt_in_pool_top3 += 1
        if pool and _matches(pool[0], gt_code):
            gt_at_pool_top += 1
        # "Reached final" = appeared as one of the two final competitors
        if tl:
            final_match = tl[-1]
            if (_matches(final_match.get("country_a"), gt_code)
                    or _matches(final_match.get("country_b"), gt_code)):
                gt_reached_final += 1
        if champ and _matches(champ, gt_code):
            gt_won_tournament += 1
        if fin and _matches(fin, gt_code):
            final_correct += 1
        if ti and ci >= 2 and _matches(ti, gt_code):
            initial_plur_correct += 1

    if n == 0:
        print("  (no GT overlap)")
        print()
        return

    def row(label, x):
        print(f"  {label:<48} {x:>4}/{n}  ({x / n * 100:>5.1f}%)")

    print(f"  Images with GT: {n}")
    print()
    row("Initial plurality (≥2/5) on GT", initial_plur_correct)
    row("GT region ∈ proposed_regions", gt_in_proposed)
    row("GT region ∈ surviving_regions", gt_in_surviving)
    row("GT region == confirmed_region", gt_is_confirmed)
    row("GT region ∈ {confirmed, runner_up}", gt_in_confirmed_or_runner)
    row("GT country ∈ candidate_pool (any rank)", gt_in_pool)
    row("GT country ∈ candidate_pool top-3", gt_in_pool_top3)
    row("GT country == pool[0] (top seed)", gt_at_pool_top)
    row("GT country reached tournament FINAL", gt_reached_final)
    row("GT country == tournament champion", gt_won_tournament)
    row("Final judge country == GT", final_correct)
    print()
    # Conversion rates between gates
    def conv(from_x, to_x, from_label, to_label):
        if from_x == 0:
            print(f"    {from_label} -> {to_label}: n/a")
            return
        print(f"    {from_label} -> {to_label}: "
              f"{to_x}/{from_x} ({to_x / from_x * 100:.1f}%)")
    print("  Gate-to-gate survival of GT:")
    conv(gt_in_proposed, gt_is_confirmed, "proposed", "confirmed")
    conv(gt_is_confirmed, gt_in_pool, "confirmed", "in pool")
    conv(gt_in_pool, gt_reached_final, "in pool", "reached final")
    conv(gt_reached_final, gt_won_tournament, "reached final", "won tournament")
    conv(gt_won_tournament, final_correct, "won tournament", "final judge = GT")
    print()


def _region_gate_analysis(results: list[dict], gt: dict) -> None:
    _section("REGION GATE - where the pipeline commits geographically")
    _blurb(
        "Focuses on the single hardest gate: PN region confirmation. "
        "Three failure modes: (1) 'dropped before confirm' - GT region "
        "didn't survive the vote at all (an agent-proposal problem); "
        "(2) 'reached vote but lost' - GT region was in the top-2 but "
        "another region beat it (a vote-scoring problem); (3) 'runner-up "
        "saves' - GT region wasn't confirmed but was retained as runner-"
        "up, which still keeps its countries in the pool. The confusion "
        "list at the bottom shows which regions the pipeline routinely "
        "mixes up (e.g. Middle East -> Europe)."
    )
    print()
    n = 0
    matrix: Counter = Counter()  # (gt_reg, conf_reg) -> count
    dropped_before_conf = 0    # GT reg not in surviving_regions
    dropped_at_conf = 0        # GT reg in surviving_regions but not confirmed
    runner_up_saves = 0        # confirmed != GT reg but runner_up == GT reg
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        gt_reg = _gt_region(truth["country_code"])
        if not gt_reg:
            continue
        n += 1
        conf = _confirmed_region(r)
        surv = _surviving_regions(r)
        ru = _runner_up_region(r)
        matrix[(gt_reg, conf or "∅")] += 1
        if gt_reg not in surv:
            dropped_before_conf += 1
        elif conf != gt_reg:
            dropped_at_conf += 1
        if conf != gt_reg and ru == gt_reg:
            runner_up_saves += 1

    if n == 0:
        print("  (no GT-region overlap)")
        print()
        return
    print(f"  GT-mapped images: {n}")
    print()
    correct_conf = sum(v for (g, c), v in matrix.items() if g == c)
    print(f"  Confirmed region == GT region:                 "
          f"{correct_conf}/{n}  ({correct_conf / n * 100:.1f}%)")
    print(f"  GT region dropped BEFORE confirm (not in surviving): "
          f"{dropped_before_conf}/{n}  "
          f"({dropped_before_conf / n * 100:.1f}%)")
    print(f"  GT region reached vote but lost (in surviving, not confirmed): "
          f"{dropped_at_conf}/{n}  ({dropped_at_conf / n * 100:.1f}%)")
    print(f"  Runner-up 'saves' GT region "
          f"(confirmed≠GT reg but runner_up==GT reg): "
          f"{runner_up_saves}/{n}  ({runner_up_saves / n * 100:.1f}%)")
    print()
    # Top confusions: GT region -> confirmed region (for wrong confirmations)
    wrong = Counter({(g, c): v for (g, c), v in matrix.items() if g != c})
    if wrong:
        print("  Top region confusions (GT region -> confirmed region):")
        for (g, c), v in wrong.most_common(10):
            print(f"    {g:<28} -> {c:<28} {v:>4}")
        print()


def _hypothesis_calibration(results: list[dict], gt: dict) -> None:
    _section("HYPOTHESIS-STANCE CALIBRATION (are stances aligned with truth?)")
    _blurb(
        "Do the agents' hypothesis stances actually track the truth? "
        "For each image we sum stance scores per hypothesis (support=+1, "
        "strongly_support=+2, contradicts=-1, etc.) and rank all "
        "hypotheses. If the pipeline is well-calibrated, the GT country "
        "and GT region should be ranked #1 with a positive aggregate "
        "score. 'Score > 0' = at least net-supported (a plausibility "
        "floor); 'ranked #1' = beat every other hypothesis on this "
        "image (the harder criterion). Per-agent tables show which "
        "agents pull their weight on GT (many s_supp/supp cells) vs "
        "which abstain or oppose it."
    )
    print()

    # Per-agent stance-vs-truth confusion.
    # For each hypothesis_evaluation entry, we tag it as GT-target if the
    # hypothesis id names either the GT country or the GT region.
    n = 0
    n_country_gt_hyp = 0                # images with a country hypothesis on GT
    country_gt_avg_score = []           # aggregate score for GT country hyp
    country_gt_rank = []                # rank (1=best) of GT country hyp
    country_gt_top1 = 0
    country_gt_top3 = 0
    country_gt_positive = 0
    country_gt_negative = 0

    region_gt_avg_score = []
    region_gt_rank = []
    region_gt_top1 = 0
    region_gt_top3 = 0
    region_gt_positive = 0
    region_gt_negative = 0
    n_region_gt_hyp = 0

    # Per-agent: how do agents stance the GT hypothesis?
    per_agent_gt_country = {a: Counter() for a in AGENT_NAMES}
    per_agent_gt_region = {a: Counter() for a in AGENT_NAMES}

    def _agg_by_hyp(evals: list[dict]) -> dict[str, tuple[int, int]]:
        # hypothesis_id -> (aggregate score, evaluator count)
        out: dict[str, list] = {}
        for e in evals:
            hid = e.get("hypothesis_id")
            if not hid:
                continue
            s = STANCE_SCORE.get((e.get("confidence") or "").lower(), 0)
            out.setdefault(hid, [0, 0])
            out[hid][0] += s
            out[hid][1] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        n += 1
        gt_code = truth["country_code"]
        gt_name = truth["country_name"]
        gt_reg = _gt_region(gt_code)

        evals = r.get("hypothesis_evaluations") or []
        agg = _agg_by_hyp(evals)

        # Country hypotheses only
        country_agg = {k: v for k, v in agg.items() if k.startswith("country_")}
        region_agg = {k: v for k, v in agg.items() if k.startswith("region_")}

        # Find GT country hypothesis
        gt_country_hid = None
        for hid in country_agg:
            country_name = hid[len("country_"):].replace("_", " ")
            if _same_country(country_name, gt_name):
                gt_country_hid = hid
                break
        if gt_country_hid:
            n_country_gt_hyp += 1
            score, _cnt = country_agg[gt_country_hid]
            country_gt_avg_score.append(score)
            ranked = sorted(country_agg.items(), key=lambda kv: -kv[1][0])
            rank = next(i for i, (h, _) in enumerate(ranked, 1)
                        if h == gt_country_hid)
            country_gt_rank.append(rank)
            if rank == 1:
                country_gt_top1 += 1
            if rank <= 3:
                country_gt_top3 += 1
            if score > 0:
                country_gt_positive += 1
            elif score < 0:
                country_gt_negative += 1
            # Per-agent tally on this hypothesis
            for e in evals:
                if e.get("hypothesis_id") != gt_country_hid:
                    continue
                a = e.get("agent_name")
                if a in per_agent_gt_country:
                    per_agent_gt_country[a][(e.get("confidence") or "").lower()] += 1

        # Find GT region hypothesis
        gt_region_hid = None
        if gt_reg:
            for hid in region_agg:
                reg_name = hid[len("region_"):].replace("_", " ")
                if _canon_region(reg_name) == gt_reg:
                    gt_region_hid = hid
                    break
        if gt_region_hid:
            n_region_gt_hyp += 1
            score, _cnt = region_agg[gt_region_hid]
            region_gt_avg_score.append(score)
            ranked = sorted(region_agg.items(), key=lambda kv: -kv[1][0])
            rank = next(i for i, (h, _) in enumerate(ranked, 1)
                        if h == gt_region_hid)
            region_gt_rank.append(rank)
            if rank == 1:
                region_gt_top1 += 1
            if rank <= 3:
                region_gt_top3 += 1
            if score > 0:
                region_gt_positive += 1
            elif score < 0:
                region_gt_negative += 1
            for e in evals:
                if e.get("hypothesis_id") != gt_region_hid:
                    continue
                a = e.get("agent_name")
                if a in per_agent_gt_region:
                    per_agent_gt_region[a][(e.get("confidence") or "").lower()] += 1

    print(f"  Images with GT: {n}")
    print()
    print(f"  ── COUNTRY hypotheses ──")
    if n_country_gt_hyp:
        print(f"  Images where GT country was actually evaluated as a hypothesis: "
              f"{n_country_gt_hyp}/{n} "
              f"({n_country_gt_hyp / n * 100:.1f}%)")
        print(f"    GT country hyp ranked #1 by aggregate stance: "
              f"{country_gt_top1}/{n_country_gt_hyp} "
              f"({country_gt_top1 / n_country_gt_hyp * 100:.1f}%)")
        print(f"    GT country hyp ranked in top 3: "
              f"{country_gt_top3}/{n_country_gt_hyp} "
              f"({country_gt_top3 / n_country_gt_hyp * 100:.1f}%)")
        print(f"    GT country hyp aggregate score > 0 (net supported): "
              f"{country_gt_positive}/{n_country_gt_hyp} "
              f"({country_gt_positive / n_country_gt_hyp * 100:.1f}%)")
        print(f"    GT country hyp aggregate score < 0 (net contradicted): "
              f"{country_gt_negative}/{n_country_gt_hyp} "
              f"({country_gt_negative / n_country_gt_hyp * 100:.1f}%)")
        print(f"    Mean rank of GT country hyp: "
              f"{_mean(country_gt_rank):.2f} "
              f"(median {_median(country_gt_rank):.0f})")
        print(f"    Mean aggregate score of GT country hyp: "
              f"{_mean(country_gt_avg_score):+.2f}")
    else:
        print("  (GT country was never evaluated as a country hypothesis)")
    print()

    print(f"  ── REGION hypotheses ──")
    if n_region_gt_hyp:
        print(f"  Images where GT region was evaluated as a hypothesis: "
              f"{n_region_gt_hyp}/{n} "
              f"({n_region_gt_hyp / n * 100:.1f}%)")
        print(f"    GT region hyp ranked #1 by aggregate stance: "
              f"{region_gt_top1}/{n_region_gt_hyp} "
              f"({region_gt_top1 / n_region_gt_hyp * 100:.1f}%)")
        print(f"    GT region hyp ranked in top 3: "
              f"{region_gt_top3}/{n_region_gt_hyp} "
              f"({region_gt_top3 / n_region_gt_hyp * 100:.1f}%)")
        print(f"    GT region hyp aggregate score > 0: "
              f"{region_gt_positive}/{n_region_gt_hyp} "
              f"({region_gt_positive / n_region_gt_hyp * 100:.1f}%)")
        print(f"    GT region hyp aggregate score < 0: "
              f"{region_gt_negative}/{n_region_gt_hyp} "
              f"({region_gt_negative / n_region_gt_hyp * 100:.1f}%)")
        print(f"    Mean rank of GT region hyp: "
              f"{_mean(region_gt_rank):.2f} "
              f"(median {_median(region_gt_rank):.0f})")
        print(f"    Mean aggregate score of GT region hyp: "
              f"{_mean(region_gt_avg_score):+.2f}")
    else:
        print("  (GT region was never evaluated as a region hypothesis)")
    print()

    short = {
        "strongly_support": "s_supp", "support": "supp", "neutral": "neut",
        "low": "low", "contradicts": "contra", "strongly_contradicts": "s_contra",
    }
    # Per-agent stance on the GT country hypothesis
    print("  Per-agent stance on the GT COUNTRY hypothesis "
          "(across images where they voted on it):")
    print(f"    {'Agent':<12} {'N':>5} "
          + " ".join(f"{short[s]:>9}" for s in STANCE_ORDER))
    print(f"    {'─' * 12} {'─' * 5} " + " ".join("─" * 9 for _ in STANCE_ORDER))
    for a in AGENT_NAMES:
        c = per_agent_gt_country[a]
        tot = sum(c.values())
        if tot == 0:
            continue
        cells = " ".join(f"{c.get(s, 0):>9}" for s in STANCE_ORDER)
        print(f"    {a:<12} {tot:>5} {cells}")
    print()
    print("  Per-agent stance on the GT REGION hypothesis:")
    print(f"    {'Agent':<12} {'N':>5} "
          + " ".join(f"{short[s]:>9}" for s in STANCE_ORDER))
    print(f"    {'─' * 12} {'─' * 5} " + " ".join("─" * 9 for _ in STANCE_ORDER))
    for a in AGENT_NAMES:
        c = per_agent_gt_region[a]
        tot = sum(c.values())
        if tot == 0:
            continue
        cells = " ".join(f"{c.get(s, 0):>9}" for s in STANCE_ORDER)
        print(f"    {a:<12} {tot:>5} {cells}")
    print()


def _tournament_gt_analysis(results: list[dict], gt: dict) -> None:
    _section("TOURNAMENT vs GT - do upsets and finals move toward truth?")
    _blurb(
        "Given the tournament matches, what fraction of decisions were "
        "correct? 'Pool recall @k' = fraction of images where GT is in "
        "the top-k of the pool (a ceiling - the tournament cannot pick "
        "GT if it isn't in the pool). 'Toward truth' = the winner was "
        "GT; 'Away from truth' = GT was in the match but lost; 'Both "
        "wrong' = GT wasn't in the match at all (a pool-level failure, "
        "not a match-level one). 'Decisive matches' isolates only those "
        "where GT was on exactly one side - the only cases the "
        "tournament can actually get right or wrong. The upset "
        "breakdown checks whether the tournament's revisions of the "
        "pool ordering actually help."
    )
    print()
    n = 0
    matches = 0
    match_toward_truth = 0     # winner correct, loser wrong
    match_away_from_truth = 0  # winner wrong, loser correct
    match_neither = 0
    match_both_wrong = 0       # both wrong, but distinct
    upsets_toward_truth = 0
    upsets_away = 0
    upsets_total = 0
    finals = 0
    finals_correct = 0
    gt_eliminated_in_semi = 0
    gt_eliminated_by = Counter()
    gt_in_pool = 0
    pool_recall_at_1 = 0
    pool_recall_at_3 = 0

    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        n += 1
        gt_code = truth["country_code"]
        pool = _candidate_pool(r)
        if pool and _matches(pool[0], gt_code):
            pool_recall_at_1 += 1
        if any(_matches(c, gt_code) for c in pool[:3]):
            pool_recall_at_3 += 1
        if any(_matches(c, gt_code) for c in pool):
            gt_in_pool += 1

        tl = _tournament(r)
        for m in tl:
            matches += 1
            ca, cb = m.get("country_a"), m.get("country_b")
            w = m.get("winner")
            ra, rb = m.get("pool_rank_a"), m.get("pool_rank_b")
            ok_a = _matches(ca, gt_code)
            ok_b = _matches(cb, gt_code)
            ok_w = _matches(w, gt_code)
            if not ok_a and not ok_b:
                match_both_wrong += 1
            elif ok_a and ok_b:
                pass  # impossible unless GT listed twice; ignore
            elif ok_w:
                match_toward_truth += 1
            else:
                match_away_from_truth += 1
            if ra is not None and rb is not None and w and ca:
                w_rank = ra if _same_country(w, ca) else rb
                l_rank = rb if _same_country(w, ca) else ra
                if w_rank > l_rank:
                    upsets_total += 1
                    if ok_w:
                        upsets_toward_truth += 1
                    elif ok_a or ok_b:
                        upsets_away += 1

        # Did GT reach the final?
        if tl:
            final_match = tl[-1]
            gt_in_final = (_matches(final_match.get("country_a"), gt_code)
                           or _matches(final_match.get("country_b"), gt_code))
            if final_match.get("round_label") == "final":
                finals += 1
                if _matches(final_match.get("winner"), gt_code):
                    finals_correct += 1
            # Track semi eliminations
            for m in tl:
                if m.get("round_label") in ("semi", "semi-1", "semi-2"):
                    ca, cb = m.get("country_a"), m.get("country_b")
                    if _matches(ca, gt_code) and not _matches(m.get("winner"), gt_code):
                        gt_eliminated_in_semi += 1
                        gt_eliminated_by[m.get("winner")] += 1
                    elif _matches(cb, gt_code) and not _matches(m.get("winner"), gt_code):
                        gt_eliminated_in_semi += 1
                        gt_eliminated_by[m.get("winner")] += 1

    if n == 0:
        print("  (no GT overlap)")
        print()
        return

    print(f"  GT-anchored images: {n}")
    print()
    print(f"  Pool recall @1: {pool_recall_at_1}/{n} "
          f"({pool_recall_at_1 / n * 100:.1f}%)   "
          f"@3: {pool_recall_at_3}/{n} "
          f"({pool_recall_at_3 / n * 100:.1f}%)   "
          f"@any: {gt_in_pool}/{n} "
          f"({gt_in_pool / n * 100:.1f}%)")
    print()
    print(f"  Per-match outcome ({matches} 1v1 matches):")
    decisive = match_toward_truth + match_away_from_truth
    print(f"    Toward truth (winner=GT, loser≠GT):    "
          f"{match_toward_truth}/{matches} "
          f"({match_toward_truth / matches * 100:.1f}%)")
    print(f"    Away from truth (winner≠GT, loser=GT): "
          f"{match_away_from_truth}/{matches} "
          f"({match_away_from_truth / matches * 100:.1f}%)")
    print(f"    Both wrong (GT not in match):          "
          f"{match_both_wrong}/{matches} "
          f"({match_both_wrong / matches * 100:.1f}%)")
    if decisive:
        print(f"    Of decisive matches (exactly one side = GT): "
              f"{match_toward_truth}/{decisive} "
              f"({match_toward_truth / decisive * 100:.1f}%) toward truth")
    print()
    print(f"  Upsets (lower-seed beat higher-seed): {upsets_total}")
    if upsets_total:
        print(f"    Upsets that moved TOWARD GT: {upsets_toward_truth}/"
              f"{upsets_total} ({upsets_toward_truth / upsets_total * 100:.1f}%)")
        print(f"    Upsets that moved AWAY from GT: {upsets_away}/"
              f"{upsets_total} ({upsets_away / upsets_total * 100:.1f}%)")
        print(f"    (remainder = neither side was GT)")
    print()
    print(f"  Finals played: {finals}    Finals won by GT: {finals_correct}/{finals}  "
          f"({finals_correct / finals * 100:.1f}%)" if finals else "  Finals played: 0")
    print(f"  GT eliminated in a semi: {gt_eliminated_in_semi}")
    if gt_eliminated_by:
        print(f"    Top eliminators of GT (winner names):")
        for k, v in gt_eliminated_by.most_common(5):
            print(f"      {k:<32} {v:>4}")
    print()


def _per_agent_calibration(results: list[dict], gt: dict) -> None:
    _section("PER-AGENT CALIBRATION - initial vs post-region-narrowing pick")
    _blurb(
        "For each agent, does re-assessing inside the confirmed region "
        "actually help? 'Init acc' = accuracy of the first pick (pre-PN); "
        "'Reg acc' = accuracy of the pick made after the region is "
        "confirmed. Δ tells you whether the region-narrowing step was a "
        "net win or loss for that agent. Constructive shifts (init "
        "wrong -> reg pick correct) are what you want; destructive "
        "shifts (init correct -> reg pick wrong) mean the constrained "
        "region actually pushed the agent off the truth. A near-zero "
        "Δ combined with a high StayOK means the agent is already "
        "committed at initial-assessment time and re-assessment is "
        "cosmetic."
    )
    print()
    print("  Init pick   = agent's top pick pre-PN")
    print("  Region pick = agent's top pick after region confirmed (country_assessment)")
    print()
    print(f"  {'Agent':<12} {'N':>5} {'Init acc':>10} {'Reg acc':>10} "
          f"{'Δ':>7} {'Constr':>7} {'Destr':>6} {'StayOK':>7} {'StayX':>6}")
    print(f"  {'─' * 12} {'─' * 5} {'─' * 10} {'─' * 10} "
          f"{'─' * 7} {'─' * 7} {'─' * 6} {'─' * 7} {'─' * 6}")
    for agent in AGENT_NAMES:
        n = ok_i = ok_f = 0
        c = d = sok = sx = 0
        for r in results:
            truth = gt.get(r["_name"])
            if not truth:
                continue
            ic = _initial_pick(r, agent)
            fc = _ca_pick(r, agent) or ic  # fall back if no re-assessment
            if not ic or not fc:
                continue
            n += 1
            oi = _matches(ic, truth["country_code"])
            of = _matches(fc, truth["country_code"])
            if oi:
                ok_i += 1
            if of:
                ok_f += 1
            if not oi and of:
                c += 1
            elif oi and not of:
                d += 1
            elif oi and of:
                sok += 1
            elif not oi and not of and _same_country(ic, fc):
                sx += 1
        if n == 0:
            print(f"  {agent:<12} {0:>5}  (no data)")
            continue
        ai = ok_i / n * 100
        af = ok_f / n * 100
        print(f"  {agent:<12} {n:>5} {ok_i}/{n}({ai:>4.1f}%) "
              f"{ok_f}/{n}({af:>4.1f}%) {af - ai:>+5.1f}% "
              f"{c:>7} {d:>6} {sok:>7} {sx:>6}")
    print()
    print("  Constr = init wrong -> region pick correct")
    print("  Destr  = init correct -> region pick wrong")
    print("  StayOK = both equal GT")
    print("  StayX  = both wrong, same country")
    print()


def _geographic_bias(results: list[dict], gt: dict) -> None:
    _section("GEOGRAPHIC BIAS (final judge prediction vs GT)")
    _blurb(
        "Systematic direction of error for the final coordinate "
        "prediction. Mean N-S / W-E offsets in km reveal whether the "
        "council is biased in a particular direction on the globe "
        "(e.g. predicting too far north or too far west on average). "
        "Comparing mean vs median exposes long-tail bias: a small "
        "median with a large mean means most predictions are close "
        "but a handful of wild misses drag the average."
    )
    print()
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
    print("  Sign convention: +Δlat = prediction north of GT, "
          "+Δlng = east of GT.")
    print("  W-E km uses cos(lat_GT).")
    print()


def _degenerate_bracket_accuracy(results: list[dict], gt: dict) -> None:
    _section("BRACKET SHAPE vs ACCURACY")
    _blurb(
        "Final accuracy split by the bracket shape (i.e. by pool size "
        "after PN). Degenerate shapes (final-only, walkover) correspond "
        "to images where PN was decisive and shrank the pool to 1-2 "
        "candidates; higher accuracy on those means PN made the right "
        "call. A large accuracy gap between full-bracket and the "
        "degenerate shapes means the pipeline knows when it's certain "
        "- the wide brackets are the genuinely hard images."
    )
    print()
    by_shape: dict[str, list[bool]] = {}
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        shape = _tournament_shape(r)
        fin = _final_country(r)
        by_shape.setdefault(shape, []).append(_matches(fin, truth["country_code"]))
    print(f"  {'Shape':<24} {'N':>5} {'Correct':>9} {'Accuracy':>10}")
    print(f"  {'─' * 24} {'─' * 5} {'─' * 9} {'─' * 10}")
    for shape in ("full-bracket", "3-way (semi+final)", "final-only",
                  "walkover", "other"):
        xs = by_shape.get(shape, [])
        if not xs:
            continue
        c = sum(xs)
        print(f"  {shape:<24} {len(xs):>5} {c:>9} "
              f"{c / len(xs) * 100:>9.1f}%")
    print()
    print("  Bracket shape is determined by candidate_pool size after PN.")
    print("  A walkover means only one country survived the pool filter.")
    print()


# ── Structured dynamics metrics (machine-readable, for the report) ────────


def compute_dynamics(results_dir: Path, gt_path: Path | None, out_dir: Path) -> dict:
    """Compute region narrowing funnel plus tournament bracket dynamics and
    write dynamics_metrics.json into out_dir.

    This is the machine readable counterpart to the printed GT pipeline
    analysis in analyze(), consumed by the report step so the single per
    approach report can carry this approach's dynamics. It reuses the same
    extractors (tournament shape, candidate pool, region helpers, matchers)
    so the JSON stays consistent with the printed tables.
    """
    results = _load_results(Path(results_dir))
    gt = _load_ground_truth(Path(gt_path)) if gt_path else {}
    total = len(results)

    # ── Bracket / pool dynamics (no GT needed) ────────────────────────────
    shape_counts: Counter = Counter()
    pool_sizes: list[int] = []
    match_count = 0
    agree = disagree = 0
    upsets = 0
    finals_played = 0
    top_seed_wins_final = 0

    # ── Region narrowing funnel (needs GT) ────────────────────────────────
    n_gt = 0
    gt_in_proposed = 0
    gt_in_surviving = 0
    gt_is_confirmed = 0
    gt_in_confirmed_or_runner = 0
    gt_in_pool = 0
    gt_in_pool_top3 = 0
    gt_at_pool_top = 0
    gt_reached_final = 0
    gt_won_tournament = 0
    final_correct = 0

    # ── Tournament vs GT (match level) ────────────────────────────────────
    decisive_matches = 0
    match_toward_truth = 0
    match_away_from_truth = 0
    match_both_wrong = 0
    upsets_total_gt = 0
    upsets_toward_truth = 0
    upsets_away = 0

    for r in results:
        shape_counts[_tournament_shape(r)] += 1
        pool_sizes.append(len(_candidate_pool(r)))

        tl = _tournament(r)
        for m in tl:
            match_count += 1
            ag = m.get("agreement")
            if ag == "agree":
                agree += 1
            elif ag == "disagree":
                disagree += 1
            ra = m.get("pool_rank_a")
            rb = m.get("pool_rank_b")
            w = m.get("winner")
            if ra is not None and rb is not None and w is not None:
                w_rank = ra if _same_country(w, m.get("country_a")) else rb
                l_rank = rb if _same_country(w, m.get("country_a")) else ra
                if w_rank > l_rank:
                    upsets += 1
                if m.get("round_label") == "final":
                    finals_played += 1
                    if w_rank == 0:
                        top_seed_wins_final += 1

        truth = gt.get(r.get("_name")) if gt else None
        if not truth:
            continue
        n_gt += 1
        gt_code = truth["country_code"]
        gt_reg = _gt_region(gt_code)

        prop = _proposed_regions(r)
        surv = _surviving_regions(r)
        conf = _confirmed_region(r)
        ru = _runner_up_region(r)
        pool = _candidate_pool(r)
        champ = _tournament_champion(r)
        fin = _final_country(r)

        if gt_reg and gt_reg in prop:
            gt_in_proposed += 1
        if gt_reg and gt_reg in surv:
            gt_in_surviving += 1
        if gt_reg and gt_reg == conf:
            gt_is_confirmed += 1
        if gt_reg and gt_reg in (conf, ru):
            gt_in_confirmed_or_runner += 1
        if any(_matches(c, gt_code) for c in pool):
            gt_in_pool += 1
        if any(_matches(c, gt_code) for c in pool[:3]):
            gt_in_pool_top3 += 1
        if pool and _matches(pool[0], gt_code):
            gt_at_pool_top += 1
        if tl:
            final_match = tl[-1]
            if (_matches(final_match.get("country_a"), gt_code)
                    or _matches(final_match.get("country_b"), gt_code)):
                gt_reached_final += 1
        if champ and _matches(champ, gt_code):
            gt_won_tournament += 1
        if fin and _matches(fin, gt_code):
            final_correct += 1

        for m in tl:
            ca, cb = m.get("country_a"), m.get("country_b")
            w = m.get("winner")
            ra, rb = m.get("pool_rank_a"), m.get("pool_rank_b")
            ok_a = _matches(ca, gt_code)
            ok_b = _matches(cb, gt_code)
            ok_w = _matches(w, gt_code)
            if not ok_a and not ok_b:
                match_both_wrong += 1
            elif ok_a and ok_b:
                pass
            elif ok_w:
                decisive_matches += 1
                match_toward_truth += 1
            else:
                decisive_matches += 1
                match_away_from_truth += 1
            if ra is not None and rb is not None and w and ca:
                w_rank = ra if _same_country(w, ca) else rb
                l_rank = rb if _same_country(w, ca) else ra
                if w_rank > l_rank:
                    upsets_total_gt += 1
                    if ok_w:
                        upsets_toward_truth += 1
                    elif ok_a or ok_b:
                        upsets_away += 1

    def _rate(a: int, b: int) -> float | None:
        return (a / b) if b else None

    metrics = {
        "n_total": total,
        "n_images_with_gt": n_gt,
        "pool": {
            "mean_size": _mean(pool_sizes),
            "median_size": _median(pool_sizes),
            "max_size": max(pool_sizes) if pool_sizes else 0,
        },
        "tournament_shape_distribution": {
            k: shape_counts.get(k, 0)
            for k in ("full-bracket", "3-way (semi+final)", "final-only",
                      "walkover", "other")
            if shape_counts.get(k, 0)
        },
        "bracket": {
            "n_matches": match_count,
            "n_agree": agree,
            "agree_rate": _rate(agree, match_count),
            "n_disagree": disagree,
            "disagree_rate": _rate(disagree, match_count),
            "n_upsets": upsets,
            "upset_rate": _rate(upsets, match_count),
            "n_finals_played": finals_played,
            "n_top_seed_wins_final": top_seed_wins_final,
            "top_seed_win_rate": _rate(top_seed_wins_final, finals_played),
        },
        "region_narrowing_funnel": {
            "n": n_gt,
            "stages": [
                {"code": "S0", "label": "GT region proposed",
                 "n": gt_in_proposed, "rate": _rate(gt_in_proposed, n_gt)},
                {"code": "S1", "label": "GT region survived PN vote",
                 "n": gt_in_surviving, "rate": _rate(gt_in_surviving, n_gt)},
                {"code": "S2", "label": "GT region confirmed",
                 "n": gt_is_confirmed, "rate": _rate(gt_is_confirmed, n_gt)},
                {"code": "S2b", "label": "GT region confirmed or runner up",
                 "n": gt_in_confirmed_or_runner,
                 "rate": _rate(gt_in_confirmed_or_runner, n_gt)},
                {"code": "S3", "label": "GT country in candidate pool",
                 "n": gt_in_pool, "rate": _rate(gt_in_pool, n_gt)},
                {"code": "S3b", "label": "GT country in pool top 3",
                 "n": gt_in_pool_top3, "rate": _rate(gt_in_pool_top3, n_gt)},
                {"code": "S3c", "label": "GT country is pool top seed",
                 "n": gt_at_pool_top, "rate": _rate(gt_at_pool_top, n_gt)},
                {"code": "S4", "label": "GT country reached tournament final",
                 "n": gt_reached_final, "rate": _rate(gt_reached_final, n_gt)},
                {"code": "S5", "label": "GT country won tournament",
                 "n": gt_won_tournament, "rate": _rate(gt_won_tournament, n_gt)},
                {"code": "S6", "label": "Final judge country is GT",
                 "n": final_correct, "rate": _rate(final_correct, n_gt)},
            ],
        },
        "tournament_gt": {
            "n_decisive_matches": decisive_matches,
            "n_toward_truth": match_toward_truth,
            "n_away_from_truth": match_away_from_truth,
            "n_both_wrong": match_both_wrong,
            "toward_truth_rate": _rate(match_toward_truth, decisive_matches),
            "upsets": {
                "n_total": upsets_total_gt,
                "n_toward_truth": upsets_toward_truth,
                "n_away_from_truth": upsets_away,
                "toward_truth_rate": _rate(upsets_toward_truth, upsets_total_gt),
            },
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dynamics_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[dynamics] wrote {out_file}")
    print(f"[dynamics] images={total} gt={n_gt} matches={match_count} "
          f"upsets={upsets} gt_won_tournament={gt_won_tournament} "
          f"final_correct={final_correct}")
    return metrics


# ── Driver ───────────────────────────────────────────────────────────────


def analyze(results_dir: Path, gt_path: Path | None = None) -> None:
    results = _load_results(results_dir)
    if not results:
        print("No results found.")
        return

    _overview(results)
    _region_proposal_behavior(results)
    _hypothesis_stance_distribution(results)
    _candidate_pool_composition(results)
    _tournament_dynamics(results)
    _initial_to_champion_shift(results)
    _timing(results)

    if gt_path:
        gt = _load_ground_truth(gt_path)
        print("=" * 74)
        print("  GROUND-TRUTH-BASED v12 ANALYSIS")
        print("=" * 74)
        print()
        _pipeline_accuracy_ladder(results, gt)
        _region_gate_analysis(results, gt)
        _hypothesis_calibration(results, gt)
        _tournament_gt_analysis(results, gt)
        _per_agent_calibration(results, gt)
        _degenerate_bracket_accuracy(results, gt)
        _geographic_bias(results, gt)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze the v12 pipeline (PN + Parallel Hypothesis + Tournament)"
    )
    parser.add_argument("results_dir", help="Directory with result.json files")
    parser.add_argument("ground_truth", nargs="?", default=None,
                        help="Optional path to georc_locations.csv")
    args = parser.parse_args()
    gt_path = Path(args.ground_truth) if args.ground_truth else None
    analyze(Path(args.results_dir), gt_path)


if __name__ == "__main__":
    main()
