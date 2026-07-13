"""Analyze the Hub-and-Spoke architecture.

Topology:  Image -> 5 agents independent assessments -> Judge interrogates
           specific agents with targeted questions (up to 3 rounds, hub
           = Judge, spokes = agents) -> Judge final decision.

Metrics tailored to hub-and-spoke (no pairings, no Round-2-for-all):
- Judge questioning behavior: who gets asked, how often, in which round
- Per-agent shift from initial assessment to last discussion response
- Convergence of final agent positions
- Question-targeting quality: does the Judge probe agents who started
  wrong, or agents who started right?

Usage:
    python -m vlm_council.analyze_rounds_hub_and_spoke results_hub_and_spoke/
    python -m vlm_council.analyze_rounds_hub_and_spoke results_hub_and_spoke/ Images/georc_locations.csv
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
KM_PER_DEG = 111.0


# ── Data loading and small helpers ───────────────────────────────────────


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


def _parse_response(text: str) -> dict:
    """Extract a candidates/evidence dict from a discussion response.

    Discussion responses come back as a JSON object inside a markdown
    code fence. Returns {} if unparseable.
    """
    if not text or not text.strip():
        return {}
    s = text.strip()
    # Strip code fences
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    s = s.strip()
    # Try direct parse
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        pass
    # Fallback: greedy match the largest {...} block
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            return d if isinstance(d, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def _initial_assessment(r: dict, agent: str) -> dict:
    return r.get("assessments", {}).get(agent, {}) or {}


def _last_response_assessment(r: dict, agent: str) -> dict | None:
    """Return the parsed last non-empty discussion response for `agent`,
    or None if the agent was never queried (or all responses were empty)."""
    log = r.get("discussion_log", [])
    last = None
    for entry in log:
        if not isinstance(entry, dict):
            continue
        if entry.get("target_agent") != agent:
            continue
        resp = entry.get("agent_response", "")
        if resp and resp.strip():
            last = resp
    if last is None:
        return None
    parsed = _parse_response(last)
    return parsed or None


def _final_pick(r: dict, agent: str) -> str | None:
    """The agent's final position: last discussion response if any,
    otherwise the initial assessment."""
    last = _last_response_assessment(r, agent)
    if last is not None:
        c = _top_country_of(last)
        if c:
            return c
    return _top_country_of(_initial_assessment(r, agent))


def _final_confidence(r: dict, agent: str) -> str | None:
    last = _last_response_assessment(r, agent)
    if last is not None:
        cf = _top_confidence_of(last)
        if cf:
            return cf
    return _top_confidence_of(_initial_assessment(r, agent))


def _initial_pick(r: dict, agent: str) -> str | None:
    return _top_country_of(_initial_assessment(r, agent))


def _initial_confidence(r: dict, agent: str) -> str | None:
    return _top_confidence_of(_initial_assessment(r, agent))


def _was_queried(r: dict, agent: str) -> bool:
    log = r.get("discussion_log", [])
    return any(
        isinstance(e, dict)
        and e.get("target_agent") == agent
        and (e.get("agent_response") or "").strip()
        for e in log
    )


def _unique_questions(r: dict) -> list[dict]:
    """Deduplicate the discussion_log: each Q&A appears twice (empty +
    filled). Keep one entry per (round_number, target_agent, question)."""
    seen: dict[tuple, dict] = {}
    for entry in r.get("discussion_log", []):
        if not isinstance(entry, dict):
            continue
        key = (
            entry.get("round_number"),
            entry.get("target_agent"),
            entry.get("judge_question", "")[:200],
        )
        existing = seen.get(key)
        if existing is None:
            seen[key] = entry
            continue
        # Prefer the entry with a non-empty response
        if not (existing.get("agent_response") or "").strip() and (
            entry.get("agent_response") or ""
        ).strip():
            seen[key] = entry
    return list(seen.values())


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
    """Return (top_country_lower, top_count, total_votes)."""
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


def _final_plurality(r: dict) -> tuple[str | None, int, int]:
    return _plurality({a: _final_pick(r, a) for a in AGENT_NAMES})


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def _hsep() -> None:
    print("-" * 70)


def _section(title: str) -> None:
    _hsep()
    print(f"  {title}")
    _hsep()


# ── Sections (no GT) ─────────────────────────────────────────────────────


def _overview(results: list[dict]) -> None:
    n = len(results)
    print("=" * 70)
    print("VLM Council - Hub-and-Spoke Analysis")
    print("=" * 70)
    print(f"Total images: {n}")
    print()

    no_disc = sum(1 for r in results if not _unique_questions(r))
    with_disc = n - no_disc

    qs_per_image = [len(_unique_questions(r)) for r in results]
    rounds_dist = Counter(r.get("discussion_rounds", 0) for r in results)
    answered = sum(1 for r in results for q in _unique_questions(r)
                   if (q.get("agent_response") or "").strip())
    total_qs = sum(qs_per_image)

    _section("HUB-AND-SPOKE OVERVIEW")
    print(f"  Images with no discussion (Judge accepted initial picks): "
          f"{no_disc}/{n} ({no_disc / n * 100:.0f}%)")
    print(f"  Images with discussion:                                   "
          f"{with_disc}/{n} ({with_disc / n * 100:.0f}%)")
    print()
    print(f"  Discussion rounds distribution:")
    for k in sorted(rounds_dist):
        print(f"    {k} rounds: {rounds_dist[k]} images "
              f"({rounds_dist[k] / n * 100:.1f}%)")
    print()
    if with_disc:
        non_zero = [q for q in qs_per_image if q > 0]
        print(f"  Among {with_disc} images with discussion:")
        print(f"    Total Judge questions: {total_qs}")
        print(f"    Questions answered:    {answered}/{total_qs} "
              f"({answered / max(total_qs, 1) * 100:.1f}%)")
        print(f"    Questions per image:   avg {_mean(non_zero):.1f}, "
              f"median {_median(non_zero):.1f}, max {max(non_zero)}")
    print()


def _judge_questioning_behavior(results: list[dict]) -> None:
    _section("JUDGE QUESTIONING BEHAVIOR (who gets probed, in which round)")
    print()
    by_agent: Counter = Counter()
    by_agent_round: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}
    answered_by_agent: Counter = Counter()
    images_queried_by_agent: Counter = Counter()

    for r in results:
        seen_agents: set[str] = set()
        for q in _unique_questions(r):
            a = q.get("target_agent")
            if a not in AGENT_NAMES:
                continue
            by_agent[a] += 1
            rnd = q.get("round_number") or 0
            by_agent_round[a][rnd] += 1
            if (q.get("agent_response") or "").strip():
                answered_by_agent[a] += 1
            seen_agents.add(a)
        for a in seen_agents:
            images_queried_by_agent[a] += 1

    total_q = sum(by_agent.values())
    if total_q == 0:
        print("  (no questions logged)")
        print()
        return

    print(f"  {'Agent':<12} {'Images':>7} {'Total Q':>8} {'R1':>5} {'R2':>5} "
          f"{'R3':>5} {'Answered':>9} {'Share %':>8}")
    print(f"  {'─' * 12} {'─' * 7} {'─' * 8} {'─' * 5} {'─' * 5} "
          f"{'─' * 5} {'─' * 9} {'─' * 8}")
    for a in AGENT_NAMES:
        n_q = by_agent[a]
        share = n_q / total_q * 100 if total_q else 0
        print(f"  {a:<12} {images_queried_by_agent[a]:>7} {n_q:>8} "
              f"{by_agent_round[a].get(1, 0):>5} "
              f"{by_agent_round[a].get(2, 0):>5} "
              f"{by_agent_round[a].get(3, 0):>5} "
              f"{answered_by_agent[a]:>9} {share:>7.1f}%")
    print()
    print("  Images   = unique images where this agent was queried at least once")
    print("  Total Q  = total Judge questions to this agent across all rounds")
    print("  R1/R2/R3 = questions split by discussion round")
    print("  Answered = questions with a non-empty response")
    print("  Share %  = Total Q / total questions across all agents")
    print()


def _per_agent_response_behavior(results: list[dict]) -> None:
    _section("PER-AGENT RESPONSE BEHAVIOR (initial -> last response, queried only)")
    print()
    print(f"  {'Agent':<12} {'Queried':>8} {'Parsed':>7} {'Changed':>8} "
          f"{'ConfUp':>7} {'ConfDown':>9} {'Change %':>9}")
    print(f"  {'─' * 12} {'─' * 8} {'─' * 7} {'─' * 8} "
          f"{'─' * 7} {'─' * 9} {'─' * 9}")
    total_up = total_down = total_parsed = 0
    for agent in AGENT_NAMES:
        queried = parsed = changed = up = down = 0
        for r in results:
            if not _was_queried(r, agent):
                continue
            queried += 1
            init = _initial_assessment(r, agent)
            last = _last_response_assessment(r, agent)
            ic = _top_country_of(init)
            lc = _top_country_of(last) if last else None
            if not ic or not lc:
                continue
            parsed += 1
            if not _same_country(ic, lc):
                changed += 1
            cf1 = CONF_ORDER.get(_top_confidence_of(init) or "", -1)
            cf2 = CONF_ORDER.get(_top_confidence_of(last) or "", -1)
            if cf2 > cf1:
                up += 1
            elif cf2 < cf1:
                down += 1
        total_up += up
        total_down += down
        total_parsed += parsed
        ch_pct = (changed / parsed * 100) if parsed else 0.0
        print(f"  {agent:<12} {queried:>8} {parsed:>7} {changed:>8} "
              f"{up:>7} {down:>9} {ch_pct:>8.0f}%")
    print()
    print("  Queried  = images where Judge sent at least one answered question to this agent")
    print("  Parsed   = subset where both initial and last response had a parseable top pick")
    print("  Changed  = last response top pick differs from initial top pick")
    print("  ConfUp   = last response confidence higher than initial; ConfDown = lower")
    print("  Change % = Changed / Parsed")
    print()
    if total_parsed:
        ratio = (f"{total_up / max(total_down, 1):.1f}×"
                 if total_down else "∞")
        print(f"  Confidence asymmetry: {total_up} ConfUp vs {total_down} ConfDown "
              f"across all {total_parsed} answered queries ({ratio}).")
        print()


def _agreement_dynamics(results: list[dict]) -> None:
    _section("AGREEMENT DYNAMICS (initial plurality vs final plurality)")
    print()
    n = len(results)
    init_hist: Counter = Counter()
    fin_hist: Counter = Counter()
    init_unan = fin_unan = same_top = became = lost = became_plur = lost_plur = 0
    for r in results:
        ti, ci, _ = _initial_plurality(r)
        tf, cf, _ = _final_plurality(r)
        if ci:
            init_hist[ci] += 1
        if cf:
            fin_hist[cf] += 1
        ui = (ci == len(AGENT_NAMES))
        uf = (cf == len(AGENT_NAMES))
        pi = (ci >= 3)
        pf = (cf >= 3)
        if ui:
            init_unan += 1
        if uf:
            fin_unan += 1
        if not ui and uf:
            became += 1
        if ui and not uf:
            lost += 1
        if not pi and pf:
            became_plur += 1
        if pi and not pf:
            lost_plur += 1
        if ti and tf and _same_country(ti, tf):
            same_top += 1

    print(f"  Plurality top-count distribution (top votes out of 5):")
    print(f"    {'Top votes':<11} {'Initial':>10} {'Final':>10}")
    print(f"    {'─' * 11} {'─' * 10} {'─' * 10}")
    for k in (1, 2, 3, 4, 5):
        print(f"    {k:<11} {init_hist.get(k, 0):>10} {fin_hist.get(k, 0):>10}")
    print()
    print(f"  Initial plurality reached (≥3/5):                 "
          f"{sum(init_hist[k] for k in (3, 4, 5))}/{n}")
    print(f"  Final plurality reached (≥3/5):                   "
          f"{sum(fin_hist[k] for k in (3, 4, 5))}/{n}")
    print(f"  Initial unanimous (5/5 same country):             {init_unan}/{n} "
          f"({init_unan / n * 100:.1f}%)")
    print(f"  Final unanimous (5/5 same country):               {fin_unan}/{n} "
          f"({fin_unan / n * 100:.1f}%)")
    print()
    print(f"  Initial sub-plurality -> Final plurality:          {became_plur}/{n}")
    print(f"  Initial plurality -> Final sub-plurality:          {lost_plur}/{n}")
    print(f"  Initial split -> Final unanimous:                  {became}/{n}")
    print(f"  Initial unanimous -> Final split:                  {lost}/{n}")
    print(f"  Same plurality top country in both phases:        {same_top}/{n}")
    print()


def _judge_source(results: list[dict]) -> None:
    _section("JUDGE FINAL CHOICE - WHERE DOES IT COME FROM?")
    print()
    n = both = init_only = fin_only = neither = 0
    for r in results:
        final = _extract_country(r.get("country_result", ""))
        if not final:
            continue
        n += 1
        ti, _, _ = _initial_plurality(r)
        tf, _, _ = _final_plurality(r)
        mi = _same_country(ti, final)
        mf = _same_country(tf, final)
        if mi and mf:
            both += 1
        elif mf and not mi:
            fin_only += 1
        elif mi and not mf:
            init_only += 1
        else:
            neither += 1

    if n == 0:
        print("  (no parsed final country)")
        print()
        return

    def pct(x: int) -> str:
        return f"{x / n * 100:>5.1f}%"

    print(f"  Images with parsed final country: {n}")
    print()
    print(f"  {'Judge matches':<40} {'Count':>6} {'Share':>7}")
    print(f"  {'─' * 40} {'─' * 6} {'─' * 7}")
    print(f"  {'Initial plurality AND Final plurality':<40} "
          f"{both:>6} {pct(both):>7}")
    print(f"  {'Final plurality only (initial differed)':<40} "
          f"{fin_only:>6} {pct(fin_only):>7}")
    print(f"  {'Initial plurality only (final differed)':<40} "
          f"{init_only:>6} {pct(init_only):>7}")
    print(f"  {'Neither (Judge picked own answer)':<40} "
          f"{neither:>6} {pct(neither):>7}")
    print()
    print(f"  Judge agrees with FINAL plurality on {both + fin_only}/{n} "
          f"({(both + fin_only) / n * 100:.1f}%); with INITIAL plurality on "
          f"{both + init_only}/{n} ({(both + init_only) / n * 100:.1f}%).")
    print()


def _timing(results: list[dict]) -> None:
    _section("TIMING")
    print()
    overall = [r.get("timing", {}).get("total_seconds") for r in results]
    overall = [t for t in overall if t is not None]
    with_disc = [r.get("timing", {}).get("total_seconds")
                 for r in results if _unique_questions(r)]
    with_disc = [t for t in with_disc if t is not None]
    no_disc = [r.get("timing", {}).get("total_seconds")
               for r in results if not _unique_questions(r)]
    no_disc = [t for t in no_disc if t is not None]
    if overall:
        print(f"  Overall:        avg {_mean(overall):.1f}s, median "
              f"{_median(overall):.1f}s, min {min(overall):.1f}s, max {max(overall):.1f}s")
    if with_disc:
        print(f"  With discussion: avg {_mean(with_disc):.1f}s, median "
              f"{_median(with_disc):.1f}s")
    if no_disc:
        print(f"  No discussion:   avg {_mean(no_disc):.1f}s, median "
              f"{_median(no_disc):.1f}s")
    if overall:
        print(f"  Total compute:   {sum(overall):.0f}s ({sum(overall) / 60:.1f} min)")
    print()


# ── Sections (GT-based) ──────────────────────────────────────────────────


def _classify_shift(c1: str, c2: str, gt_code: str) -> str:
    ok1 = _matches(c1, gt_code)
    ok2 = _matches(c2, gt_code)
    if ok1 and ok2:
        return "STAYED_CORRECT"
    if not ok1 and ok2:
        return "CONSTRUCTIVE"
    if ok1 and not ok2:
        return "DESTRUCTIVE"
    if _same_country(c1, c2):
        return "STAYED_WRONG"
    return "WRONG_TO_WRONG"


def _gt_questioning_outcomes(results: list[dict], gt: dict) -> None:
    _section("CONSTRUCTIVE vs DESTRUCTIVE QUESTIONING (per answered query)")
    print()
    counts: Counter = Counter()
    n_total = 0
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        gt_code = truth["country_code"]
        for agent in AGENT_NAMES:
            if not _was_queried(r, agent):
                continue
            init = _initial_assessment(r, agent)
            last = _last_response_assessment(r, agent)
            ic = _top_country_of(init)
            lc = _top_country_of(last) if last else None
            if not ic or not lc:
                continue
            n_total += 1
            counts[_classify_shift(ic, lc, gt_code)] += 1

    if n_total == 0:
        print("  (no answered queries with parseable picks)")
        print()
        return

    moved = (counts["CONSTRUCTIVE"] + counts["DESTRUCTIVE"]
             + counts["WRONG_TO_WRONG"])
    print(f"  Total answered queries with parseable picks: {n_total}")
    print()
    print(f"  ── Movement-only view (agent actually changed its top pick) ──")
    print(f"  Queries where pick changed: {moved}/{n_total} "
          f"({moved / n_total * 100:.1f}%)")
    if moved:
        c = counts["CONSTRUCTIVE"]
        d = counts["DESTRUCTIVE"]
        w = counts["WRONG_TO_WRONG"]
        print(f"    CONSTRUCTIVE  (wrong -> GT):              "
              f"{c:>4}/{moved} ({c / moved * 100:>5.1f}%)")
        print(f"    DESTRUCTIVE   (GT -> wrong):              "
              f"{d:>4}/{moved} ({d / moved * 100:>5.1f}%)")
        print(f"    LATERAL       (wrong -> other wrong):     "
              f"{w:>4}/{moved} ({w / moved * 100:>5.1f}%)")
    print()
    print(f"  ── Full 5-bucket view (all answered queries) ──")

    def line(label: str, key: str, hint: str) -> None:
        v = counts[key]
        pct = v / n_total * 100
        print(f"  {label:<22}{v:>4}  ({pct:>5.1f}%)")
        print(f"    └ {hint}")

    line("CONSTRUCTIVE",
         "CONSTRUCTIVE",
         "agent was wrong initially, response moved onto GT")
    line("DESTRUCTIVE",
         "DESTRUCTIVE",
         "agent was correct initially, response moved away from GT")
    line("STAYED_CORRECT",
         "STAYED_CORRECT",
         "both initial and final equal GT - questioning held the truth")
    line("STAYED_WRONG",
         "STAYED_WRONG",
         "both wrong on the same wrong country")
    line("WRONG_TO_WRONG",
         "WRONG_TO_WRONG",
         "both wrong on different countries (lateral move)")
    print()
    decisive = counts["CONSTRUCTIVE"] + counts["DESTRUCTIVE"]
    if decisive:
        c = counts["CONSTRUCTIVE"]
        d = counts["DESTRUCTIVE"]
        print(f"  Among the {decisive} queries where exactly one phase had GT:")
        print(f"    Constructive: {c}/{decisive} ({c / decisive * 100:.1f}%)")
        print(f"    Destructive:  {d}/{decisive} ({d / decisive * 100:.1f}%)")
    print()


def _gt_question_targeting(results: list[dict], gt: dict) -> None:
    _section("JUDGE QUESTION TARGETING - does Judge probe the right agents?")
    print()
    print("  For each answered query, we ask: at the time the Judge posed the")
    print("  question, was the target agent's INITIAL pick correct or wrong?")
    print("  Good question targeting probes wrong agents (more room to fix).")
    print()
    n = correct_target = wrong_target = 0
    by_agent_correct: Counter = Counter()
    by_agent_wrong: Counter = Counter()
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        gt_code = truth["country_code"]
        for agent in AGENT_NAMES:
            if not _was_queried(r, agent):
                continue
            init_c = _initial_pick(r, agent)
            if init_c is None:
                continue
            n += 1
            if _matches(init_c, gt_code):
                correct_target += 1
                by_agent_correct[agent] += 1
            else:
                wrong_target += 1
                by_agent_wrong[agent] += 1

    if n == 0:
        print("  (no queryable agents with GT)")
        print()
        return

    print(f"  Total queried agent-images with parseable initial pick: {n}")
    print(f"    Target was already CORRECT initially: {correct_target}/{n} "
          f"({correct_target / n * 100:.1f}%)")
    print(f"    Target was WRONG initially:           {wrong_target}/{n} "
          f"({wrong_target / n * 100:.1f}%)")
    print()
    print(f"  Per-agent breakdown:")
    print(f"    {'Agent':<12} {'Queried':>7} {'WrongInit':>10} "
          f"{'CorrectInit':>12} {'WrongInit %':>12}")
    print(f"    {'─' * 12} {'─' * 7} {'─' * 10} {'─' * 12} {'─' * 12}")
    for agent in AGENT_NAMES:
        w = by_agent_wrong[agent]
        c = by_agent_correct[agent]
        tot = w + c
        if tot == 0:
            continue
        print(f"    {agent:<12} {tot:>7} {w:>10} {c:>12} "
              f"{w / tot * 100:>11.1f}%")
    print()


def _gt_convergence(results: list[dict], gt: dict) -> None:
    _section("GT-BASED FINAL CONVERGENCE (per image, plurality ≥3/5)")
    print()
    n = plur_correct = plur_wrong = split = 0
    unan_correct = unan_wrong = 0
    examples_correct: list[str] = []
    examples_wrong: list[str] = []
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        top, cnt, total = _final_plurality(r)
        if total == 0:
            continue
        n += 1
        if cnt >= 3:
            ok = _matches(top, truth["country_code"])
            if ok:
                plur_correct += 1
                if cnt == len(AGENT_NAMES):
                    unan_correct += 1
                if len(examples_correct) < 5:
                    examples_correct.append(
                        f"{r['_name']} -> {top} ({cnt}/5, GT: {truth['country_name']})"
                    )
            else:
                plur_wrong += 1
                if cnt == len(AGENT_NAMES):
                    unan_wrong += 1
                if len(examples_wrong) < 5:
                    examples_wrong.append(
                        f"{r['_name']} -> {top} ({cnt}/5, GT: {truth['country_name']})"
                    )
        else:
            split += 1

    if n == 0:
        print("  (no images with final picks)")
        print()
        return

    print(f"  Images with at least one final pick: {n}")
    print(f"    Plurality on GT (correct):    {plur_correct}/{n} "
          f"({plur_correct / n * 100:.1f}%)")
    print(f"    Plurality on wrong country:   {plur_wrong}/{n} "
          f"({plur_wrong / n * 100:.1f}%)")
    print(f"    No plurality (top ≤ 2/5):     {split}/{n} "
          f"({split / n * 100:.1f}%)")
    plur_total = plur_correct + plur_wrong
    if plur_total:
        print(f"    Of plurality-converged: {plur_correct}/{plur_total} "
              f"({plur_correct / plur_total * 100:.1f}%) landed on GT")
    print()
    print(f"  Strict-unanimous (5/5) reference: "
          f"{unan_correct}/{n} on GT, {unan_wrong}/{n} on wrong country.")
    print()
    if examples_correct:
        print("  Examples (correct plurality):")
        for ex in examples_correct:
            print(f"    {ex}")
        print()
    if examples_wrong:
        print("  Examples (wrong plurality):")
        for ex in examples_wrong:
            print(f"    {ex}")
        print()


def _per_agent_shift_matrix(results: list[dict], gt: dict) -> None:
    _section("PER-AGENT INITIAL->FINAL SHIFT MATRIX (queried agents only)")
    print()
    print(f"  {'Agent':<12} {'N':>5} {'Constr':>7} {'Destr':>6} {'StayOK':>7} "
          f"{'StayX':>6} {'WrongShift':>11} {'NetTruth':>9}")
    print(f"  {'─' * 12} {'─' * 5} {'─' * 7} {'─' * 6} {'─' * 7} "
          f"{'─' * 6} {'─' * 11} {'─' * 9}")
    for agent in AGENT_NAMES:
        n = c = d = sok = sx = ws = 0
        for r in results:
            truth = gt.get(r["_name"])
            if not truth:
                continue
            if not _was_queried(r, agent):
                continue
            init = _initial_assessment(r, agent)
            last = _last_response_assessment(r, agent)
            ic = _top_country_of(init)
            lc = _top_country_of(last) if last else None
            if not ic or not lc:
                continue
            n += 1
            cls = _classify_shift(ic, lc, truth["country_code"])
            if cls == "CONSTRUCTIVE":
                c += 1
            elif cls == "DESTRUCTIVE":
                d += 1
            elif cls == "STAYED_CORRECT":
                sok += 1
            elif cls == "STAYED_WRONG":
                sx += 1
            else:
                ws += 1
        if n == 0:
            print(f"  {agent:<12} {0:>5}  (no queried agent-images with GT)")
            continue
        net = c - d
        print(f"  {agent:<12} {n:>5} {c:>7} {d:>6} {sok:>7} {sx:>6} {ws:>11} "
              f"{net:>+9}")
    print()
    print("  Constr     = initial wrong, response moved onto GT")
    print("  Destr      = initial correct, response moved away from GT")
    print("  StayOK     = both equal GT")
    print("  StayX      = both wrong, same country")
    print("  WrongShift = both wrong, different countries (lateral)")
    print("  NetTruth   = Constr - Destr")
    print()


def _per_agent_accuracy_delta(results: list[dict], gt: dict) -> None:
    _section("PER-AGENT INITIAL vs FINAL ACCURACY (all images)")
    print()
    print(f"  {'Agent':<12} {'N':>5} {'Init acc':>11} {'Final acc':>12} {'Δ':>7}")
    print(f"  {'─' * 12} {'─' * 5} {'─' * 11} {'─' * 12} {'─' * 7}")
    for agent in AGENT_NAMES:
        n = ok_i = ok_f = 0
        for r in results:
            truth = gt.get(r["_name"])
            if not truth:
                continue
            ic = _initial_pick(r, agent)
            fc = _final_pick(r, agent)
            if not ic or not fc:
                continue
            n += 1
            if _matches(ic, truth["country_code"]):
                ok_i += 1
            if _matches(fc, truth["country_code"]):
                ok_f += 1
        if n == 0:
            print(f"  {agent:<12} {0:>5}  (no GT data)")
            continue
        ai = ok_i / n * 100
        af = ok_f / n * 100
        print(f"  {agent:<12} {n:>5} {ok_i}/{n} ({ai:>4.1f}%) "
              f"{ok_f}/{n} ({af:>4.1f}%) {af - ai:>+6.1f}%")
    print()
    print("  Init acc  = share of images where the agent's initial top pick equals GT")
    print("  Final acc = same for the final position (= last response if queried, else initial)")
    print("  Δ = Final - Init in percentage points")
    print()


def _geographic_bias(results: list[dict], gt: dict) -> None:
    _section("GEOGRAPHIC BIAS (Council prediction vs ground truth)")
    print()
    dlat: list[float] = []
    dlng: list[float] = []
    ns_km: list[float] = []
    we_km: list[float] = []
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        pred = _extract_coordinates(r.get("country_result", ""))
        if not pred:
            continue
        gt_la, gt_lo = truth["lat"], truth["lng"]
        pred_la, pred_lo = pred
        dlo = pred_lo - gt_lo
        if dlo > 180:
            dlo -= 360
        elif dlo < -180:
            dlo += 360
        dlat.append(pred_la - gt_la)
        dlng.append(dlo)
        ns_km.append((pred_la - gt_la) * KM_PER_DEG)
        we_km.append(dlo * KM_PER_DEG * math.cos(math.radians(gt_la)))

    n = len(dlat)
    if n == 0:
        print("  (no images with parsed prediction coordinates)")
        print()
        return
    mean_ns, med_ns = _mean(ns_km), _median(ns_km)
    mean_we, med_we = _mean(we_km), _median(we_km)
    print(f"  Images with parsed prediction coordinates: {n}/{len(results)}")
    print()
    print(f"  Mean Δlat:           {_mean(dlat):+.2f}°")
    print(f"  Mean N-S offset:     {mean_ns:+.0f} km   (median {med_ns:+.0f} km)")
    print(f"    └ predictions tend {'NORTH' if mean_ns >= 0 else 'SOUTH'} of GT on average")
    print()
    print(f"  Mean Δlng:           {_mean(dlng):+.2f}°")
    print(f"  Mean W-E offset:     {mean_we:+.0f} km   (median {med_we:+.0f} km)")
    print(f"    └ predictions tend {'EAST' if mean_we >= 0 else 'WEST'} of GT on average")
    print()
    print("  Sign convention: positive Δlat = prediction north of GT;")
    print("                   positive Δlng = prediction east of GT.")
    print("  W-E km uses cos(lat_GT) so high-latitude shifts are not over-counted.")
    print()


def _per_agent_geographic_bias(results: list[dict], gt: dict) -> None:
    _section("PER-AGENT GEOGRAPHIC BIAS - initial and final (top pick -> centroid)")
    print()
    centroid_acc: dict[str, list[float]] = {}
    centroid_cnt: dict[str, int] = {}
    for entry in gt.values():
        code = entry["country_code"]
        centroid_acc.setdefault(code, [0.0, 0.0])
        centroid_acc[code][0] += entry["lat"]
        centroid_acc[code][1] += entry["lng"]
        centroid_cnt[code] = centroid_cnt.get(code, 0) + 1
    centroids = {
        c: (centroid_acc[c][0] / centroid_cnt[c], centroid_acc[c][1] / centroid_cnt[c])
        for c in centroid_acc
    }

    def _pick_centroid(country: str | None) -> tuple[float, float] | None:
        if not country:
            return None
        code = _NAME_TO_CODE.get(_normalize_country(country))
        return centroids.get(code) if code else None

    def _stats(picker, agent: str):
        dlat_a: list[float] = []
        dlng_a: list[float] = []
        ns: list[float] = []
        we: list[float] = []
        for r in results:
            truth = gt.get(r["_name"])
            if not truth:
                continue
            c = picker(r, agent)
            cen = _pick_centroid(c)
            if cen is None:
                continue
            gt_la, gt_lo = truth["lat"], truth["lng"]
            dlo = cen[1] - gt_lo
            if dlo > 180:
                dlo -= 360
            elif dlo < -180:
                dlo += 360
            dlat_a.append(cen[0] - gt_la)
            dlng_a.append(dlo)
            ns.append((cen[0] - gt_la) * KM_PER_DEG)
            we.append(dlo * KM_PER_DEG * math.cos(math.radians(gt_la)))
        return (len(ns), _mean(dlat_a), _mean(ns), _median(ns),
                _mean(dlng_a), _mean(we), _median(we))

    print(f"  {'Agent':<12} {'Phase':<7} {'N':>5} "
          f"{'MeanΔlat':>9} {'MeanN-S':>9} {'MedN-S':>8} "
          f"{'MeanΔlng':>9} {'MeanW-E':>9} {'MedW-E':>8}")
    print(f"  {'─' * 12} {'─' * 7} {'─' * 5} "
          f"{'─' * 9} {'─' * 9} {'─' * 8} "
          f"{'─' * 9} {'─' * 9} {'─' * 8}")
    for agent in AGENT_NAMES:
        for label, picker in (("Init", _initial_pick), ("Final", _final_pick)):
            n, mdlat, mns, mdns, mdlng, mwe, mdwe = _stats(picker, agent)
            if n == 0:
                continue
            print(f"  {agent:<12} {label:<7} {n:>5} "
                  f"{mdlat:>+8.2f}° {mns:>+8.0f} {mdns:>+7.0f} "
                  f"{mdlng:>+8.2f}° {mwe:>+8.0f} {mdwe:>+7.0f}")
    print()
    print(f"  Country centroids derived from {sum(centroid_cnt.values())} GT entries "
          f"across {len(centroids)} countries.")
    print("  Initial = each agent's pre-discussion top pick.")
    print("  Final   = last response if the agent was queried, else initial.")
    print()


# ── Driver ───────────────────────────────────────────────────────────────


def compute_dynamics(results_dir: Path, gt_path: Path | None, out_dir: Path) -> dict:
    """Compute structured hub and spoke dynamics metrics and write dynamics_metrics.json.

    This is the machine readable counterpart to ``analyze()``'s printed
    GT pipeline analysis, consumed by the ``report`` step so the single
    per approach report can carry the discussion dynamics. It reuses the
    existing helpers in this module: question deduplication, per agent
    response parsing, plurality voting and GT based shift classification.

    Covers: discussion rounds, plurality convergence, and per agent update
    behaviour (constructive / destructive / stayed / lateral shifts).
    """
    import json as _json

    results = _load_results(Path(results_dir))
    gt = _load_ground_truth(Path(gt_path)) if gt_path else {}
    total = len(results)

    # ── Discussion rounds distribution + questioning volume ──────────────
    rounds_dist: Counter = Counter(r.get("discussion_rounds", 0) for r in results)
    n_no_discussion = sum(1 for r in results if not _unique_questions(r))
    n_with_discussion = total - n_no_discussion
    qs_per_image = [len(_unique_questions(r)) for r in results]
    total_questions = sum(qs_per_image)
    answered_questions = sum(
        1 for r in results for q in _unique_questions(r)
        if (q.get("agent_response") or "").strip()
    )
    non_zero_qs = [q for q in qs_per_image if q > 0]

    # ── Plurality convergence (initial vs final, threshold >= 3/5) ───────
    init_plurality = 0
    final_plurality = 0
    init_unanimous = 0
    final_unanimous = 0
    same_top_country = 0
    for r in results:
        ti, ci, _ = _initial_plurality(r)
        tf, cf, _ = _final_plurality(r)
        if ci >= 3:
            init_plurality += 1
        if cf >= 3:
            final_plurality += 1
        if ci == len(AGENT_NAMES):
            init_unanimous += 1
        if cf == len(AGENT_NAMES):
            final_unanimous += 1
        if ti and tf and _same_country(ti, tf):
            same_top_country += 1

    # ── GT based per agent update behaviour (queried agents only) ────────
    per_agent_shift: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}
    shift_totals: Counter = Counter()
    n_answered_with_gt = 0
    n_images_with_gt = 0
    plur_correct = 0
    plur_wrong = 0
    plur_split = 0

    if gt:
        for r in results:
            truth = gt.get(r["_name"])
            if not truth:
                continue
            gt_code = truth["country_code"]
            n_images_with_gt += 1

            # per agent initial -> last response shift
            for agent in AGENT_NAMES:
                if not _was_queried(r, agent):
                    continue
                init = _initial_assessment(r, agent)
                last = _last_response_assessment(r, agent)
                ic = _top_country_of(init)
                lc = _top_country_of(last) if last else None
                if not ic or not lc:
                    continue
                n_answered_with_gt += 1
                cls = _classify_shift(ic, lc, gt_code)
                per_agent_shift[agent][cls] += 1
                shift_totals[cls] += 1

            # final plurality vs GT
            top, cnt, tot = _final_plurality(r)
            if tot:
                if cnt >= 3:
                    if _matches(top, gt_code):
                        plur_correct += 1
                    else:
                        plur_wrong += 1
                else:
                    plur_split += 1

    per_agent_out: dict[str, dict] = {}
    for agent in AGENT_NAMES:
        c = per_agent_shift[agent]
        constructive = c.get("CONSTRUCTIVE", 0)
        destructive = c.get("DESTRUCTIVE", 0)
        n_agent = sum(c.values())
        per_agent_out[agent] = {
            "n_answered": n_agent,
            "constructive": constructive,
            "destructive": destructive,
            "stayed_correct": c.get("STAYED_CORRECT", 0),
            "stayed_wrong": c.get("STAYED_WRONG", 0),
            "wrong_to_wrong": c.get("WRONG_TO_WRONG", 0),
            "net_truth": constructive - destructive,
        }

    plur_total = plur_correct + plur_wrong

    metrics = {
        "n_total": total,
        "discussion": {
            "n_no_discussion": n_no_discussion,
            "n_with_discussion": n_with_discussion,
            "discussion_rate": (n_with_discussion / total) if total else None,
            "rounds_distribution": {str(k): v for k, v in sorted(rounds_dist.items())},
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "mean_questions_per_discussion": _mean(non_zero_qs) if non_zero_qs else None,
        },
        "convergence": {
            "init_plurality": init_plurality,
            "final_plurality": final_plurality,
            "init_unanimous": init_unanimous,
            "final_unanimous": final_unanimous,
            "same_top_country": same_top_country,
            "final_plurality_rate": (final_plurality / total) if total else None,
        },
        "gt_plurality": {
            "n_images_with_gt": n_images_with_gt,
            "plurality_correct": plur_correct,
            "plurality_wrong": plur_wrong,
            "no_plurality": plur_split,
            "landed_on_gt_rate": (plur_correct / plur_total) if plur_total else None,
        },
        "per_agent_update_behaviour": {
            "n_answered_queries": n_answered_with_gt,
            "totals": {
                "constructive": shift_totals.get("CONSTRUCTIVE", 0),
                "destructive": shift_totals.get("DESTRUCTIVE", 0),
                "stayed_correct": shift_totals.get("STAYED_CORRECT", 0),
                "stayed_wrong": shift_totals.get("STAYED_WRONG", 0),
                "wrong_to_wrong": shift_totals.get("WRONG_TO_WRONG", 0),
            },
            "per_agent": per_agent_out,
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dynamics_metrics.json"
    with open(out_file, "w") as f:
        _json.dump(metrics, f, indent=2)
    print(f"[dynamics] wrote {out_file}")
    print(f"[dynamics] with_discussion={n_with_discussion} "
          f"final_plurality={final_plurality} "
          f"answered_queries_with_gt={n_answered_with_gt} "
          f"constructive={shift_totals.get('CONSTRUCTIVE', 0)} "
          f"destructive={shift_totals.get('DESTRUCTIVE', 0)}")
    return metrics


def analyze(results_dir: Path, gt_path: Path | None = None) -> None:
    results = _load_results(results_dir)
    if not results:
        print("No results found.")
        return

    _overview(results)
    _judge_questioning_behavior(results)
    _per_agent_response_behavior(results)
    _agreement_dynamics(results)
    _judge_source(results)
    _timing(results)

    if gt_path:
        gt = _load_ground_truth(gt_path)
        print("=" * 70)
        print("GROUND-TRUTH-BASED HUB-AND-SPOKE ANALYSIS")
        print("=" * 70)
        print()
        _gt_questioning_outcomes(results, gt)
        _gt_question_targeting(results, gt)
        _gt_convergence(results, gt)
        _per_agent_shift_matrix(results, gt)
        _per_agent_accuracy_delta(results, gt)
        _geographic_bias(results, gt)
        _per_agent_geographic_bias(results, gt)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze the Hub-and-Spoke architecture (Judge interrogates agents)"
    )
    parser.add_argument("results_dir", help="Directory with result.json files")
    parser.add_argument("ground_truth", nargs="?", default=None,
                        help="Optional path to georc_locations.csv")
    args = parser.parse_args()
    gt_path = Path(args.ground_truth) if args.ground_truth else None
    analyze(Path(args.results_dir), gt_path)


if __name__ == "__main__":
    main()
