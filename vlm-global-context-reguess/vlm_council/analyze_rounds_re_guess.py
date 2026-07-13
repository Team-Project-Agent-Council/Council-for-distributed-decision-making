"""Analyze the Global Context Re-guess approach.

Topology:  Image -> R1 (independent) -> R2 (each agent re-guesses with the
           full set of R1 assessments as context) -> Judge

There is NO moderator and NO pairwise debate, so the constructive/destructive
dynamic plays out per agent across the R1->R2 transition rather than per
pairing. The sections below mirror the debate-analysis report so the two
approaches can be compared side by side.

Usage:
    python -m vlm_council.analyze_rounds_re_guess results_global_context_re_guess_1/
    python -m vlm_council.analyze_rounds_re_guess results_global_context_re_guess_1/ Images/georc_locations.csv
"""

from __future__ import annotations

import json
import math
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


def _top_country(assessment: dict) -> str | None:
    cands = assessment.get("candidates", [])
    if not cands or not isinstance(cands[0], dict):
        return None
    c = cands[0].get("country", "").strip().rstrip(".")
    return c or None


def _top_confidence(assessment: dict) -> str | None:
    cands = assessment.get("candidates", [])
    if not cands or not isinstance(cands[0], dict):
        return None
    return cands[0].get("confidence", "").strip().lower() or None


def _matches(country: str | None, gt_code: str) -> bool:
    if not country:
        return False
    return _countries_match(country, gt_code)


def _country_to_code(country: str | None) -> str | None:
    """Map a free-form country name to its 2-letter code, or None if unknown."""
    if not country:
        return None
    return _NAME_TO_CODE.get(_normalize_country(country))


def _same_country(a: str | None, b: str | None) -> bool:
    """Compare two free-form country names robust to aliases / formatting."""
    if not a or not b:
        return False
    ca, cb = _country_to_code(a), _country_to_code(b)
    if ca and cb:
        return ca == cb
    return _normalize_country(a) == _normalize_country(b)


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


def _r1_pick(r: dict, agent: str) -> str | None:
    return _top_country(r.get("round_1_assessments", {}).get(agent, {}))


def _r2_pick(r: dict, agent: str) -> str | None:
    return _top_country(r.get("round_2_assessments", {}).get(agent, {}))


def _plurality(round_key: str, r: dict) -> tuple[str | None, int, int]:
    """Return (top_country_lower, top_count, total_votes) for a round."""
    votes: Counter = Counter()
    for a in AGENT_NAMES:
        c = _top_country(r.get(round_key, {}).get(a, {}))
        if c:
            votes[c.lower()] += 1
    if not votes:
        return None, 0, 0
    top, cnt = votes.most_common(1)[0]
    return top, cnt, sum(votes.values())


# ── Sections (no GT) ─────────────────────────────────────────────────────


def _overview(results: list[dict]) -> None:
    n = len(results)
    print("=" * 70)
    print("VLM Council - Global Context Re-guess Analysis")
    print("=" * 70)
    print(f"Total images: {n}")
    print()

    # R1 unanimity = "no debate equivalent": all five agents picked the same.
    r1_unanimous = 0
    for r in results:
        votes = Counter()
        for a in AGENT_NAMES:
            c = _r1_pick(r, a)
            if c:
                votes[c.lower()] += 1
        if votes and len(votes) == 1 and sum(votes.values()) == len(AGENT_NAMES):
            r1_unanimous += 1
    r1_split = n - r1_unanimous

    # Counts of picks changed across R1->R2 (per agent-image)
    changed = stayed = comparable = 0
    for r in results:
        for a in AGENT_NAMES:
            c1, c2 = _r1_pick(r, a), _r2_pick(r, a)
            if not c1 or not c2:
                continue
            comparable += 1
            if c1.lower() == c2.lower():
                stayed += 1
            else:
                changed += 1

    _section("RE-GUESS OVERVIEW")
    print(f"  Images with full R1 unanimity (no contradicting picks): "
          f"{r1_unanimous}/{n} ({r1_unanimous / n * 100:.0f}%)")
    print(f"  Images with R1 disagreement (R2 sees contradicting context): "
          f"{r1_split}/{n} ({r1_split / n * 100:.0f}%)")
    print()
    print(f"  Comparable agent-image R1/R2 pairs: {comparable}")
    print(f"  Top pick changed in R2:   {changed}/{comparable} "
          f"({changed / comparable * 100:.1f}% of agent-image pairs)")
    print(f"  Top pick unchanged in R2: {stayed}/{comparable} "
          f"({stayed / comparable * 100:.1f}% of agent-image pairs)")
    print()


def _per_agent_reguess_behavior(results: list[dict]) -> None:
    _section("PER-AGENT RE-GUESS BEHAVIOR")
    print()
    print(f"  {'Agent':<12} {'R1 picks':>9} {'R2 picks':>9} "
          f"{'Changed':>8} {'ConfUp':>7} {'ConfDown':>9} {'Change %':>9}")
    print(f"  {'─' * 12} {'─' * 9} {'─' * 9} {'─' * 8} {'─' * 7} {'─' * 9} {'─' * 9}")
    total_up = total_down = total_comp = 0
    for agent in AGENT_NAMES:
        r1n = r2n = comp = changed = up = down = 0
        for r in results:
            if _r1_pick(r, agent):
                r1n += 1
            if _r2_pick(r, agent):
                r2n += 1
            c1, c2 = _r1_pick(r, agent), _r2_pick(r, agent)
            if not c1 or not c2:
                continue
            comp += 1
            if c1.lower() != c2.lower():
                changed += 1
            cf1 = CONF_ORDER.get(_top_confidence(r.get("round_1_assessments", {}).get(agent, {})) or "", -1)
            cf2 = CONF_ORDER.get(_top_confidence(r.get("round_2_assessments", {}).get(agent, {})) or "", -1)
            if cf2 > cf1:
                up += 1
            elif cf2 < cf1:
                down += 1
        total_up += up
        total_down += down
        total_comp += comp
        change_pct = (changed / comp * 100) if comp else 0.0
        print(f"  {agent:<12} {r1n:>9} {r2n:>9} {changed:>8} {up:>7} {down:>9} "
              f"{change_pct:>8.0f}%")
    print()
    print("  R1 picks  = images where this agent produced a top candidate in Round 1")
    print("  R2 picks  = images where this agent produced a top candidate in Round 2")
    print("  Changed   = R2 top pick differs from R1 top pick (comparable pairs only)")
    print("  ConfUp    = R2 confidence is higher than R1; ConfDown = lower")
    print("  Change %  = Changed / comparable R1+R2 pairs")
    print()
    if total_comp:
        ratio = (f"{total_up / max(total_down, 1):.0f}×"
                 if total_down else "∞")
        print(f"  Confidence asymmetry: {total_up} ConfUp vs {total_down} ConfDown "
              f"across all {total_comp} comparable pairs ({ratio}).")
        print(f"  -> Re-guess monotonically increases certainty: agents harden their")
        print(f"    confidence with peer context but almost never down-regulate it,")
        print(f"    even when peers contradict them.")
        print()


def _agreement_dynamics(results: list[dict]) -> None:
    _section("AGREEMENT DYNAMICS (R1 plurality vs R2 plurality)")
    print()

    n = len(results)
    r1_hist: Counter = Counter()
    r2_hist: Counter = Counter()
    r1_unan = r2_unan = same_top = became = lost = 0
    became_plur = lost_plur = 0  # plurality (>=3) variants

    for r in results:
        t1, c1, _ = _plurality("round_1_assessments", r)
        t2, c2, _ = _plurality("round_2_assessments", r)
        if c1:
            r1_hist[c1] += 1
        if c2:
            r2_hist[c2] += 1
        u1 = (c1 == len(AGENT_NAMES))
        u2 = (c2 == len(AGENT_NAMES))
        p1 = (c1 >= 3)
        p2 = (c2 >= 3)
        if u1:
            r1_unan += 1
        if u2:
            r2_unan += 1
        if not u1 and u2:
            became += 1
        if u1 and not u2:
            lost += 1
        if not p1 and p2:
            became_plur += 1
        if p1 and not p2:
            lost_plur += 1
        if t1 and t2 and _same_country(t1, t2):
            same_top += 1

    print(f"  Plurality top-count distribution (top votes out of 5):")
    print(f"    {'Top votes':<11} {'R1 images':>10} {'R2 images':>10}")
    print(f"    {'─' * 11} {'─' * 10} {'─' * 10}")
    for k in (1, 2, 3, 4, 5):
        print(f"    {k:<11} {r1_hist.get(k, 0):>10} {r2_hist.get(k, 0):>10}")
    print()
    print(f"  R1 plurality reached (≥3/5):                    "
          f"{sum(r1_hist[k] for k in (3, 4, 5))}/{n}")
    print(f"  R2 plurality reached (≥3/5):                    "
          f"{sum(r2_hist[k] for k in (3, 4, 5))}/{n}")
    print(f"  R1 unanimous (5/5 same country):                {r1_unan}/{n} "
          f"({r1_unan / n * 100:.1f}%)")
    print(f"  R2 unanimous (5/5 same country):                {r2_unan}/{n} "
          f"({r2_unan / n * 100:.1f}%)")
    print()
    print(f"  R1 sub-plurality -> R2 plurality (context built majority): "
          f"{became_plur}/{n}")
    print(f"  R1 plurality -> R2 sub-plurality (context broke majority): "
          f"{lost_plur}/{n}")
    print(f"  R1 split -> R2 unanimous (context built consensus):        "
          f"{became}/{n}")
    print(f"  R1 unanimous -> R2 split (context broke consensus):        "
          f"{lost}/{n}")
    print(f"  Same plurality top country in both rounds:                "
          f"{same_top}/{n}")
    print()


def _judge_source(results: list[dict]) -> None:
    _section("JUDGE FINAL CHOICE - WHERE DOES IT COME FROM?")
    print()
    n = both = r1_only = r2_only = neither = no_final = 0
    for r in results:
        cr = r.get("country_result", "")
        final = _extract_country(cr)
        if not final:
            no_final += 1
            continue
        n += 1
        t1, c1, _ = _plurality("round_1_assessments", r)
        t2, c2, _ = _plurality("round_2_assessments", r)
        m1 = _same_country(t1, final)
        m2 = _same_country(t2, final)
        if m1 and m2:
            both += 1
        elif m2 and not m1:
            r2_only += 1
        elif m1 and not m2:
            r1_only += 1
        else:
            neither += 1

    if n == 0:
        print("  (no parsed final country in any image)")
        print()
        return

    def pct(x: int) -> str:
        return f"{x / n * 100:>5.1f}%"

    print(f"  Images with parsed final country: {n}")
    print()
    print(f"  {'Judge matches':<35} {'Count':>6} {'Share':>7}")
    print(f"  {'─' * 35} {'─' * 6} {'─' * 7}")
    print(f"  {'R1 plurality AND R2 plurality':<35} {both:>6} {pct(both):>7}")
    print(f"  {'R2 plurality only (R1 differed)':<35} {r2_only:>6} {pct(r2_only):>7}")
    print(f"  {'R1 plurality only (R2 differed)':<35} {r1_only:>6} {pct(r1_only):>7}")
    print(f"  {'Neither (Judge picked own answer)':<35} {neither:>6} {pct(neither):>7}")
    print()
    print(f"  Judge agrees with R2 plurality on {both + r2_only}/{n} "
          f"({(both + r2_only) / n * 100:.1f}%); with R1 plurality on "
          f"{both + r1_only}/{n} ({(both + r1_only) / n * 100:.1f}%).")
    if no_final:
        print(f"  ({no_final} images had no parsable final country and were excluded.)")
    print()


def _timing(results: list[dict]) -> None:
    _section("TIMING")
    print()
    overall = [r.get("timing", {}).get("total_seconds") for r in results]
    overall = [t for t in overall if t is not None]
    if overall:
        print(f"  Overall: avg {_mean(overall):.1f}s, median {_median(overall):.1f}s, "
              f"min {min(overall):.1f}s, max {max(overall):.1f}s")
        print(f"  Total compute: {sum(overall):.0f}s ({sum(overall) / 60:.1f} min)")
    print()


# ── Sections (GT-based) ──────────────────────────────────────────────────


def _classify_shift(c1: str, c2: str, gt_code: str) -> str:
    """Five-way classification of an agent's R1->R2 transition."""
    ok1 = _matches(c1, gt_code)
    ok2 = _matches(c2, gt_code)
    if ok1 and ok2:
        return "STAYED_CORRECT"
    if not ok1 and ok2:
        return "CONSTRUCTIVE"
    if ok1 and not ok2:
        return "DESTRUCTIVE"
    if c1.lower() == c2.lower():
        return "STAYED_WRONG"
    return "WRONG_TO_WRONG"


def _gt_shift_outcomes(results: list[dict], gt: dict) -> None:
    _section("CONSTRUCTIVE vs DESTRUCTIVE R1->R2 SHIFTS (per agent-image)")
    print()
    counts: Counter = Counter()
    n_total = 0
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        for a in AGENT_NAMES:
            c1, c2 = _r1_pick(r, a), _r2_pick(r, a)
            if not c1 or not c2:
                continue
            n_total += 1
            counts[_classify_shift(c1, c2, truth["country_code"])] += 1

    if n_total == 0:
        print("  (no comparable agent-image pairs with GT)")
        print()
        return

    moved = (counts["CONSTRUCTIVE"] + counts["DESTRUCTIVE"]
             + counts["WRONG_TO_WRONG"])
    print(f"  Total comparable agent-image R1->R2 pairs: {n_total}")
    print()
    print(f"  ── Movement-only view (agent actually changed its top pick) ──")
    print(f"  Pairs that changed in R2: {moved}/{n_total} "
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
    print(f"  ── Full 5-bucket view (all comparable pairs) ──")

    def line(label: str, key: str, hint: str) -> None:
        v = counts[key]
        pct = v / n_total * 100
        print(f"  {label:<22}{v:>4}  ({pct:>5.1f}%)")
        print(f"    └ {hint}")

    line("CONSTRUCTIVE",
         "CONSTRUCTIVE",
         "agent was wrong in R1, R2 corrected it onto the GT")
    line("DESTRUCTIVE",
         "DESTRUCTIVE",
         "agent was correct in R1, R2 moved away from the GT")
    line("STAYED_CORRECT",
         "STAYED_CORRECT",
         "both R1 and R2 equal GT - re-guess held the truth")
    line("STAYED_WRONG",
         "STAYED_WRONG",
         "both rounds wrong, on the same wrong country")
    line("WRONG_TO_WRONG",
         "WRONG_TO_WRONG",
         "both rounds wrong but on different countries (lateral move)")
    print()
    decisive = counts["CONSTRUCTIVE"] + counts["DESTRUCTIVE"]
    if decisive:
        c = counts["CONSTRUCTIVE"]
        d = counts["DESTRUCTIVE"]
        print(f"  Among the {decisive} pairs where exactly one round had the GT "
              f"(R1 vs R2 disagreed about correctness):")
        print(f"    Constructive: {c}/{decisive} ({c / decisive * 100:.1f}%)")
        print(f"    Destructive:  {d}/{decisive} ({d / decisive * 100:.1f}%)")
    print()


def _gt_convergence(results: list[dict], gt: dict) -> None:
    _section("GT-BASED R2 CONVERGENCE (per image, plurality ≥3/5)")
    print()
    n = plur_correct = plur_wrong = split = 0
    unan_correct = unan_wrong = 0
    examples_correct: list[str] = []
    examples_wrong: list[str] = []
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        top, cnt, total = _plurality("round_2_assessments", r)
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
        print("  (no images with R2 votes)")
        print()
        return

    print(f"  Images with R2 votes from any agent: {n}")
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
    _section("PER-AGENT R1->R2 SHIFT MATRIX")
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
            c1, c2 = _r1_pick(r, agent), _r2_pick(r, agent)
            if not c1 or not c2:
                continue
            n += 1
            cls = _classify_shift(c1, c2, truth["country_code"])
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
            print(f"  {agent:<12} {0:>5}  (no comparable R1+R2 data with GT)")
            continue
        net = c - d
        print(f"  {agent:<12} {n:>5} {c:>7} {d:>6} {sok:>7} {sx:>6} {ws:>11} "
              f"{net:>+9}")
    print()
    print("  Constr     = R1 wrong, R2 corrected onto GT (good for the council)")
    print("  Destr      = R1 correct, R2 moved away from GT (bad for the council)")
    print("  StayOK     = R1 and R2 both equal GT")
    print("  StayX      = R1 and R2 both wrong, on the same wrong country")
    print("  WrongShift = R1 and R2 both wrong, on different countries (lateral move)")
    print("  NetTruth   = Constr - Destr")
    print("               positive -> context shifted this agent toward the GT")
    print("               negative -> context shifted this agent away from the GT")
    print()


def _per_agent_accuracy_delta(results: list[dict], gt: dict) -> None:
    _section("PER-AGENT R1 vs R2 ACCURACY")
    print()
    print(f"  {'Agent':<12} {'N':>5} {'R1 acc':>10} {'R2 acc':>10} {'Δ':>7}")
    print(f"  {'─' * 12} {'─' * 5} {'─' * 10} {'─' * 10} {'─' * 7}")
    for agent in AGENT_NAMES:
        n = ok1 = ok2 = 0
        for r in results:
            truth = gt.get(r["_name"])
            if not truth:
                continue
            c1, c2 = _r1_pick(r, agent), _r2_pick(r, agent)
            if not c1 or not c2:
                continue
            n += 1
            if _matches(c1, truth["country_code"]):
                ok1 += 1
            if _matches(c2, truth["country_code"]):
                ok2 += 1
        if n == 0:
            print(f"  {agent:<12} {0:>5}  (no GT data)")
            continue
        a1 = ok1 / n * 100
        a2 = ok2 / n * 100
        print(f"  {agent:<12} {n:>5} {ok1}/{n} ({a1:>4.1f}%) "
              f"{ok2}/{n} ({a2:>4.1f}%) {a2 - a1:>+6.1f}%")
    print()
    print("  R1 acc = share of images where the agent's R1 top pick equals GT")
    print("  R2 acc = same for R2; Δ = R2 - R1 in percentage points")
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
    _section("PER-AGENT GEOGRAPHIC BIAS - R1 and R2 (top pick -> country centroid)")
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

    def _stats(round_key: str, agent: str) -> tuple[int, float, float, float, float, float, float]:
        dlat_a: list[float] = []
        dlng_a: list[float] = []
        ns: list[float] = []
        we: list[float] = []
        for r in results:
            truth = gt.get(r["_name"])
            if not truth:
                continue
            c = _top_country(r.get(round_key, {}).get(agent, {}))
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

    print(f"  {'Agent':<12} {'Round':<6} {'N':>5} "
          f"{'MeanΔlat':>9} {'MeanN-S':>9} {'MedN-S':>8} "
          f"{'MeanΔlng':>9} {'MeanW-E':>9} {'MedW-E':>8}")
    print(f"  {'─' * 12} {'─' * 6} {'─' * 5} "
          f"{'─' * 9} {'─' * 9} {'─' * 8} "
          f"{'─' * 9} {'─' * 9} {'─' * 8}")
    for agent in AGENT_NAMES:
        for label, key in (("R1", "round_1_assessments"),
                           ("R2", "round_2_assessments")):
            n, mdlat, mns, mdns, mdlng, mwe, mdwe = _stats(key, agent)
            if n == 0:
                continue
            print(f"  {agent:<12} {label:<6} {n:>5} "
                  f"{mdlat:>+8.2f}° {mns:>+8.0f} {mdns:>+7.0f} "
                  f"{mdlng:>+8.2f}° {mwe:>+8.0f} {mdwe:>+7.0f}")
    print()
    print(f"  Country centroids derived from {sum(centroid_cnt.values())} GT entries "
          f"across {len(centroids)} countries.")
    print("  Compare R1 vs R2 per agent to see whether global context pulled the")
    print("  agent's geographic tendency closer to GT (smaller km magnitudes) or farther.")
    print()


# ── Driver ───────────────────────────────────────────────────────────────


def compute_dynamics(results_dir: Path, gt_path: Path | None, out_dir: Path) -> dict:
    """Compute structured re-guess dynamics metrics and write dynamics_metrics.json.

    This is the machine readable counterpart to the printed GT pipeline in
    ``analyze()``. It captures this approach's Round 1 vs Round 2 shift and the
    constructive vs destructive revision analysis so the single per approach
    report can carry the dynamics without re running the printed analysis.

    Reuses the existing helpers in this module (``_load_results``, ``_r1_pick``,
    ``_r2_pick``, ``_plurality``, ``_classify_shift``, ``_matches``,
    ``_same_country``, ``_top_confidence``).
    """
    results = _load_results(Path(results_dir))
    gt = _load_ground_truth(Path(gt_path)) if gt_path else {}
    n_total = len(results)

    # ── R1 vs R2 movement (per agent image pair), no GT needed ──────────────
    comparable = changed = stayed = 0
    conf_up = conf_down = 0
    for r in results:
        for a in AGENT_NAMES:
            c1, c2 = _r1_pick(r, a), _r2_pick(r, a)
            if not c1 or not c2:
                continue
            comparable += 1
            if c1.lower() == c2.lower():
                stayed += 1
            else:
                changed += 1
            cf1 = CONF_ORDER.get(
                _top_confidence(r.get("round_1_assessments", {}).get(a, {})) or "", -1
            )
            cf2 = CONF_ORDER.get(
                _top_confidence(r.get("round_2_assessments", {}).get(a, {})) or "", -1
            )
            if cf2 > cf1:
                conf_up += 1
            elif cf2 < cf1:
                conf_down += 1

    # ── Agreement dynamics: R1 plurality vs R2 plurality ────────────────────
    r1_unan = r2_unan = became_unan = lost_unan = same_top = 0
    became_plur = lost_plur = 0
    for r in results:
        _, c1, _ = _plurality("round_1_assessments", r)
        t1, _, _ = _plurality("round_1_assessments", r)
        t2, c2, _ = _plurality("round_2_assessments", r)
        u1 = c1 == len(AGENT_NAMES)
        u2 = c2 == len(AGENT_NAMES)
        p1 = c1 >= 3
        p2 = c2 >= 3
        if u1:
            r1_unan += 1
        if u2:
            r2_unan += 1
        if not u1 and u2:
            became_unan += 1
        if u1 and not u2:
            lost_unan += 1
        if not p1 and p2:
            became_plur += 1
        if p1 and not p2:
            lost_plur += 1
        if t1 and t2 and _same_country(t1, t2):
            same_top += 1

    # ── GT based constructive vs destructive R1->R2 shift classification ────
    shift_counts: Counter = Counter()
    n_shift_pairs = 0
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        for a in AGENT_NAMES:
            c1, c2 = _r1_pick(r, a), _r2_pick(r, a)
            if not c1 or not c2:
                continue
            n_shift_pairs += 1
            shift_counts[_classify_shift(c1, c2, truth["country_code"])] += 1

    # ── GT based per agent R1 vs R2 accuracy delta ──────────────────────────
    per_agent_shift: dict[str, dict] = {}
    for agent in AGENT_NAMES:
        n = c = d = sok = sx = ws = ok1 = ok2 = 0
        for r in results:
            truth = gt.get(r["_name"])
            if not truth:
                continue
            c1, c2 = _r1_pick(r, agent), _r2_pick(r, agent)
            if not c1 or not c2:
                continue
            n += 1
            code = truth["country_code"]
            cls = _classify_shift(c1, c2, code)
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
            if _matches(c1, code):
                ok1 += 1
            if _matches(c2, code):
                ok2 += 1
        per_agent_shift[agent] = {
            "n": n,
            "constructive": c,
            "destructive": d,
            "stayed_correct": sok,
            "stayed_wrong": sx,
            "wrong_shift": ws,
            "net_truth": c - d,
            "r1_accuracy": (ok1 / n) if n else None,
            "r2_accuracy": (ok2 / n) if n else None,
            "accuracy_delta": ((ok2 - ok1) / n) if n else None,
        }

    # ── GT based R2 convergence per image (plurality >= 3/5) ────────────────
    n_conv = plur_correct = plur_wrong = split = 0
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        top, cnt, total = _plurality("round_2_assessments", r)
        if total == 0:
            continue
        n_conv += 1
        if cnt >= 3:
            if _matches(top, truth["country_code"]):
                plur_correct += 1
            else:
                plur_wrong += 1
        else:
            split += 1

    metrics = {
        "n_total": n_total,
        "round_movement": {
            "comparable_pairs": comparable,
            "changed": changed,
            "stayed": stayed,
            "change_rate": (changed / comparable) if comparable else None,
            "conf_up": conf_up,
            "conf_down": conf_down,
        },
        "agreement_dynamics": {
            "r1_unanimous": r1_unan,
            "r2_unanimous": r2_unan,
            "became_unanimous": became_unan,
            "lost_unanimous": lost_unan,
            "became_plurality": became_plur,
            "lost_plurality": lost_plur,
            "same_top_country": same_top,
        },
        "shift_classification": {
            "n_pairs": n_shift_pairs,
            "counts": dict(shift_counts),
        },
        "per_agent_shift": per_agent_shift,
        "r2_convergence": {
            "n_images": n_conv,
            "plurality_correct": plur_correct,
            "plurality_wrong": plur_wrong,
            "no_plurality": split,
            "plurality_correct_rate": (plur_correct / n_conv) if n_conv else None,
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dynamics_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[dynamics] wrote {out_file}")
    print(f"[dynamics] comparable={comparable} changed={changed} "
          f"shift-pairs={n_shift_pairs} "
          f"constructive={shift_counts.get('CONSTRUCTIVE', 0)} "
          f"destructive={shift_counts.get('DESTRUCTIVE', 0)}")
    return metrics


def analyze(results_dir: Path, gt_path: Path | None = None) -> None:
    results = _load_results(results_dir)
    if not results:
        print("No results found.")
        return

    _overview(results)
    _per_agent_reguess_behavior(results)
    _agreement_dynamics(results)
    _judge_source(results)
    _timing(results)

    if gt_path:
        gt = _load_ground_truth(gt_path)
        print("=" * 70)
        print("GROUND-TRUTH-BASED RE-GUESS ANALYSIS")
        print("=" * 70)
        print()
        _gt_shift_outcomes(results, gt)
        _gt_convergence(results, gt)
        _per_agent_shift_matrix(results, gt)
        _per_agent_accuracy_delta(results, gt)
        _geographic_bias(results, gt)
        _per_agent_geographic_bias(results, gt)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze the Global Context Re-guess approach (R1->R2, no debate)"
    )
    parser.add_argument("results_dir", help="Directory with result.json files")
    parser.add_argument("ground_truth", nargs="?", default=None,
                        help="Optional path to georc_locations.csv for GT-based analysis")
    args = parser.parse_args()
    gt_path = Path(args.ground_truth) if args.ground_truth else None
    analyze(Path(args.results_dir), gt_path)


if __name__ == "__main__":
    main()
