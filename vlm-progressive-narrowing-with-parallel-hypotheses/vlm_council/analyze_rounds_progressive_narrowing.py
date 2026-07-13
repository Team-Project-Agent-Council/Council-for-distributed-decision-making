"""Analyze the Progressive Narrowing architecture.

Topology:
    Image -> 5 agents independent assessments -> region proposal phase
    -> if region_consensus: Path A (jump straight to Judge)
      else:                Path B (hypothesis evaluations + per-agent
                                    country re-assessment inside the
                                    confirmed region) -> Judge.

Metrics tailored to PN (no symmetric R1/R2; the "second round" only
happens on Path B):
- Path A vs Path B distribution and region-consensus rate
- Region-voting behaviour (per agent: votes inside the confirmed region)
- Hypothesis-evaluation stance distribution per agent
- Initial -> country-assessment shifts (Path B only)
- Judge source split by path
- GT-based: region accuracy, hypothesis-quality, accuracy delta, etc.

Usage:
    python -m vlm_council.analyze_rounds_progressive_narrowing results_pn_500/
    python -m vlm_council.analyze_rounds_progressive_narrowing \\
        results_pn_500/ Images/georc_locations.csv
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
HE_STANCE_ORDER = [
    "strongly_support", "support", "neutral",
    "contradicts", "strongly_contradicts",
]
KM_PER_DEG = 111.0


# ── ISO-code -> PN region map ─────────────────────────────────────────────
#
# The label set comes from the PN pipeline itself (the regions actually
# observed as `confirmed_region`). The mapping is hand-curated by ISO code
# so it composes with the alias logic in `evaluate.py`.

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

# Build a normalized lookup so confirmed_region strings like
# "sub-saharan_africa" or "central asia" all collapse to one bucket.
_REGION_KEYS = {
    re.sub(r"[^a-z]+", "", r.lower()): r for r in PN_REGIONS
}


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


def _country_in_region(country: str | None, region: str | None) -> bool:
    if not country or not region:
        return False
    code = _NAME_TO_CODE.get(_normalize_country(country))
    if not code:
        return False
    return code in PN_REGIONS.get(region, set())


# ── Data loading + small helpers ─────────────────────────────────────────


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


def _initial_pick(r: dict, agent: str) -> str | None:
    return _top_country_of(r.get("assessments", {}).get(agent, {}) or {})


def _ca_pick(r: dict, agent: str) -> str | None:
    return _top_country_of(r.get("country_assessments", {}).get(agent, {}) or {})


def _final_pick(r: dict, agent: str) -> str | None:
    """Final position: CA pick if Path B, else initial pick."""
    if (r.get("progressive_narrowing", {}) or {}).get("path") == "B":
        c = _ca_pick(r, agent)
        if c:
            return c
    return _initial_pick(r, agent)


def _initial_confidence(r: dict, agent: str) -> str | None:
    return _top_confidence_of(r.get("assessments", {}).get(agent, {}) or {})


def _ca_confidence(r: dict, agent: str) -> str | None:
    return _top_confidence_of(r.get("country_assessments", {}).get(agent, {}) or {})


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


def _ca_plurality(r: dict) -> tuple[str | None, int, int]:
    return _plurality({a: _ca_pick(r, a) for a in AGENT_NAMES})


def _final_plurality(r: dict) -> tuple[str | None, int, int]:
    """Plurality of final picks (CA on Path B, initial on Path A)."""
    return _plurality({a: _final_pick(r, a) for a in AGENT_NAMES})


def _path(r: dict) -> str:
    return (r.get("progressive_narrowing", {}) or {}).get("path") or "?"


def _confirmed_region(r: dict) -> str | None:
    return _canon_region((r.get("progressive_narrowing", {}) or {}).get("confirmed_region"))


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
    print("VLM Council - Progressive Narrowing Analysis")
    print("=" * 70)
    print(f"Total images: {n}")
    print()

    paths = Counter(_path(r) for r in results)
    consensus = sum(1 for r in results
                    if (r.get("progressive_narrowing", {}) or {}).get("region_consensus"))
    n_prop = Counter(
        len((r.get("progressive_narrowing", {}) or {}).get("proposed_regions") or [])
        for r in results
    )
    conf_regions = Counter(_confirmed_region(r) or "?" for r in results)

    _section("PN OVERVIEW")
    print(f"  Region consensus reached (Path A): "
          f"{consensus}/{n} ({consensus / n * 100:.1f}%)")
    print(f"  No consensus (Path B, full pipeline): "
          f"{n - consensus}/{n} ({(n - consensus) / n * 100:.1f}%)")
    print()
    print(f"  Path distribution:")
    for k in sorted(paths):
        print(f"    Path {k}: {paths[k]} images ({paths[k] / n * 100:.1f}%)")
    print()
    print(f"  Proposed-regions count distribution:")
    for k in sorted(n_prop):
        print(f"    {k} region{'s' if k != 1 else ''} proposed: "
              f"{n_prop[k]} images ({n_prop[k] / n * 100:.1f}%)")
    print()
    print(f"  Confirmed-region distribution (top 12):")
    for region, c in conf_regions.most_common(12):
        print(f"    {region:<30} {c:>4} ({c / n * 100:>4.1f}%)")
    print()


def _per_agent_initial_picks(results: list[dict]) -> None:
    _section("PER-AGENT INITIAL PICKS (top countries per agent)")
    print()
    for agent in AGENT_NAMES:
        countries: Counter = Counter()
        confs: Counter = Counter()
        for r in results:
            c = _initial_pick(r, agent)
            cf = _initial_confidence(r, agent)
            if c:
                countries[c] += 1
            if cf:
                confs[cf] += 1
        top = ", ".join(f"{k} ({v})" for k, v in countries.most_common(5))
        conf_str = ", ".join(f"{k}: {v}" for k, v in sorted(confs.items(),
                                                             key=lambda kv: -CONF_ORDER.get(kv[0], -1)))
        print(f"  {agent.upper():<12} top picks: {top}")
        print(f"  {'':<12} confidence: {conf_str}")
        print()


def _region_voting_behavior(results: list[dict]) -> None:
    _section("REGION VOTING BEHAVIOR (votes inside the confirmed region)")
    print()
    print("  For each image, the confirmed region is the region the council")
    print("  ended up working in. We ask: at the time of the INITIAL")
    print("  assessment, did each agent's top pick fall inside that region?")
    print()
    print(f"  {'Agent':<12} {'Inside':>8} {'Outside':>8} {'Total':>7} "
          f"{'Inside %':>9}")
    print(f"  {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 7} {'─' * 9}")
    for agent in AGENT_NAMES:
        inside = outside = 0
        for r in results:
            region = _confirmed_region(r)
            pick = _initial_pick(r, agent)
            if not region or not pick:
                continue
            if _country_in_region(pick, region):
                inside += 1
            else:
                outside += 1
        total = inside + outside
        pct = inside / total * 100 if total else 0
        print(f"  {agent:<12} {inside:>8} {outside:>8} {total:>7} "
              f"{pct:>8.1f}%")
    print()


def _hypothesis_stance_dist(results: list[dict]) -> None:
    _section("HYPOTHESIS-EVALUATION STANCES (per agent, Path B only)")
    print()
    by_agent: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}
    n_path_b = 0
    n_evals = 0
    for r in results:
        if _path(r) != "B":
            continue
        n_path_b += 1
        for h in r.get("hypothesis_evaluations", []) or []:
            ag = h.get("agent_name")
            stance = (h.get("confidence") or "").lower()
            if ag in AGENT_NAMES:
                by_agent[ag][stance] += 1
                n_evals += 1

    if n_path_b == 0:
        print("  (no Path B images)")
        print()
        return

    print(f"  {n_path_b} Path-B images; {n_evals} hypothesis evaluations parsed.")
    print()
    header_keys = HE_STANCE_ORDER
    print(f"  {'Agent':<12}" + "".join(f" {k:>20}" for k in header_keys))
    print(f"  {'─' * 12}" + "".join(f" {'─' * 20}" for _ in header_keys))
    for agent in AGENT_NAMES:
        c = by_agent[agent]
        total = sum(c.values())
        row = f"  {agent:<12}"
        for k in header_keys:
            v = c.get(k, 0)
            pct = v / total * 100 if total else 0.0
            row += f" {v:>5} ({pct:>5.1f}%)   "
        print(row)
    print()
    print("  Other stance labels observed (off-spec): ", end="")
    other = Counter()
    for c in by_agent.values():
        for k, v in c.items():
            if k not in header_keys:
                other[k] += v
    if other:
        print(", ".join(f"{k}={v}" for k, v in other.most_common()))
    else:
        print("(none)")
    print()


def _agreement_dynamics(results: list[dict]) -> None:
    _section("AGREEMENT DYNAMICS (Path B: initial plurality vs CA plurality)")
    print()
    path_b = [r for r in results if _path(r) == "B"]
    n = len(path_b)
    if n == 0:
        print("  (no Path B images)")
        print()
        return

    init_hist: Counter = Counter()
    ca_hist: Counter = Counter()
    init_unan = ca_unan = same_top = became = lost = 0
    became_plur = lost_plur = 0
    for r in path_b:
        ti, ci, _ = _initial_plurality(r)
        tc, cc, _ = _ca_plurality(r)
        if ci:
            init_hist[ci] += 1
        if cc:
            ca_hist[cc] += 1
        ui = (ci == len(AGENT_NAMES))
        uc = (cc == len(AGENT_NAMES))
        pi = (ci >= 3)
        pc = (cc >= 3)
        if ui:
            init_unan += 1
        if uc:
            ca_unan += 1
        if not ui and uc:
            became += 1
        if ui and not uc:
            lost += 1
        if not pi and pc:
            became_plur += 1
        if pi and not pc:
            lost_plur += 1
        if ti and tc and _same_country(ti, tc):
            same_top += 1

    print(f"  Plurality top-count distribution (votes out of 5):")
    print(f"    {'Top votes':<11} {'Initial':>10} {'CountryAsm':>12}")
    print(f"    {'─' * 11} {'─' * 10} {'─' * 12}")
    for k in (1, 2, 3, 4, 5):
        print(f"    {k:<11} {init_hist.get(k, 0):>10} {ca_hist.get(k, 0):>12}")
    print()
    print(f"  Initial plurality reached (≥3/5):        "
          f"{sum(init_hist[k] for k in (3, 4, 5))}/{n}")
    print(f"  CA plurality reached (≥3/5):             "
          f"{sum(ca_hist[k] for k in (3, 4, 5))}/{n}")
    print(f"  Initial unanimous (5/5):                 {init_unan}/{n} "
          f"({init_unan / n * 100:.1f}%)")
    print(f"  CA unanimous (5/5):                      {ca_unan}/{n} "
          f"({ca_unan / n * 100:.1f}%)")
    print()
    print(f"  Initial sub-plurality -> CA plurality:    {became_plur}/{n}")
    print(f"  Initial plurality -> CA sub-plurality:    {lost_plur}/{n}")
    print(f"  Initial split -> CA unanimous:            {became}/{n}")
    print(f"  Initial unanimous -> CA split:            {lost}/{n}")
    print(f"  Same plurality top country in both phases: {same_top}/{n}")
    print()
    print("  Note: on Path A the country assessment is skipped, so this")
    print("  section is meaningful only for Path B (where re-assessment")
    print("  inside the confirmed region actually runs).")
    print()


def _judge_source(results: list[dict]) -> None:
    _section("JUDGE FINAL CHOICE - WHERE DOES IT COME FROM?")
    print()
    print("  Bucketed by path: Path A images have no CA stage, so the only")
    print("  signal the Judge can draw from is the initial plurality.")
    print()

    # Path A
    a_results = [r for r in results if _path(r) == "A"]
    a_match = a_diff = a_no_init = 0
    for r in a_results:
        final = _extract_country(r.get("country_result", ""))
        if not final:
            continue
        ti, _, _ = _initial_plurality(r)
        if not ti:
            a_no_init += 1
            continue
        if _same_country(ti, final):
            a_match += 1
        else:
            a_diff += 1
    a_total = a_match + a_diff
    print(f"  ── Path A ({len(a_results)} images, no CA stage) ──")
    if a_total:
        print(f"    Judge matches initial plurality:     "
              f"{a_match}/{a_total} ({a_match / a_total * 100:.1f}%)")
        print(f"    Judge picks something else:          "
              f"{a_diff}/{a_total} ({a_diff / a_total * 100:.1f}%)")
    print()

    # Path B
    b_results = [r for r in results if _path(r) == "B"]
    both = ca_only = init_only = neither = 0
    n_b = 0
    for r in b_results:
        final = _extract_country(r.get("country_result", ""))
        if not final:
            continue
        n_b += 1
        ti, _, _ = _initial_plurality(r)
        tc, _, _ = _ca_plurality(r)
        mi = _same_country(ti, final)
        mc = _same_country(tc, final)
        if mi and mc:
            both += 1
        elif mc and not mi:
            ca_only += 1
        elif mi and not mc:
            init_only += 1
        else:
            neither += 1

    print(f"  ── Path B ({len(b_results)} images, full pipeline) ──")
    if n_b == 0:
        print("    (no parsed final country)")
        print()
        return

    def pct(x: int) -> str:
        return f"{x / n_b * 100:>5.1f}%"

    print(f"    {'Judge matches':<40} {'Count':>6} {'Share':>7}")
    print(f"    {'─' * 40} {'─' * 6} {'─' * 7}")
    print(f"    {'Initial plurality AND CA plurality':<40} "
          f"{both:>6} {pct(both):>7}")
    print(f"    {'CA plurality only (initial differed)':<40} "
          f"{ca_only:>6} {pct(ca_only):>7}")
    print(f"    {'Initial plurality only (CA differed)':<40} "
          f"{init_only:>6} {pct(init_only):>7}")
    print(f"    {'Neither (Judge picked own answer)':<40} "
          f"{neither:>6} {pct(neither):>7}")
    print()
    print(f"    Judge agrees with CA plurality on {both + ca_only}/{n_b} "
          f"({(both + ca_only) / n_b * 100:.1f}%); "
          f"with INITIAL plurality on {both + init_only}/{n_b} "
          f"({(both + init_only) / n_b * 100:.1f}%).")
    print()


def _timing(results: list[dict]) -> None:
    _section("TIMING")
    print()
    all_t = [r.get("timing", {}).get("total_seconds") for r in results]
    all_t = [t for t in all_t if t is not None]
    a_t = [r.get("timing", {}).get("total_seconds")
           for r in results if _path(r) == "A"]
    a_t = [t for t in a_t if t is not None]
    b_t = [r.get("timing", {}).get("total_seconds")
           for r in results if _path(r) == "B"]
    b_t = [t for t in b_t if t is not None]
    if all_t:
        print(f"  Overall:      avg {_mean(all_t):.1f}s, median {_median(all_t):.1f}s, "
              f"min {min(all_t):.1f}s, max {max(all_t):.1f}s")
    if a_t:
        print(f"  Path A only:  avg {_mean(a_t):.1f}s, median {_median(a_t):.1f}s "
              f"({len(a_t)} images)")
    if b_t:
        print(f"  Path B only:  avg {_mean(b_t):.1f}s, median {_median(b_t):.1f}s "
              f"({len(b_t)} images)")
    if all_t:
        print(f"  Total compute: {sum(all_t):.0f}s ({sum(all_t) / 60:.1f} min)")
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


def _gt_region_accuracy(results: list[dict], gt: dict) -> None:
    _section("REGION ACCURACY (does `confirmed_region` contain the GT?)")
    print()
    n = ok = miss = unk_region = no_gt = 0
    by_path = {"A": [0, 0], "B": [0, 0]}  # [hits, total]
    miss_examples: list[str] = []
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            no_gt += 1
            continue
        region = _confirmed_region(r)
        if not region:
            unk_region += 1
            continue
        n += 1
        gt_code = truth["country_code"]
        gt_region = _gt_region(gt_code)
        if gt_region is None:
            continue  # GT country not mapped (skip silently rather than penalising)
        path = _path(r)
        if path in by_path:
            by_path[path][1] += 1
        if region == gt_region:
            ok += 1
            if path in by_path:
                by_path[path][0] += 1
        else:
            miss += 1
            if len(miss_examples) < 10:
                miss_examples.append(
                    f"{r['_name']}: confirmed={region}, GT={truth['country_name']} ({gt_region})"
                )

    if n == 0:
        print("  (no images with GT and a confirmed region)")
        print()
        return

    print(f"  Images with GT and a confirmed region: {n}")
    print(f"    Region matches GT:            {ok}/{n} ({ok / n * 100:.1f}%)")
    print(f"    Region does NOT match GT:     {miss}/{n} ({miss / n * 100:.1f}%)")
    print()
    print(f"  Split by path:")
    for path in ("A", "B"):
        h, t = by_path[path]
        if t == 0:
            continue
        print(f"    Path {path}: {h}/{t} correct region ({h / t * 100:.1f}%)")
    print()
    if miss_examples:
        print("  Sample region mismatches:")
        for ex in miss_examples:
            print(f"    {ex}")
        print()


def _gt_hypothesis_quality(results: list[dict], gt: dict) -> None:
    _section("HYPOTHESIS-EVALUATION QUALITY (per-agent stance toward GT region)")
    print()
    print("  For each Path-B image whose GT country falls in one of the")
    print("  proposed regions, we look at each agent's stance toward the")
    print("  region_<gt_region> hypothesis. A good agent strongly supports")
    print("  the correct region and contradicts the wrong ones.")
    print()

    # Build a normalized lookup of region hypothesis IDs once
    def _hyp_region_key(hid: str | None) -> str | None:
        if not hid or not hid.startswith("region_"):
            return None
        return re.sub(r"[^a-z]+", "", hid[len("region_"):].lower())

    stance_score = {"strongly_support": 2, "support": 1, "neutral": 0,
                    "contradicts": -1, "strongly_contradicts": -2}

    # per-agent stance distributions: toward GT region and toward wrong regions
    by_agent_gt: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}
    by_agent_other: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}
    n_evaluable = 0
    for r in results:
        if _path(r) != "B":
            continue
        truth = gt.get(r["_name"])
        if not truth:
            continue
        gt_region = _gt_region(truth["country_code"])
        if not gt_region:
            continue
        proposed = (r.get("progressive_narrowing", {}) or {}).get(
            "proposed_regions") or []
        proposed_canon = {_canon_region(p) for p in proposed}
        if gt_region not in proposed_canon:
            continue
        n_evaluable += 1
        gt_key = re.sub(r"[^a-z]+", "", gt_region.lower())
        for h in r.get("hypothesis_evaluations", []) or []:
            ag = h.get("agent_name")
            if ag not in AGENT_NAMES:
                continue
            hk = _hyp_region_key(h.get("hypothesis_id"))
            if not hk:
                continue
            stance = (h.get("confidence") or "").lower()
            if hk == gt_key:
                by_agent_gt[ag][stance] += 1
            else:
                by_agent_other[ag][stance] += 1

    if n_evaluable == 0:
        print("  (no Path-B images where GT region was among the proposals)")
        print()
        return

    print(f"  Evaluable Path-B images (GT region among proposals): {n_evaluable}")
    print()
    print(f"  ── Mean stance score toward the GT region (higher = better) ──")
    print(f"  Score scale: strongly_support=+2, support=+1, neutral=0, "
          f"contradicts=-1, strongly_contradicts=-2")
    print()
    print(f"  {'Agent':<12} {'N(GT-hyp)':>10} {'MeanGT':>8} "
          f"{'N(OtherHyp)':>12} {'MeanOther':>10} {'Δ (GT-Other)':>14}")
    print(f"  {'─' * 12} {'─' * 10} {'─' * 8} {'─' * 12} {'─' * 10} {'─' * 14}")
    for agent in AGENT_NAMES:
        gt_c = by_agent_gt[agent]
        ot_c = by_agent_other[agent]
        n_gt = sum(gt_c.values())
        n_ot = sum(ot_c.values())
        m_gt = (sum(stance_score.get(k, 0) * v for k, v in gt_c.items()) / n_gt
                if n_gt else 0.0)
        m_ot = (sum(stance_score.get(k, 0) * v for k, v in ot_c.items()) / n_ot
                if n_ot else 0.0)
        delta = m_gt - m_ot
        print(f"  {agent:<12} {n_gt:>10} {m_gt:>+8.2f} {n_ot:>12} "
              f"{m_ot:>+10.2f} {delta:>+14.2f}")
    print()
    print("  Δ (GT - Other): how much more favourably the agent rates the")
    print("  TRUE region's hypothesis compared to the wrong-region hypotheses.")
    print("  Larger positive Δ = the agent's region judgement is discriminative.")
    print()


def _gt_shift_outcomes(results: list[dict], gt: dict) -> None:
    _section("CONSTRUCTIVE vs DESTRUCTIVE INITIAL->CA SHIFTS (Path B only)")
    print()
    counts: Counter = Counter()
    n_total = 0
    for r in results:
        if _path(r) != "B":
            continue
        truth = gt.get(r["_name"])
        if not truth:
            continue
        gt_code = truth["country_code"]
        for agent in AGENT_NAMES:
            ic = _initial_pick(r, agent)
            cc = _ca_pick(r, agent)
            if not ic or not cc:
                continue
            n_total += 1
            counts[_classify_shift(ic, cc, gt_code)] += 1

    if n_total == 0:
        print("  (no Path-B agent-images with both initial and CA picks)")
        print()
        return

    moved = (counts["CONSTRUCTIVE"] + counts["DESTRUCTIVE"]
             + counts["WRONG_TO_WRONG"])
    print(f"  Total Path-B agent-images with parseable picks (initial + CA): "
          f"{n_total}")
    print()
    print(f"  ── Movement-only view (agent actually changed its top pick) ──")
    print(f"  Agent-images where pick changed: {moved}/{n_total} "
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
    print(f"  ── Full 5-bucket view (all comparable agent-images) ──")

    def line(label: str, key: str, hint: str) -> None:
        v = counts[key]
        pct = v / n_total * 100
        print(f"  {label:<22}{v:>4}  ({pct:>5.1f}%)")
        print(f"    └ {hint}")

    line("CONSTRUCTIVE", "CONSTRUCTIVE",
         "agent was wrong initially, CA assessment moved onto GT")
    line("DESTRUCTIVE", "DESTRUCTIVE",
         "agent was correct initially, CA assessment moved away from GT")
    line("STAYED_CORRECT", "STAYED_CORRECT",
         "both initial and CA equal GT - region narrowing held the truth")
    line("STAYED_WRONG", "STAYED_WRONG",
         "both wrong on the same wrong country")
    line("WRONG_TO_WRONG", "WRONG_TO_WRONG",
         "both wrong on different countries (lateral move)")
    print()
    decisive = counts["CONSTRUCTIVE"] + counts["DESTRUCTIVE"]
    if decisive:
        c = counts["CONSTRUCTIVE"]
        d = counts["DESTRUCTIVE"]
        print(f"  Among the {decisive} pairs where exactly one phase had GT:")
        print(f"    Constructive: {c}/{decisive} ({c / decisive * 100:.1f}%)")
        print(f"    Destructive:  {d}/{decisive} ({d / decisive * 100:.1f}%)")
    print()


def _gt_convergence(results: list[dict], gt: dict) -> None:
    _section("GT-BASED FINAL CONVERGENCE (plurality ≥3/5 on final picks)")
    print()
    print("  Final picks = country_assessments on Path B, initial assessments")
    print("  on Path A (since Path A has no CA stage).")
    print()
    n = plur_correct = plur_wrong = split = 0
    unan_correct = unan_wrong = 0
    by_path = {"A": [0, 0, 0, 0], "B": [0, 0, 0, 0]}  # plur_ok, plur_x, split, n
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        top, cnt, total = _final_plurality(r)
        if total == 0:
            continue
        n += 1
        path = _path(r)
        if path in by_path:
            by_path[path][3] += 1
        if cnt >= 3:
            ok = _matches(top, truth["country_code"])
            if ok:
                plur_correct += 1
                if cnt == len(AGENT_NAMES):
                    unan_correct += 1
                if path in by_path:
                    by_path[path][0] += 1
            else:
                plur_wrong += 1
                if cnt == len(AGENT_NAMES):
                    unan_wrong += 1
                if path in by_path:
                    by_path[path][1] += 1
        else:
            split += 1
            if path in by_path:
                by_path[path][2] += 1

    if n == 0:
        print("  (no images with final picks and GT)")
        print()
        return

    print(f"  Images with final picks and GT: {n}")
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
    print(f"  Split by path:")
    print(f"    {'Path':<6} {'N':>5} {'PlurOK':>8} {'PlurWrong':>11} "
          f"{'Split':>7} {'OK %':>7}")
    print(f"    {'─' * 6} {'─' * 5} {'─' * 8} {'─' * 11} {'─' * 7} {'─' * 7}")
    for path in ("A", "B"):
        ok, x, sp, total = by_path[path]
        if total == 0:
            continue
        print(f"    {path:<6} {total:>5} {ok:>8} {x:>11} {sp:>7} "
              f"{ok / total * 100:>6.1f}%")
    print()


def _per_agent_shift_matrix(results: list[dict], gt: dict) -> None:
    _section("PER-AGENT INITIAL->CA SHIFT MATRIX (Path B only)")
    print()
    print(f"  {'Agent':<12} {'N':>5} {'Constr':>7} {'Destr':>6} {'StayOK':>7} "
          f"{'StayX':>6} {'WrongShift':>11} {'NetTruth':>9}")
    print(f"  {'─' * 12} {'─' * 5} {'─' * 7} {'─' * 6} {'─' * 7} "
          f"{'─' * 6} {'─' * 11} {'─' * 9}")
    for agent in AGENT_NAMES:
        n = c = d = sok = sx = ws = 0
        for r in results:
            if _path(r) != "B":
                continue
            truth = gt.get(r["_name"])
            if not truth:
                continue
            ic = _initial_pick(r, agent)
            cc = _ca_pick(r, agent)
            if not ic or not cc:
                continue
            n += 1
            cls = _classify_shift(ic, cc, truth["country_code"])
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
            print(f"  {agent:<12} {0:>5}  (no Path-B agent-images with GT)")
            continue
        net = c - d
        print(f"  {agent:<12} {n:>5} {c:>7} {d:>6} {sok:>7} {sx:>6} {ws:>11} "
              f"{net:>+9}")
    print()
    print("  Constr     = initial wrong, CA pick moved onto GT")
    print("  Destr      = initial correct, CA pick moved away from GT")
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
    print("  Final acc = CA pick on Path B, initial pick on Path A")
    print("  Δ = Final - Init in percentage points (positive = re-assessment helped)")
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
    print(f"  Images with parsed prediction coordinates: {n}/{len(results)}")
    print()
    print(f"  Mean Δlat:           {_mean(dlat):+.2f}°")
    print(f"  Mean N-S offset:     {_mean(ns_km):+.0f} km   "
          f"(median {_median(ns_km):+.0f} km)")
    print(f"    └ predictions tend {'NORTH' if _mean(ns_km) >= 0 else 'SOUTH'} of GT on average")
    print()
    print(f"  Mean Δlng:           {_mean(dlng):+.2f}°")
    print(f"  Mean W-E offset:     {_mean(we_km):+.0f} km   "
          f"(median {_median(we_km):+.0f} km)")
    print(f"    └ predictions tend {'EAST' if _mean(we_km) >= 0 else 'WEST'} of GT on average")
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
        c: (centroid_acc[c][0] / centroid_cnt[c],
            centroid_acc[c][1] / centroid_cnt[c])
        for c in centroid_acc
    }

    def _pick_centroid(country: str | None):
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
    print("  Initial = each agent's pre-narrowing top pick.")
    print("  Final   = CA top pick on Path B, initial on Path A.")
    print()


# ── Machine-readable dynamics (for the consolidated report) ────────────────


def compute_dynamics(results_dir: Path, gt_path: Path | None, out_dir: Path) -> dict:
    """Compute structured Progressive Narrowing dynamics and write dynamics_metrics.json.

    This is the machine-readable counterpart to ``analyze()``'s printed GT
    pipeline analysis, consumed by the ``report`` step so the single
    per-approach report can carry the PN dynamics. Reuses the existing
    helper functions in this module wherever possible.

    Content: Path A / Path B split, region-consensus rate, the region and
    country narrowing funnel (region accuracy split by path, initial -> CA
    shift outcomes on Path B).
    """
    results = _load_results(Path(results_dir))
    gt = _load_ground_truth(Path(gt_path)) if gt_path else {}

    n_total = len(results)

    # ── Path A / Path B split + region-consensus rate ──────────────────────
    paths = Counter(_path(r) for r in results)
    n_path_a = paths.get("A", 0)
    n_path_b = paths.get("B", 0)
    n_consensus = sum(
        1 for r in results
        if (r.get("progressive_narrowing", {}) or {}).get("region_consensus")
    )

    path_split = {
        "n_total": n_total,
        "n_path_a": n_path_a,
        "n_path_b": n_path_b,
        "path_a_rate": (n_path_a / n_total) if n_total else None,
        "path_b_rate": (n_path_b / n_total) if n_total else None,
        "region_consensus": n_consensus,
        "region_consensus_rate": (n_consensus / n_total) if n_total else None,
    }

    # ── Region narrowing funnel: does confirmed_region contain the GT? ─────
    reg_n = reg_ok = 0
    reg_by_path = {"A": [0, 0], "B": [0, 0]}  # [hits, total]
    for r in results:
        truth = gt.get(r["_name"])
        if not truth:
            continue
        region = _confirmed_region(r)
        if not region:
            continue
        gt_region = _gt_region(truth["country_code"])
        if gt_region is None:
            continue
        reg_n += 1
        path = _path(r)
        if path in reg_by_path:
            reg_by_path[path][1] += 1
        if region == gt_region:
            reg_ok += 1
            if path in reg_by_path:
                reg_by_path[path][0] += 1

    def _rate(hits: int, total: int) -> float | None:
        return (hits / total) if total else None

    region_funnel = {
        "n": reg_n,
        "n_match": reg_ok,
        "match_rate": _rate(reg_ok, reg_n),
        "path_a": {
            "n": reg_by_path["A"][1],
            "n_match": reg_by_path["A"][0],
            "match_rate": _rate(reg_by_path["A"][0], reg_by_path["A"][1]),
        },
        "path_b": {
            "n": reg_by_path["B"][1],
            "n_match": reg_by_path["B"][0],
            "match_rate": _rate(reg_by_path["B"][0], reg_by_path["B"][1]),
        },
    }

    # ── Country narrowing funnel: initial -> CA shift outcomes (Path B) ────
    shift_counts: Counter = Counter()
    n_shift = 0
    for r in results:
        if _path(r) != "B":
            continue
        truth = gt.get(r["_name"])
        if not truth:
            continue
        gt_code = truth["country_code"]
        for agent in AGENT_NAMES:
            ic = _initial_pick(r, agent)
            cc = _ca_pick(r, agent)
            if not ic or not cc:
                continue
            n_shift += 1
            shift_counts[_classify_shift(ic, cc, gt_code)] += 1

    country_funnel = {
        "n_agent_images": n_shift,
        "counts": dict(shift_counts),
    }

    metrics = {
        "n_total": n_total,
        "path_split": path_split,
        "region_funnel": region_funnel,
        "country_funnel": country_funnel,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dynamics_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[dynamics] wrote {out_file}")
    print(f"[dynamics] path_a={n_path_a} path_b={n_path_b} "
          f"region_consensus={n_consensus} "
          f"region_match={reg_ok}/{reg_n} "
          f"shift_agent_images={n_shift}")
    return metrics


# ── Driver ───────────────────────────────────────────────────────────────


def analyze(results_dir: Path, gt_path: Path | None = None) -> None:
    results = _load_results(results_dir)
    if not results:
        print("No results found.")
        return

    _overview(results)
    _per_agent_initial_picks(results)
    _region_voting_behavior(results)
    _hypothesis_stance_dist(results)
    _agreement_dynamics(results)
    _judge_source(results)
    _timing(results)

    if gt_path:
        gt = _load_ground_truth(gt_path)
        print("=" * 70)
        print("GROUND-TRUTH-BASED PROGRESSIVE NARROWING ANALYSIS")
        print("=" * 70)
        print()
        _gt_region_accuracy(results, gt)
        _gt_hypothesis_quality(results, gt)
        _gt_shift_outcomes(results, gt)
        _gt_convergence(results, gt)
        _per_agent_shift_matrix(results, gt)
        _per_agent_accuracy_delta(results, gt)
        _geographic_bias(results, gt)
        _per_agent_geographic_bias(results, gt)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze the Progressive Narrowing architecture"
    )
    parser.add_argument("results_dir", help="Directory with result.json files")
    parser.add_argument("ground_truth", nargs="?", default=None,
                        help="Optional path to georc_locations.csv")
    args = parser.parse_args()
    gt_path = Path(args.ground_truth) if args.ground_truth else None
    analyze(Path(args.results_dir), gt_path)


if __name__ == "__main__":
    main()
