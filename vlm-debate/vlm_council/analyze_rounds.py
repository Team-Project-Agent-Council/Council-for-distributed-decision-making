"""Analyze debate dynamics in VLM Council results.

Shows: debate frequency, revision rates, convergence patterns,
which agents revised most, and how debate affected the final outcome.

When a ground-truth CSV is provided as a second argument, also computes
constructive vs destructive debate dynamics, GT-based convergence, and
per-agent win/loss matrix.

Usage:
    python -m vlm_council.analyze_rounds results_debate/
    python -m vlm_council.analyze_rounds results_debate/ Images/georc_locations.csv
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

from vlm_council.evaluate import _countries_match, _load_ground_truth


AGENT_NAMES = ["linguistic", "landscape", "botanics", "regulatory", "meta"]


def _agent_correct(country: str, gt_code: str) -> bool:
    """True iff the country string matches the GT country code (strict, with aliases)."""
    if not country or not gt_code:
        return False
    return _countries_match(country, gt_code)


def _last_position_in_pairing(pairing: dict, agent_name: str) -> str:
    """Return the agent's final position in this pairing (last message), or initial if no message."""
    for ex in reversed(pairing.get("exchanges", [])):
        if ex.get("agent_name") == agent_name:
            return (ex.get("position") or "").strip()
    if agent_name == pairing.get("agent_a"):
        return (pairing.get("agent_a_initial_position") or "").strip()
    if agent_name == pairing.get("agent_b"):
        return (pairing.get("agent_b_initial_position") or "").strip()
    return ""


def _classify_pairing(pairing: dict, gt_code: str) -> dict:
    """Classify a single pairing relative to ground truth.

    Returns dict with:
        category: one of CONSTRUCTIVE, DESTRUCTIVE, STAND_CORRECT,
                  STAND_WRONG, BOTH_WRONG_NEUTRAL, BOTH_CORRECT
        agent_a, agent_b, init_a_correct, init_b_correct,
        end_a_correct, end_b_correct,
        winner: agent that "convinced" (or None)
        loser: agent that revised (or None)
    """
    a = pairing.get("agent_a", "")
    b = pairing.get("agent_b", "")
    init_a = (pairing.get("agent_a_initial_position") or "").strip()
    init_b = (pairing.get("agent_b_initial_position") or "").strip()
    end_a = _last_position_in_pairing(pairing, a)
    end_b = _last_position_in_pairing(pairing, b)

    init_a_correct = _agent_correct(init_a, gt_code)
    init_b_correct = _agent_correct(init_b, gt_code)
    end_a_correct = _agent_correct(end_a, gt_code)
    end_b_correct = _agent_correct(end_b, gt_code)

    a_revised = end_a.lower() != init_a.lower() and end_a != ""
    b_revised = end_b.lower() != init_b.lower() and end_b != ""

    info = {
        "agent_a": a,
        "agent_b": b,
        "init_a": init_a,
        "init_b": init_b,
        "end_a": end_a,
        "end_b": end_b,
        "init_a_correct": init_a_correct,
        "init_b_correct": init_b_correct,
        "end_a_correct": end_a_correct,
        "end_b_correct": end_b_correct,
        "a_revised": a_revised,
        "b_revised": b_revised,
        "category": "",
        "winner": None,
        "loser": None,
    }

    # Both initially correct (rare, only if pairing fired despite agreement on GT)
    if init_a_correct and init_b_correct:
        info["category"] = "BOTH_CORRECT"
        return info

    # Both initially wrong -> no truth-bearer in the pairing
    if not init_a_correct and not init_b_correct:
        info["category"] = "BOTH_WRONG_NEUTRAL"
        return info

    # Exactly one initially correct
    correct_agent = a if init_a_correct else b
    wrong_agent = b if init_a_correct else a
    correct_revised = a_revised if init_a_correct else b_revised
    wrong_revised = b_revised if init_a_correct else a_revised
    correct_end_correct = end_a_correct if init_a_correct else end_b_correct
    wrong_end_correct = end_b_correct if init_a_correct else end_a_correct

    if wrong_revised and wrong_end_correct and not (correct_revised and not correct_end_correct):
        # Truth-bearer convinced the wrong one to land on GT
        info["category"] = "CONSTRUCTIVE"
        info["winner"] = correct_agent
        info["loser"] = wrong_agent
        return info

    if correct_revised and not correct_end_correct:
        # Truth-bearer abandoned the truth -> destructive pull from the wrong one
        info["category"] = "DESTRUCTIVE"
        info["winner"] = wrong_agent
        info["loser"] = correct_agent
        return info

    # Truth-bearer held the line; wrong agent did not move to GT
    info["category"] = "STAND_CORRECT"
    return info


def analyze(results_dir: Path, gt_path: Path | None = None) -> None:
    results = []
    name_to_data: dict[str, dict] = {}
    for img_dir in sorted(results_dir.iterdir()):
        result_file = img_dir / "result.json"
        if not result_file.exists():
            continue
        try:
            with open(result_file) as f:
                data = json.load(f)
            if not data.get("error"):
                results.append(data)
                name_to_data[img_dir.name] = data
        except (json.JSONDecodeError, OSError):
            continue

    gt = _load_ground_truth(gt_path) if gt_path else {}

    total = len(results)
    if total == 0:
        print("No results found.")
        return

    print("=" * 70)
    print("VLM Council, Debate Analysis")
    print("=" * 70)
    print(f"Total images: {total}")
    print()

    # Overall debate statistics
    debate_counts = []
    exchange_counts = []
    termination_reasons = Counter()
    images_no_debate = 0
    images_with_debate = 0
    total_revisions = 0
    total_exchanges = 0

    for r in results:
        debate = r.get("debate", {})
        n_rounds = debate.get("total_rounds", 0)
        pairings = debate.get("pairings", [])
        moderator_decisions = debate.get("moderator_decisions", [])

        if not pairings:
            images_no_debate += 1
        else:
            images_with_debate += 1

        debate_counts.append(len(pairings))

        for pairing in pairings:
            exchanges = pairing.get("exchanges", [])
            exchange_counts.append(len(exchanges))
            total_exchanges += len(exchanges)
            for ex in exchanges:
                if ex.get("revised"):
                    total_revisions += 1

        if moderator_decisions:
            last = moderator_decisions[-1]
            reason = last.get("termination_reason", "unknown")
            if reason:
                termination_reasons[reason] += 1

    print("-" * 70)
    print("  DEBATE OVERVIEW")
    print("-" * 70)
    print(f"  Images with no debate (consensus after R1): {images_no_debate}/{total} "
          f"({images_no_debate / total * 100:.0f}%)")
    print(f"  Images with debate:                         {images_with_debate}/{total} "
          f"({images_with_debate / total * 100:.0f}%)")
    print()

    if images_with_debate > 0:
        avg_pairings = sum(debate_counts) / total
        max_pairings = max(debate_counts)
        print(f"  Avg pairings per image: {avg_pairings:.1f}")
        print(f"  Max pairings in one image: {max_pairings}")
        print()

    if exchange_counts:
        avg_exchanges = sum(exchange_counts) / len(exchange_counts)
        print(f"  Total debate exchanges: {total_exchanges}")
        print(f"  Avg exchanges per pairing: {avg_exchanges:.1f}")
        print(f"  Total revisions: {total_revisions}")
        if total_exchanges > 0:
            print(f"  Revision rate: {total_revisions / total_exchanges * 100:.1f}% of exchanges")
    print()

    if termination_reasons:
        print(f"  Termination reasons:")
        for reason, count in termination_reasons.most_common():
            print(f"    {reason:20s}: {count} ({count / total * 100:.0f}%)")
    print()

    # Per-agent revision analysis
    print("-" * 70)
    print("  PER-AGENT DEBATE BEHAVIOR")
    print("-" * 70)

    agent_debates = Counter()
    agent_revisions = Counter()
    agent_defended = Counter()
    agent_exchanges = Counter()

    for r in results:
        debate = r.get("debate", {})
        for pairing in debate.get("pairings", []):
            agents_in = {pairing.get("agent_a"), pairing.get("agent_b")}
            for agent in agents_in:
                if agent in AGENT_NAMES:
                    agent_debates[agent] += 1

            for ex in pairing.get("exchanges", []):
                name = ex.get("agent_name", "")
                if name in AGENT_NAMES:
                    agent_exchanges[name] += 1
                    if ex.get("revised"):
                        agent_revisions[name] += 1
                    else:
                        agent_defended[name] += 1

    print()
    print(f"  {'Agent':<12s} {'Debates':<9s} {'Exchanges':<11s} {'Revised':<10s} {'Defended':<10s} {'Revision %'}")
    print(f"  {'─' * 12} {'─' * 9} {'─' * 11} {'─' * 10} {'─' * 10} {'─' * 10}")
    for agent in AGENT_NAMES:
        n_debates = agent_debates[agent]
        n_exchanges = agent_exchanges[agent]
        n_revised = agent_revisions[agent]
        n_defended = agent_defended[agent]
        rev_pct = f"{n_revised / n_exchanges * 100:.0f}%" if n_exchanges > 0 else ", "
        print(f"  {agent:<12s} {n_debates:<9d} {n_exchanges:<11d} {n_revised:<10d} {n_defended:<10d} {rev_pct}")
    print()

    # Pairing frequency
    print("-" * 70)
    print("  PAIRING FREQUENCY (which agents debated most)")
    print("-" * 70)

    pairing_counter = Counter()
    for r in results:
        debate = r.get("debate", {})
        for pairing in debate.get("pairings", []):
            a = pairing.get("agent_a", "?")
            b = pairing.get("agent_b", "?")
            key = tuple(sorted([a, b]))
            pairing_counter[key] += 1

    print()
    if pairing_counter:
        for (a, b), count in pairing_counter.most_common(10):
            print(f"  {a} vs {b}: {count} debates")
    else:
        print("  (no debates)")
    print()

    # Convergence analysis
    print("-" * 70)
    print("  CONVERGENCE ANALYSIS")
    print("-" * 70)
    print()

    converged_after_debate = 0
    still_disagreed = 0

    for r in results:
        debate = r.get("debate", {})
        pairings = debate.get("pairings", [])
        if not pairings:
            continue

        final_positions = set()
        for pairing in pairings:
            exchanges = pairing.get("exchanges", [])
            if exchanges:
                last_a = None
                last_b = None
                agent_a = pairing.get("agent_a")
                agent_b = pairing.get("agent_b")
                for ex in reversed(exchanges):
                    if ex.get("agent_name") == agent_a and last_a is None:
                        last_a = ex.get("position", "")
                    elif ex.get("agent_name") == agent_b and last_b is None:
                        last_b = ex.get("position", "")
                    if last_a and last_b:
                        break
                if last_a:
                    final_positions.add(last_a.lower().strip())
                if last_b:
                    final_positions.add(last_b.lower().strip())

        if len(final_positions) <= 1:
            converged_after_debate += 1
        else:
            still_disagreed += 1

    if images_with_debate > 0:
        print(f"  Debating agents converged:    {converged_after_debate}/{images_with_debate} "
              f"({converged_after_debate / images_with_debate * 100:.0f}%)")
        print(f"  Still disagreed after debate: {still_disagreed}/{images_with_debate} "
              f"({still_disagreed / images_with_debate * 100:.0f}%)")
    else:
        print("  (no debates occurred)")
    print()

    # Round 1 agreement vs debate outcome
    print("-" * 70)
    print("  ROUND 1 AGREEMENT vs FINAL RESULT")
    print("-" * 70)
    print()

    r1_majority_correct = 0
    r1_majority_wrong_debate_fixed = 0
    r1_majority_wrong_debate_didnt_fix = 0

    for r in results:
        r1 = r.get("round_1_assessments", {})
        country_result = r.get("country_result", "")
        if not country_result:
            continue

        # Extract final country from "Country: X\nCoordinates..." format
        final_country = ""
        for line in country_result.split("\n"):
            if line.strip().lower().startswith("country:"):
                final_country = line.split(":", 1)[1].strip()
                break

        if not final_country:
            final_country = country_result.split("\n")[0].strip()

        # Get Round 1 majority
        r1_countries = Counter()
        for agent in AGENT_NAMES:
            assessment = r1.get(agent, {})
            candidates = assessment.get("candidates", [])
            if candidates:
                top = candidates[0].get("country", "").strip()
                if top:
                    r1_countries[top.lower()] += 1

        if r1_countries:
            majority_country, majority_count = r1_countries.most_common(1)[0]
            if majority_country == final_country.lower():
                r1_majority_correct += 1

    print(f"  Final result matches Round 1 majority: {r1_majority_correct}/{total}")
    print()

    # Timing
    print("-" * 70)
    print("  TIMING")
    print("-" * 70)
    print()

    times = [r.get("timing", {}).get("total_seconds", 0) for r in results if r.get("timing")]
    debate_times = [r.get("timing", {}).get("total_seconds", 0)
                    for r in results if r.get("debate", {}).get("pairings")]
    no_debate_times = [r.get("timing", {}).get("total_seconds", 0)
                       for r in results if not r.get("debate", {}).get("pairings")]

    if times:
        print(f"  Overall:     avg {sum(times) / len(times):.1f}s, "
              f"median {sorted(times)[len(times) // 2]:.1f}s")
    if debate_times:
        print(f"  With debate: avg {sum(debate_times) / len(debate_times):.1f}s, "
              f"median {sorted(debate_times)[len(debate_times) // 2]:.1f}s")
    if no_debate_times:
        print(f"  No debate:   avg {sum(no_debate_times) / len(no_debate_times):.1f}s, "
              f"median {sorted(no_debate_times)[len(no_debate_times) // 2]:.1f}s")
    print()

    if gt:
        _gt_analysis(name_to_data, gt)


def _gt_analysis(name_to_data: dict[str, dict], gt: dict) -> None:
    """Ground-truth-based debate analysis: constructive/destructive, convergence, per-agent."""
    print("=" * 70)
    print("GROUND-TRUTH-BASED DEBATE ANALYSIS")
    print("=" * 70)
    print()

    # Collect classified pairings
    pairing_records: list[dict] = []  # one per pairing across all images
    images_with_debate = 0
    images_without_gt = 0

    for name, data in name_to_data.items():
        truth = gt.get(name)
        if not truth:
            images_without_gt += 1
            continue
        gt_code = truth["country_code"]

        pairings = data.get("debate", {}).get("pairings", [])
        if not pairings:
            continue
        images_with_debate += 1

        for p in pairings:
            cls = _classify_pairing(p, gt_code)
            cls["image"] = name
            cls["gt_code"] = gt_code
            cls["gt_name"] = truth["country_name"]
            pairing_records.append(cls)

    if images_without_gt:
        print(f"  (skipped {images_without_gt} images without ground truth)")
        print()

    if not pairing_records:
        print("  No debating pairings to analyze against ground truth.")
        return

    # ── Section 1: Constructive vs destructive overview ───────────────────
    print("-" * 70)
    print("  CONSTRUCTIVE vs DESTRUCTIVE DEBATE OUTCOMES (per pairing)")
    print("-" * 70)
    print()

    cat_counter = Counter(r["category"] for r in pairing_records)
    total_p = len(pairing_records)

    def pct(n):
        return f"{n / total_p * 100:.1f}%" if total_p else ", "

    constructive = cat_counter["CONSTRUCTIVE"]
    destructive = cat_counter["DESTRUCTIVE"]
    stand_correct = cat_counter["STAND_CORRECT"]
    both_wrong = cat_counter["BOTH_WRONG_NEUTRAL"]
    both_correct = cat_counter["BOTH_CORRECT"]

    print(f"  Total pairings analysed:     {total_p}")
    print(f"  CONSTRUCTIVE  (truth wins):  {constructive}  ({pct(constructive)})")
    print(f"    └ correct agent convinced wrong agent to land on GT")
    print(f"  DESTRUCTIVE   (truth loses): {destructive}  ({pct(destructive)})")
    print(f"    └ wrong agent pulled correct agent away from GT")
    print(f"  STAND_CORRECT (truth holds): {stand_correct}  ({pct(stand_correct)})")
    print(f"    └ correct agent held position, wrong agent did not move to GT")
    print(f"  BOTH_WRONG_NEUTRAL:          {both_wrong}  ({pct(both_wrong)})")
    print(f"    └ neither agent had GT initially; debate cannot be constructive")
    print(f"  BOTH_CORRECT (rare):         {both_correct}  ({pct(both_correct)})")
    print()

    pairings_with_truth_bearer = constructive + destructive + stand_correct
    if pairings_with_truth_bearer:
        print(f"  Among the {pairings_with_truth_bearer} pairings where exactly one agent had the GT initially:")
        print(f"    Constructive: {constructive}/{pairings_with_truth_bearer} "
              f"({constructive / pairings_with_truth_bearer * 100:.1f}%)")
        print(f"    Destructive:  {destructive}/{pairings_with_truth_bearer} "
              f"({destructive / pairings_with_truth_bearer * 100:.1f}%)")
        print(f"    Stand:        {stand_correct}/{pairings_with_truth_bearer} "
              f"({stand_correct / pairings_with_truth_bearer * 100:.1f}%)")
    print()

    # ── Section 2: GT-based convergence per image ─────────────────────────
    print("-" * 70)
    print("  GT-BASED CONVERGENCE (per image with debate)")
    print("-" * 70)
    print()

    converged_correct = 0
    converged_wrong = 0
    not_converged = 0
    convergence_examples_correct: list[str] = []
    convergence_examples_wrong: list[str] = []

    # Per-agent convergence tracking: for each image, did this agent participate, and what was the outcome
    per_agent_conv: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}

    for name, data in name_to_data.items():
        truth = gt.get(name)
        if not truth:
            continue
        pairings = data.get("debate", {}).get("pairings", [])
        if not pairings:
            continue

        gt_code = truth["country_code"]
        # Final position of every debating agent across all pairings (last message wins)
        debating_agents: dict[str, str] = {}
        for p in pairings:
            for agent_name in (p.get("agent_a"), p.get("agent_b")):
                if not agent_name:
                    continue
                last = _last_position_in_pairing(p, agent_name)
                if last:
                    debating_agents[agent_name] = last

        if not debating_agents:
            continue

        positions_lower = {pos.lower().strip() for pos in debating_agents.values() if pos}
        if len(positions_lower) > 1:
            outcome = "not_converged"
            not_converged += 1
        else:
            only_pos = next(iter(positions_lower))
            if _agent_correct(only_pos, gt_code):
                outcome = "converged_correct"
                converged_correct += 1
                if len(convergence_examples_correct) < 5:
                    convergence_examples_correct.append(f"{name} -> {only_pos} (GT: {truth['country_name']})")
            else:
                outcome = "converged_wrong"
                converged_wrong += 1
                if len(convergence_examples_wrong) < 5:
                    convergence_examples_wrong.append(f"{name} -> {only_pos} (GT: {truth['country_name']})")

        # Per-agent participation tally
        for agent_name, end_pos in debating_agents.items():
            if agent_name not in per_agent_conv:
                continue
            per_agent_conv[agent_name][outcome] += 1
            per_agent_conv[agent_name]["total"] += 1
            if _agent_correct(end_pos, gt_code):
                per_agent_conv[agent_name]["end_correct"] += 1

    total_debating = converged_correct + converged_wrong + not_converged
    if total_debating:
        print(f"  Images with debate: {total_debating}")
        print(f"    Converged to GT (correct):    {converged_correct}/{total_debating} "
              f"({converged_correct / total_debating * 100:.1f}%)")
        print(f"    Converged but wrong:          {converged_wrong}/{total_debating} "
              f"({converged_wrong / total_debating * 100:.1f}%)")
        print(f"    Did not converge:             {not_converged}/{total_debating} "
              f"({not_converged / total_debating * 100:.1f}%)")
        total_converged = converged_correct + converged_wrong
        if total_converged:
            print(f"    Of converged: {converged_correct}/{total_converged} "
                  f"({converged_correct / total_converged * 100:.1f}%) landed on GT")
        print()

        if convergence_examples_correct:
            print("  Examples (correct convergence):")
            for ex in convergence_examples_correct:
                print(f"    {ex}")
            print()
        if convergence_examples_wrong:
            print("  Examples (wrong convergence):")
            for ex in convergence_examples_wrong:
                print(f"    {ex}")
            print()

    # ── Section 2b: Per-agent convergence participation ───────────────────
    print("-" * 70)
    print("  PER-AGENT CONVERGENCE PARTICIPATION")
    print("-" * 70)
    print()
    print("  Across images where this agent participated in debate, how often did the")
    print("  council converge correctly, wrongly, or not at all? And how often did the")
    print("  agent's OWN end-position match the ground truth?")
    print()
    print(f"  {'Agent':<12} {'Debated':>8} {'ConvOK':>7} {'ConvX':>6} {'NoConv':>7} {'EndOK':>6} {'EndOK%':>8}")
    print(f"  {'─' * 12} {'─' * 8} {'─' * 7} {'─' * 6} {'─' * 7} {'─' * 6} {'─' * 8}")
    for agent in AGENT_NAMES:
        c = per_agent_conv[agent]
        total = c["total"]
        if total == 0:
            print(f"  {agent:<12} {0:>8} {0:>7} {0:>6} {0:>7} {0:>6} {', ':>8}")
            continue
        end_ok_pct = f"{c['end_correct'] / total * 100:.1f}%"
        print(f"  {agent:<12} {total:>8} {c['converged_correct']:>7} "
              f"{c['converged_wrong']:>6} {c['not_converged']:>7} "
              f"{c['end_correct']:>6} {end_ok_pct:>8}")
    print()
    print("  Debated  = images with debate where this agent participated")
    print("  ConvOK   = images where the council converged on the GT")
    print("  ConvX    = images where the council converged but on a wrong country")
    print("  NoConv   = images where debating agents stayed split")
    print("  EndOK    = images where this agent's own end-position equals the GT")
    print()

    # ── Section 3: Per-agent win/loss matrix ──────────────────────────────
    print("-" * 70)
    print("  PER-AGENT WIN/LOSS MATRIX (across all pairings)")
    print("-" * 70)
    print()
    print("  Wins/Losses are counted only in pairings with exactly one initially-correct agent:")
    print("    constructive_win  = was correct, convinced wrong opponent to land on GT")
    print("    destructive_win   = was wrong,   pulled correct opponent away from GT")
    print("    constructive_loss = was wrong,   was convinced by opponent to land on GT (good for the council)")
    print("    destructive_loss  = was correct, was pulled away from GT (bad for the council)")
    print("    stand_correct     = was correct, held position; opponent did not move to GT")
    print("    stand_wrong       = was wrong,   held position vs a correct opponent")
    print()

    per_agent: dict[str, Counter] = {a: Counter() for a in AGENT_NAMES}

    for r in pairing_records:
        cat = r["category"]
        a = r["agent_a"]
        b = r["agent_b"]

        if cat == "CONSTRUCTIVE":
            winner, loser = r["winner"], r["loser"]
            if winner in per_agent:
                per_agent[winner]["constructive_win"] += 1
            if loser in per_agent:
                per_agent[loser]["constructive_loss"] += 1
        elif cat == "DESTRUCTIVE":
            winner, loser = r["winner"], r["loser"]
            if winner in per_agent:
                per_agent[winner]["destructive_win"] += 1
            if loser in per_agent:
                per_agent[loser]["destructive_loss"] += 1
        elif cat == "STAND_CORRECT":
            # Whichever agent was initially correct held the line; the other stood wrong.
            correct = a if r["init_a_correct"] else b
            wrong = b if r["init_a_correct"] else a
            if correct in per_agent:
                per_agent[correct]["stand_correct"] += 1
            if wrong in per_agent:
                per_agent[wrong]["stand_wrong"] += 1
        # BOTH_WRONG_NEUTRAL and BOTH_CORRECT do not contribute to per-agent win/loss

    header = (f"  {'Agent':<12} {'C-Win':>6} {'C-Loss':>7} {'D-Win':>6} {'D-Loss':>7} "
              f"{'StandOK':>8} {'StandX':>7} {'NetTruth':>9}")
    print(header)
    print(f"  {'─' * 12} {'─' * 6} {'─' * 7} {'─' * 6} {'─' * 7} {'─' * 8} {'─' * 7} {'─' * 9}")
    for agent in AGENT_NAMES:
        c = per_agent[agent]
        cw = c["constructive_win"]
        cl = c["constructive_loss"]
        dw = c["destructive_win"]
        dl = c["destructive_loss"]
        sc = c["stand_correct"]
        sw = c["stand_wrong"]
        # Net contribution to truth: +1 if the agent helped truth, -1 if hurt truth
        net = (cw + cl + sc) - (dw + dl + sw)
        print(f"  {agent:<12} {cw:>6} {cl:>7} {dw:>6} {dl:>7} {sc:>8} {sw:>7} {net:>+9}")
    print()
    print("  NetTruth = (C-Win + C-Loss + StandOK) - (D-Win + D-Loss + StandX)")
    print("             positive -> agent's debate behavior pushed council toward GT")
    print("             negative -> agent's debate behavior pushed council away from GT")
    print()

    # ── Section 4: Constructive/destructive by debating-pair ──────────────
    print("-" * 70)
    print("  PAIR-SPECIFIC OUTCOMES (top pairings)")
    print("-" * 70)
    print()
    pair_stats: dict[tuple[str, str], Counter] = {}
    for r in pairing_records:
        key = tuple(sorted([r["agent_a"], r["agent_b"]]))
        pair_stats.setdefault(key, Counter())[r["category"]] += 1

    print(f"  {'Pair':<32} {'Total':>6} {'Constr':>7} {'Destr':>6} {'Stand':>6} {'BothW':>6}")
    print(f"  {'─' * 32} {'─' * 6} {'─' * 7} {'─' * 6} {'─' * 6} {'─' * 6}")
    for key, c in sorted(pair_stats.items(), key=lambda kv: -sum(kv[1].values())):
        a, b = key
        total = sum(c.values())
        print(f"  {a + ' vs ' + b:<32} {total:>6} {c['CONSTRUCTIVE']:>7} "
              f"{c['DESTRUCTIVE']:>6} {c['STAND_CORRECT']:>6} {c['BOTH_WRONG_NEUTRAL']:>6}")
    print()

    # ── Geographic bias (prediction vs. GT) ─────────────────────────────
    from vlm_council.evaluate import _extract_coordinates

    print("-" * 70)
    print("  GEOGRAPHIC BIAS (prediction vs. ground truth)")
    print("-" * 70)
    print()

    KM_PER_DEG = 111.0
    dlat_deg: list[float] = []
    dlng_deg: list[float] = []
    ns_km: list[float] = []
    we_km: list[float] = []

    for name, data in name_to_data.items():
        truth = gt.get(name)
        if not truth:
            continue
        pred = _extract_coordinates(data.get("country_result", ""))
        if not pred:
            continue
        gt_la, gt_lo = truth["lat"], truth["lng"]
        pred_la, pred_lo = pred[0], pred[1]
        dlo = pred_lo - gt_lo
        if dlo > 180:
            dlo -= 360
        elif dlo < -180:
            dlo += 360
        dlat_deg.append(pred_la - gt_la)
        dlng_deg.append(dlo)
        ns_km.append((pred_la - gt_la) * KM_PER_DEG)
        we_km.append(dlo * KM_PER_DEG * math.cos(math.radians(gt_la)))

    n_geo = len(dlat_deg)
    if n_geo == 0:
        print("  (no images with parsed prediction coordinates)")
        print()
    else:
        def _mean(xs): return sum(xs) / len(xs)
        def _median(xs):
            s = sorted(xs)
            m = len(s) // 2
            return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])

        mean_dlat = _mean(dlat_deg)
        mean_dlng = _mean(dlng_deg)
        mean_ns, med_ns = _mean(ns_km), _median(ns_km)
        mean_we, med_we = _mean(we_km), _median(we_km)
        ns_label = "north" if mean_ns >= 0 else "south"
        we_label = "east" if mean_we >= 0 else "west"

        print(f"  Images with parsed prediction coordinates: {n_geo}/{len(name_to_data)}")
        print()
        print(f"  Mean Δlat:           {mean_dlat:+.2f}°")
        print(f"  Mean N-S offset:     {mean_ns:+.0f} km   (median {med_ns:+.0f} km)")
        print(f"    └ predictions tend {ns_label.upper()} of GT on average")
        print()
        print(f"  Mean Δlng:           {mean_dlng:+.2f}°")
        print(f"  Mean W-E offset:     {mean_we:+.0f} km   (median {med_we:+.0f} km)")
        print(f"    └ predictions tend {we_label.upper()} of GT on average")
        print()
        print(f"  Sign convention: positive Δlat = prediction north of GT;")
        print(f"                   positive Δlng = prediction east of GT.")
        print(f"  W-E km uses cos(lat_GT) so high-latitude shifts are not over-counted.")
        print()

    # ── Per-agent geographic bias ───────────────────────────────────────
    from vlm_council.evaluate import _NAME_TO_CODE, _normalize_country

    print("-" * 70)
    print("  PER-AGENT GEOGRAPHIC BIAS (Round-1 top pick -> country centroid)")
    print("-" * 70)
    print()

    # Country-code -> centroid (mean lat/lng across all GT entries of that code).
    centroid_acc: dict[str, list[float]] = {}
    centroid_cnt: dict[str, int] = {}
    for entry in gt.values():
        code = entry["country_code"]
        if code not in centroid_acc:
            centroid_acc[code] = [0.0, 0.0]
            centroid_cnt[code] = 0
        centroid_acc[code][0] += entry["lat"]
        centroid_acc[code][1] += entry["lng"]
        centroid_cnt[code] += 1
    centroids: dict[str, tuple[float, float]] = {
        c: (centroid_acc[c][0] / centroid_cnt[c], centroid_acc[c][1] / centroid_cnt[c])
        for c in centroid_acc
    }

    def _agent_pick_centroid(country_str: str) -> tuple[float, float] | None:
        if not country_str:
            return None
        code = _NAME_TO_CODE.get(_normalize_country(country_str))
        if code is None:
            return None
        return centroids.get(code)

    per_agent_stats: dict[str, dict[str, list[float]]] = {
        a: {"dlat": [], "dlng": [], "ns_km": [], "we_km": []} for a in AGENT_NAMES
    }
    per_agent_unmatched: dict[str, int] = {a: 0 for a in AGENT_NAMES}

    for name, data in name_to_data.items():
        truth = gt.get(name)
        if not truth:
            continue
        gt_la, gt_lo = truth["lat"], truth["lng"]
        r1 = data.get("round_1_assessments", {})
        for agent in AGENT_NAMES:
            cands = r1.get(agent, {}).get("candidates", [])
            if not cands:
                continue
            pick = cands[0].get("country", "")
            cen = _agent_pick_centroid(pick)
            if cen is None:
                per_agent_unmatched[agent] += 1
                continue
            pred_la, pred_lo = cen
            dlo = pred_lo - gt_lo
            if dlo > 180:
                dlo -= 360
            elif dlo < -180:
                dlo += 360
            per_agent_stats[agent]["dlat"].append(pred_la - gt_la)
            per_agent_stats[agent]["dlng"].append(dlo)
            per_agent_stats[agent]["ns_km"].append((pred_la - gt_la) * KM_PER_DEG)
            per_agent_stats[agent]["we_km"].append(
                dlo * KM_PER_DEG * math.cos(math.radians(gt_la))
            )

    print(f"  {'Agent':<12} {'N':>5} {'MeanΔlat':>9} {'MeanN-S(km)':>12} "
          f"{'MedN-S':>8} {'MeanΔlng':>9} {'MeanW-E(km)':>12} {'MedW-E':>8}")
    print(f"  {'─' * 12} {'─' * 5} {'─' * 9} {'─' * 12} {'─' * 8} "
          f"{'─' * 9} {'─' * 12} {'─' * 8}")

    def _mean(xs): return sum(xs) / len(xs) if xs else 0.0
    def _median(xs):
        if not xs:
            return 0.0
        s = sorted(xs)
        m = len(s) // 2
        return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])

    for agent in AGENT_NAMES:
        s = per_agent_stats[agent]
        n = len(s["dlat"])
        if n == 0:
            print(f"  {agent:<12} {0:>5}   (no R1 picks could be resolved to a country centroid)")
            continue
        print(f"  {agent:<12} {n:>5} "
              f"{_mean(s['dlat']):>+8.2f}° "
              f"{_mean(s['ns_km']):>+11.0f} "
              f"{_median(s['ns_km']):>+7.0f} "
              f"{_mean(s['dlng']):>+8.2f}° "
              f"{_mean(s['we_km']):>+11.0f} "
              f"{_median(s['we_km']):>+7.0f}")
    print()
    print(f"  Country centroids derived from {sum(centroid_cnt.values())} GT entries "
          f"across {len(centroids)} countries.")
    print(f"  N = images where this agent's Round-1 top-pick country was resolvable to a centroid.")
    unmatched_summary = ", ".join(f"{a}: {n}" for a, n in per_agent_unmatched.items() if n)
    if unmatched_summary:
        print(f"  Unresolved picks (free-text country didn't match any code): {unmatched_summary}.")
    print()


def compute_dynamics(results_dir: Path, gt_path: Path | None, out_dir: Path) -> dict:
    """Compute structured debate-dynamics metrics and write dynamics_metrics.json.

    This is the machine-readable counterpart to ``analyze()``'s printed
    GT-pipeline analysis, consumed by the ``report`` step so the single
    per-approach report can carry the debate dynamics.
    """
    import json as _json
    from collections import Counter as _Counter

    results: list[dict] = []
    for img_dir in sorted(Path(results_dir).iterdir()):
        if not img_dir.is_dir():
            continue
        rj = img_dir / "result.json"
        if not rj.is_file():
            continue
        try:
            results.append(_json.loads(rj.read_text()))
        except (OSError, ValueError):
            continue

    gt = _load_ground_truth(gt_path) if gt_path else {}
    total = len(results)

    images_with_debate = 0
    converged = 0
    still_disagreed = 0
    r1_majority_matches_final = 0
    n_final = 0

    # GT-based pairing classification tallies
    cat_counts: _Counter = _Counter()
    n_pairings_gt = 0

    for r in results:
        debate = r.get("debate", {}) or {}
        pairings = debate.get("pairings", []) or []
        if pairings:
            images_with_debate += 1
            # convergence: do the debated agents' final positions collapse to one?
            final_positions = set()
            for p in pairings:
                for ag in (p.get("agent_a"), p.get("agent_b")):
                    pos = _last_position_in_pairing(p, ag) if ag else ""
                    if pos:
                        final_positions.add(pos.lower().strip())
            if len(final_positions) <= 1:
                converged += 1
            else:
                still_disagreed += 1

        # R1 majority vs final
        country_result = r.get("country_result", "") or ""
        if country_result:
            n_final += 1
            final_country = ""
            for line in country_result.split("\n"):
                if line.strip().lower().startswith("country:"):
                    final_country = line.split(":", 1)[1].strip()
                    break
            if not final_country:
                final_country = country_result.split("\n")[0].strip()
            r1 = r.get("round_1_assessments", {}) or {}
            r1_countries: _Counter = _Counter()
            for agent in AGENT_NAMES:
                cands = (r1.get(agent, {}) or {}).get("candidates", []) or []
                if cands:
                    top = (cands[0].get("country", "") or "").strip()
                    if top:
                        r1_countries[top.lower()] += 1
            if r1_countries:
                maj, _ = r1_countries.most_common(1)[0]
                if maj == final_country.lower():
                    r1_majority_matches_final += 1

        # GT-based classification of each pairing
        img_id = r.get("image_path", "")
        # map to gt via image dir name is not available here; use sample stem from image_path
        from pathlib import Path as _P
        stem = _P(img_id).stem if img_id else ""
        gt_code = (gt.get(stem, {}) or {}).get("country_code", "") if gt else ""
        if gt_code:
            for p in pairings:
                info = _classify_pairing(p, gt_code)
                cat_counts[info.get("category", "")] += 1
                n_pairings_gt += 1

    metrics = {
        "n_total": total,
        "n_images_with_debate": images_with_debate,
        "convergence": {
            "converged": converged,
            "still_disagreed": still_disagreed,
            "converged_rate": (converged / images_with_debate) if images_with_debate else None,
        },
        "round1_majority_matches_final": {
            "n_match": r1_majority_matches_final,
            "n_total": n_final,
            "rate": (r1_majority_matches_final / n_final) if n_final else None,
        },
        "gt_pairing_classification": {
            "n_pairings": n_pairings_gt,
            "counts": dict(cat_counts),
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dynamics_metrics.json"
    with open(out_file, "w") as f:
        _json.dump(metrics, f, indent=2)
    print(f"[dynamics] wrote {out_file}")
    print(f"[dynamics] debated={images_with_debate} converged={converged} "
          f"R1-maj-matches-final={r1_majority_matches_final}/{n_final} "
          f"GT-pairings={n_pairings_gt}")
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze VLM Council debate dynamics")
    parser.add_argument("results_dir", help="Directory with result.json files")
    parser.add_argument("ground_truth", nargs="?", default=None,
                        help="Optional path to georc_locations.csv for GT-based analysis")
    args = parser.parse_args()
    gt_path = Path(args.ground_truth) if args.ground_truth else None
    analyze(Path(args.results_dir), gt_path)


if __name__ == "__main__":
    main()
