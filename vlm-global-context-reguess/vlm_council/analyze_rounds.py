"""Analyze how agent assessments change between Round 1 and Round 2.

Compares top candidates, confidence shifts, and country changes per agent
across the two rounds of the Global Context Re-guess pipeline.

Usage:
    python -m vlm_council.analyze_rounds results_global_context_re_guess_niklas_1/
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


AGENT_NAMES = ["linguistic", "landscape", "botanics", "regulatory", "meta"]


def _get_top_candidate(assessment: dict) -> dict | None:
    candidates = assessment.get("candidates", [])
    if candidates and isinstance(candidates[0], dict):
        return candidates[0]
    return None


def analyze(results_dir: Path) -> None:
    results = []
    for img_dir in sorted(results_dir.iterdir()):
        result_file = img_dir / "result.json"
        if not result_file.exists():
            continue
        try:
            with open(result_file) as f:
                data = json.load(f)
            if not data.get("error"):
                results.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    total = len(results)
    if total == 0:
        print("No results found.")
        return

    print("=" * 70)
    print("Round 1 -> Round 2: Agent Assessment Changes")
    print("=" * 70)
    print(f"Total images: {total}")
    print()

    # Check if Round 2 has data
    r2_with_data = 0
    for r in results:
        r2 = r.get("round_2_assessments", {})
        for agent in AGENT_NAMES:
            if r2.get(agent, {}).get("candidates"):
                r2_with_data += 1
                break

    if r2_with_data == 0:
        print("WARNING: Round 2 assessments are empty (0 candidates parsed).")
        print("This is likely a parsing bug, the agents responded but the")
        print("JSON could not be extracted from their output.")
        print()
        print("Showing Round 1 statistics only:")
        print()
        _show_round1_stats(results)
        return

    print(f"Images with Round 2 data: {r2_with_data}/{total}")
    print()

    # Per-agent analysis
    for agent in AGENT_NAMES:
        print("-" * 70)
        print(f"  {agent.upper()} AGENT")
        print("-" * 70)

        country_changed = 0
        confidence_up = 0
        confidence_down = 0
        confidence_same = 0
        top_pick_stayed = 0
        total_comparable = 0

        conf_order = {"high": 3, "medium": 2, "low": 1, "speculative": 0}
        r1_confidences = Counter()
        r2_confidences = Counter()
        changes = []

        for r in results:
            r1 = r.get("round_1_assessments", {}).get(agent, {})
            r2 = r.get("round_2_assessments", {}).get(agent, {})

            r1_top = _get_top_candidate(r1)
            r2_top = _get_top_candidate(r2)

            if not r1_top or not r2_top:
                continue

            total_comparable += 1
            r1_country = r1_top.get("country", "").strip().rstrip(".")
            r2_country = r2_top.get("country", "").strip().rstrip(".")
            r1_conf = r1_top.get("confidence", "unknown")
            r2_conf = r2_top.get("confidence", "unknown")

            r1_confidences[r1_conf] += 1
            r2_confidences[r2_conf] += 1

            if r1_country.lower() == r2_country.lower():
                top_pick_stayed += 1
            else:
                country_changed += 1
                changes.append({
                    "image": r.get("image_path", "").split("/")[-1],
                    "r1": f"{r1_country} ({r1_conf})",
                    "r2": f"{r2_country} ({r2_conf})",
                })

            r1_val = conf_order.get(r1_conf, -1)
            r2_val = conf_order.get(r2_conf, -1)
            if r2_val > r1_val:
                confidence_up += 1
            elif r2_val < r1_val:
                confidence_down += 1
            else:
                confidence_same += 1

        if total_comparable == 0:
            print("    (no comparable data)")
            print()
            continue

        print(f"    Comparable images: {total_comparable}")
        print()
        print(f"    Top pick unchanged: {top_pick_stayed}/{total_comparable} "
              f"({top_pick_stayed / total_comparable * 100:.0f}%)")
        print(f"    Top pick CHANGED:   {country_changed}/{total_comparable} "
              f"({country_changed / total_comparable * 100:.0f}%)")
        print()
        print(f"    Confidence shifts:")
        print(f"      Increased: {confidence_up} ({confidence_up / total_comparable * 100:.0f}%)")
        print(f"      Decreased: {confidence_down} ({confidence_down / total_comparable * 100:.0f}%)")
        print(f"      Same:      {confidence_same} ({confidence_same / total_comparable * 100:.0f}%)")
        print()
        print(f"    Confidence distribution:")
        print(f"      Round 1: {dict(r1_confidences.most_common())}")
        print(f"      Round 2: {dict(r2_confidences.most_common())}")

        if changes:
            print()
            print(f"    Country changes (top {min(10, len(changes))}):")
            for c in changes[:10]:
                print(f"      {c['r1']:25s} -> {c['r2']}")
            if len(changes) > 10:
                print(f"      ... and {len(changes) - 10} more")
        print()

    # Overall summary
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    all_changes = 0
    all_comparable = 0
    all_conf_up = 0
    all_conf_down = 0

    for agent in AGENT_NAMES:
        for r in results:
            r1 = r.get("round_1_assessments", {}).get(agent, {})
            r2 = r.get("round_2_assessments", {}).get(agent, {})
            r1_top = _get_top_candidate(r1)
            r2_top = _get_top_candidate(r2)
            if not r1_top or not r2_top:
                continue
            all_comparable += 1
            r1_country = r1_top.get("country", "").strip().rstrip(".")
            r2_country = r2_top.get("country", "").strip().rstrip(".")
            if r1_country.lower() != r2_country.lower():
                all_changes += 1
            conf_order = {"high": 3, "medium": 2, "low": 1, "speculative": 0}
            r1_val = conf_order.get(r1_top.get("confidence", ""), -1)
            r2_val = conf_order.get(r2_top.get("confidence", ""), -1)
            if r2_val > r1_val:
                all_conf_up += 1
            elif r2_val < r1_val:
                all_conf_down += 1

    if all_comparable > 0:
        print(f"  Total agent-image pairs compared: {all_comparable}")
        print(f"  Country changed: {all_changes}/{all_comparable} "
              f"({all_changes / all_comparable * 100:.1f}%)")
        print(f"  Confidence increased: {all_conf_up}/{all_comparable} "
              f"({all_conf_up / all_comparable * 100:.1f}%)")
        print(f"  Confidence decreased: {all_conf_down}/{all_comparable} "
              f"({all_conf_down / all_comparable * 100:.1f}%)")


def _show_round1_stats(results: list) -> None:
    """Show Round 1 statistics when Round 2 is unavailable."""
    for agent in AGENT_NAMES:
        confidences = Counter()
        countries = Counter()
        for r in results:
            r1 = r.get("round_1_assessments", {}).get(agent, {})
            top = _get_top_candidate(r1)
            if top:
                confidences[top.get("confidence", "unknown")] += 1
                countries[top.get("country", "unknown").strip().rstrip(".")] += 1

        print(f"  {agent.upper():12s}: ", end="")
        conf_str = ", ".join(f"{k}: {v}" for k, v in sorted(confidences.items()))
        print(f"confidence=[{conf_str}]  ", end="")
        print(f"top countries={[c for c, _ in countries.most_common(3)]}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Round 1 vs Round 2 changes")
    parser.add_argument("results_dir", help="Directory with result.json files")
    args = parser.parse_args()
    analyze(Path(args.results_dir))


if __name__ == "__main__":
    main()
