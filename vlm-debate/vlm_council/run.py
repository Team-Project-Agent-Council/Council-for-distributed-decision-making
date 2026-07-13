"""CLI entry point for single-image runs.

Usage:
    python -m vlm_council.run path/to/streetview.png
"""

import asyncio
import json
import sys
from pathlib import Path

from vlm_council.config import load_config
from vlm_council.coordinates import parse_coordinates
from vlm_council.graph import build_graph


AGENT_NAMES = ["linguistic", "landscape", "botanics", "regulatory", "meta"]

INITIAL_STATE = {
    "image_path": "",
    "image_b64": "",
    "image_mime": "",
    "round_1_linguistic": {},
    "round_1_landscape": {},
    "round_1_botanics": {},
    "round_1_regulatory": {},
    "round_1_meta": {},
    "debate_pairings": [],
    "moderator_decisions": [],
    "current_debate_round": 0,
    "debate_terminated": False,
    "country_result": "",
    "coordinates": None,
    "final_reasoning": "",
    "error": None,
}


async def run_single(image_path: str) -> dict:
    """Run the VLM Council on a single image."""
    config = load_config()
    graph = build_graph(config)

    state = dict(INITIAL_STATE)
    state["image_path"] = str(Path(image_path).resolve())

    result = await graph.ainvoke(state)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="VLM Council, Debate: single image")
    parser.add_argument("image_path", help="Path to a Street View image")
    args = parser.parse_args()

    image_path = str(Path(args.image_path).resolve())
    if not Path(image_path).is_file():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(run_single(image_path))

    country_result = result.get("country_result", "")
    graph_coords = result.get("coordinates")
    if isinstance(graph_coords, dict) and "lat" in graph_coords and "lng" in graph_coords:
        coordinates: object = graph_coords
    else:
        coordinates = parse_coordinates(country_result)

    output = {
        "image_path": result.get("image_path", ""),
        "round_1_assessments": {
            name: result.get(f"round_1_{name}", {})
            for name in AGENT_NAMES
        },
        "debate": {
            "total_rounds": result.get("current_debate_round", 0),
            "moderator_decisions": result.get("moderator_decisions", []),
            "pairings": result.get("debate_pairings", []),
        },
        "country_result": country_result,
        "coordinates": coordinates,
        "final_reasoning": result.get("final_reasoning", ""),
    }
    if result.get("error"):
        output["error"] = result["error"]
    print(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n--- VLM Council Result (Debate) ---", file=sys.stderr)
    print(f"Image: {Path(image_path).name}", file=sys.stderr)
    print(f"\n  Round 1 (independent):", file=sys.stderr)
    for name in AGENT_NAMES:
        a = result.get(f"round_1_{name}", {})
        candidates = a.get("candidates", [])
        top = candidates[0]["country"] if candidates else "insufficient"
        conf = candidates[0]["confidence"] if candidates else "?"
        print(f"    {name:12s}: {top} ({conf})", file=sys.stderr)

    debate_pairings = result.get("debate_pairings", [])
    moderator_decisions = result.get("moderator_decisions", [])
    if debate_pairings:
        print(f"\n  Debate ({len(debate_pairings)} pairing(s) across {result.get('current_debate_round', 0)} round(s)):", file=sys.stderr)
        for pairing in debate_pairings:
            r = pairing.get("debate_round", "?")
            a = pairing.get("agent_a", "?")
            b = pairing.get("agent_b", "?")
            print(f"    Round {r}: {a} vs {b}", file=sys.stderr)
            for ex in pairing.get("exchanges", []):
                revised = " → REVISED" if ex.get("revised") else ""
                print(f"      {ex.get('agent_name', '?')}: {ex.get('position', '?')} ({ex.get('confidence', '?')}){revised}", file=sys.stderr)
    else:
        term_reason = moderator_decisions[-1].get("termination_reason", "?") if moderator_decisions else "?"
        print(f"\n  Debate: skipped ({term_reason})", file=sys.stderr)

    print(f"\n  Final: {result.get('country_result', '?')}", file=sys.stderr)


if __name__ == "__main__":
    main()
