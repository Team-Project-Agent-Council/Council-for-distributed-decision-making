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


INITIAL_STATE = {
    "image_path": "",
    "image_b64": "",
    "image_mime": "",
    # Initial assessments
    "linguistic_assessment": {},
    "landscape_assessment": {},
    "botanics_assessment": {},
    "regulatory_assessment": {},
    "meta_assessment": {},
    # Progressive Narrowing
    "current_phase": "initial",
    "region_consensus": False,
    "confirmed_region": "",
    "proposed_regions": [],
    "region_candidates": {},
    # Hypotheses & Evaluations
    "active_hypotheses": [],
    "hypothesis_evaluations": [],
    # Country-Level Assessments (Path B)
    "linguistic_country_assessment": {},
    "landscape_country_assessment": {},
    "botanics_country_assessment": {},
    "regulatory_country_assessment": {},
    "meta_country_assessment": {},
    # Results
    "region_decision_reasoning": "",
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

    parser = argparse.ArgumentParser(description="VLM Council, Progressive Narrowing, single image")
    parser.add_argument("image_path", help="Path to a Street View image")
    args = parser.parse_args()

    image_path = str(Path(args.image_path).resolve())
    if not Path(image_path).is_file():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(run_single(image_path))

    country_result = result.get("country_result", "")
    # Prefer the graph-populated coordinates (country_decision_node writes
    # {"lat", "lng"} or None). Fall back to re-parsing country_result as a
    # legacy safety net, then to None.
    graph_coords = result.get("coordinates")
    if isinstance(graph_coords, dict) and "lat" in graph_coords and "lng" in graph_coords:
        coordinates: object = graph_coords
    else:
        coordinates = parse_coordinates(country_result)

    output = {
        "image_path": result.get("image_path", ""),
        "assessments": {
            name: result.get(f"{name}_assessment", {})
            for name in ["linguistic", "landscape", "botanics", "regulatory", "meta"]
        },
        "progressive_narrowing": {
            "region_consensus": result.get("region_consensus", False),
            "confirmed_region": result.get("confirmed_region", ""),
            "proposed_regions": result.get("proposed_regions", []),
            "region_decision_reasoning": result.get("region_decision_reasoning", ""),
        },
        "country_assessments": {
            name: result.get(f"{name}_country_assessment", {})
            for name in ["linguistic", "landscape", "botanics", "regulatory", "meta"]
            if result.get(f"{name}_country_assessment")
        },
        "hypothesis_evaluations": result.get("hypothesis_evaluations", []),
        "country_result": country_result,
        "coordinates": coordinates,
        "final_reasoning": result.get("final_reasoning", ""),
    }
    if result.get("error"):
        output["error"] = result["error"]
    print(json.dumps(output, indent=2, ensure_ascii=False))

    # Summary on stderr
    print(f"\n--- VLM Council Result ---", file=sys.stderr)
    print(f"Image: {Path(image_path).name}", file=sys.stderr)
    print(f"Path: {'A (consensus)' if result.get('region_consensus') else 'B (no consensus)'}", file=sys.stderr)
    print(f"Region: {result.get('confirmed_region', '?')}", file=sys.stderr)
    for name in ["linguistic", "landscape", "botanics", "regulatory", "meta"]:
        a = result.get(f"{name}_assessment", {})
        candidates = [c.get("country", "?") for c in a.get("candidates", [])]
        candidates_str = ", ".join(candidates) if candidates else "insufficient"
        print(f"  {name:12s}: {candidates_str}", file=sys.stderr)
    print(f"  Final: {result.get('country_result', '?')}", file=sys.stderr)


if __name__ == "__main__":
    main()
