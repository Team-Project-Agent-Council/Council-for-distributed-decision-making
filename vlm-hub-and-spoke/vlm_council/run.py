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
    "linguistic_assessment": {},
    "landscape_assessment": {},
    "botanics_assessment": {},
    "regulatory_assessment": {},
    "meta_assessment": {},
    "discussion_log": [],
    "discussion_round": 0,
    "judge_messages": [],
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

    parser = argparse.ArgumentParser(description="VLM Council, Hub-and-Spoke: single image")
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
        "assessments": {
            name: result.get(f"{name}_assessment", {})
            for name in ["linguistic", "landscape", "botanics", "regulatory", "meta"]
        },
        "discussion_log": result.get("discussion_log", []),
        "discussion_rounds": result.get("discussion_round", 0),
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
    for name in ["linguistic", "landscape", "botanics", "regulatory", "meta"]:
        a = result.get(f"{name}_assessment", {})
        candidates = a.get("candidates", [])
        candidates_str = ", ".join(candidates) if candidates else "insufficient"
        conf = a.get("confidence", "?")
        print(f"  {name:12s}: {candidates_str} ({conf})", file=sys.stderr)
    print(f"  Discussion rounds: {result.get('discussion_round', 0)}", file=sys.stderr)
    print(f"  Final: {result.get('country_result', '?')}", file=sys.stderr)


if __name__ == "__main__":
    main()
