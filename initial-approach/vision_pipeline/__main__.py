"""CLI entry point for the vision pipeline.

Usage:
    python -m vision_pipeline path/to/streetview.jpg
"""

import json
import sys
from pathlib import Path

from vision_pipeline.config import load_config
from vision_pipeline.graph import build_graph
from vision_pipeline.visualize import draw_bboxes

INITIAL_STATE = {
    "image_path": "",
    "scene_description": "",
    "details": [],
    "has_details": False,
    "errors": [],
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Vision pipeline")
    parser.add_argument("image_path", help="Path to a Street View image")
    args = parser.parse_args()

    image_path = str(Path(args.image_path).resolve())

    if not Path(image_path).is_file():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config()
    graph = build_graph(config)

    state = dict(INITIAL_STATE)
    state["image_path"] = image_path
    result = graph.invoke(state)

    # Structured output on stdout
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Generate bbox visualization
    stem = Path(image_path).stem
    viz_path = draw_bboxes(image_path, result.get("details", []), f"crops/{stem}_bboxes.png")
    if viz_path:
        print(f"\nBbox visualization: {viz_path}", file=sys.stderr)

    # Errors on stderr
    if result.get("errors"):
        print("\n--- Errors ---", file=sys.stderr)
        for err in result["errors"]:
            print(f"  - {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
