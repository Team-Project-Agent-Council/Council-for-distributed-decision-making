"""Batch processor for the VLM Council, Debate approach.

Output structure per image:
    <output_dir>/<image-stem>/
        result.json        # full pipeline output (round 1, debate, final)

Resume-capable: already processed images are skipped.

Usage:
    python -m vlm_council.batch Images/ results/ --limit 10
    python -m vlm_council.batch Images/ results/ --offset 10 --limit 10
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from vlm_council.config import load_config
from vlm_council.coordinates import parse_coordinates
from vlm_council.graph import build_graph
from vlm_council.run import INITIAL_STATE


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

AGENT_NAMES = ["linguistic", "landscape", "botanics", "regulatory", "meta"]


def _serialize_result(result: dict, elapsed_seconds: float, config) -> dict:
    """Extract serializable fields from the graph result.

    Coordinates: prefers the graph-populated `{"lat", "lng"}` dict, falls
    back to a re-parse from `country_result`, and finally to None. No
    `(0, 0)` fake fallback.

    Error propagation: if the graph set `error` (only judge_final_node does
    this today), the top-level `error` field is written so evaluate.py can
    treat the run as an error rather than a valid Unknown-@-(0,0).
    """
    country_result = result.get("country_result", "")

    graph_coords = result.get("coordinates")
    if isinstance(graph_coords, dict) and "lat" in graph_coords and "lng" in graph_coords:
        coordinates: object = graph_coords
    else:
        coordinates = parse_coordinates(country_result)

    output = {
        "image_path": result.get("image_path", ""),
        "model": config.vlm_model,
        "judge_thinking": config.judge_thinking,
        "image_token_budget": config.image_token_budget,
        "debate_max_rounds": config.debate_max_rounds,
        "debate_max_exchanges": config.debate_max_exchanges,
        "debate_min_confidence": config.debate_min_confidence,
        "timing": {
            "total_seconds": round(elapsed_seconds, 1),
        },
        "round_1_assessments": {},
        "debate": {
            "total_rounds": result.get("current_debate_round", 0),
            "moderator_decisions": result.get("moderator_decisions", []),
            "pairings": result.get("debate_pairings", []),
        },
        "country_result": country_result,
        "coordinates": coordinates,
        "final_reasoning": result.get("final_reasoning", ""),
    }

    judge_error = result.get("error")
    if judge_error:
        output["error"] = judge_error

    for name in AGENT_NAMES:
        r1 = result.get(f"round_1_{name}", {})
        output["round_1_assessments"][name] = dict(r1) if r1 else {
            "agent_name": name, "candidates": [], "evidence": [],
        }
    return output


async def run_batch(
    image_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    offset: int = 0,
    file_list: Path | None = None,
) -> None:
    """Process all images in a directory through the VLM Council."""
    config = load_config()

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if file_list is not None:
        with open(file_list) as f:
            wanted = {line.strip() for line in f if line.strip()}
        images = [img for img in images if img.name in wanted]
    else:
        if offset > 0:
            images = images[offset:]
        if limit is not None:
            images = images[:limit]

    # Resume: skip already processed images (but retry failed ones)
    done = set()
    for p in output_dir.iterdir():
        if p.is_dir() and (p / "result.json").exists():
            try:
                with open(p / "result.json") as f:
                    result_data = json.load(f)
                if not result_data.get("error"):
                    done.add(p.name)
            except (json.JSONDecodeError, OSError):
                pass

    remaining = [img for img in images if img.stem not in done]

    print(
        f"VLM Council, Debate\n"
        f"Model: {config.vlm_model}\n"
        f"Judge thinking: {config.judge_thinking}\n"
        f"Image token budget: {config.image_token_budget}\n"
        f"Debate max rounds: {config.debate_max_rounds}\n"
        f"Debate min confidence: {config.debate_min_confidence}\n"
        f"API: {config.api_base}\n"
        f"Topology: Round 1 -> Moderator -> Debate (loop) -> Judge\n"
        f"Assigned {len(images)} images, {len(done)} already processed, "
        f"{len(remaining)} remaining.",
        file=sys.stderr,
    )

    if not remaining:
        print("All images already processed. Done.", file=sys.stderr)
        return

    graph = build_graph(config)
    errors_total = 0
    start_time = time.time()

    for i, image_path in enumerate(remaining, 1):
        img_start = time.time()
        print(
            f"[{i}/{len(remaining)}] Processing {image_path.name} ...",
            file=sys.stderr,
            end="",
            flush=True,
        )

        img_output_dir = output_dir / image_path.stem
        img_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            state = dict(INITIAL_STATE)
            state["image_path"] = str(image_path.resolve())
            result = await graph.ainvoke(state)

            elapsed = time.time() - img_start
            output = _serialize_result(result, elapsed, config)
            with open(img_output_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            country = output.get("country_result", "?")[:60]
            n_debates = len(output["debate"]["pairings"])
            print(f" {country} ({elapsed:.1f}s, {n_debates} debate(s))", file=sys.stderr)

        except Exception as e:
            elapsed = time.time() - img_start
            print(f" FAIL ({elapsed:.1f}s): {e}", file=sys.stderr)
            errors_total += 1

            err_result = {
                "image_path": str(image_path.resolve()),
                "error": f"{type(e).__name__}: {e}",
                "round_1_assessments": {},
                "debate": {"total_rounds": 0, "moderator_decisions": [], "pairings": []},
                "country_result": "",
                "coordinates": None,
                "final_reasoning": "",
            }
            with open(img_output_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(err_result, f, indent=2, ensure_ascii=False)

    total_time = time.time() - start_time
    avg = total_time / len(remaining) if remaining else 0

    print(
        f"\nDone. Processed {len(remaining)} images in {total_time:.0f}s "
        f"(avg {avg:.1f}s/image). Errors: {errors_total}.",
        file=sys.stderr,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Batch VLM Council, Debate")
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory for JSON results")
    parser.add_argument("--limit", type=int, default=None, help="Process only N images")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N images (for parallel jobs)")
    parser.add_argument("--file-list", type=str, default=None, help="File with image names (one per line)")
    args = parser.parse_args()

    asyncio.run(run_batch(
        image_dir=Path(args.image_dir),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        offset=args.offset,
        file_list=Path(args.file_list) if args.file_list else None,
    ))


if __name__ == "__main__":
    main()
