"""Batch processor for the VLM Council.

Output structure per image:
    <output_dir>/<image-stem>/
        result.json        # full pipeline output

Resume-capable: already processed images are skipped.

Usage:
    python -m vlm_council.batch Images/ results/ --limit 10
    python -m vlm_council.batch Images/ results/ --offset 10 --limit 10  # used for parallel SLURM jobs
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

    Coordinates: prefers the tournament-populated `{"lat", "lng"}` dict,
    falls back to a re-parse from `country_result`, and finally to None.
    No `(0, 0)` fake fallback (walkover and no-candidates cases now emit
    coordinates=None from tournament.py).

    Error propagation: if any node set `error`, the top-level `error`
    field is written so evaluate.py can treat this run as an error.
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
        "judge_model": config.judge_model,
        "timing": {
            "total_seconds": round(elapsed_seconds, 1),
        },
        "assessments": {},
        "progressive_narrowing": {
            "region_consensus": result.get("region_consensus", False),
            "confirmed_region": result.get("confirmed_region", ""),
            "runner_up_region": result.get("runner_up_region"),
            "surviving_regions": result.get("surviving_regions", []),
            "proposed_regions": result.get("proposed_regions", []),
            "region_candidates": result.get("region_candidates", {}),
            "region_decision_reasoning": result.get("region_decision_reasoning", ""),
            "path": "A" if result.get("region_consensus", False) else "B",
        },
        "road_evidence": result.get("road_evidence", {}),
        "country_assessments": {},
        "hypothesis_evaluations": result.get("hypothesis_evaluations", []),
        "candidate_pool": result.get("candidate_pool", []),
        "rag_findings": result.get("rag_findings", []),
        "rag_refs_seen": result.get("rag_refs_seen", []),
        "road_filter_warnings": result.get("road_filter_warnings", []),
        "tournament_log": result.get("tournament_log", []),
        "country_result": country_result,
        "coordinates": coordinates,
        "final_reasoning": result.get("final_reasoning", ""),
    }

    judge_error = result.get("error")
    if judge_error:
        output["error"] = judge_error

    for name in AGENT_NAMES:
        assessment = result.get(f"{name}_assessment", {})
        if assessment:
            output["assessments"][name] = dict(assessment)
        else:
            output["assessments"][name] = {
                "agent_name": name,
                "candidates": [],
                "evidence": [],
            }

    # Country assessments
    for name in AGENT_NAMES:
        ca = result.get(f"{name}_country_assessment", {})
        if ca and ca.get("candidates"):
            output["country_assessments"][name] = dict(ca)

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

    judge_info = config.judge_model
    if config.judge_model != config.vlm_model:
        judge_info += " (separate model)"
    print(
        f"VLM Council \n"
        f"Model: {config.vlm_model}\n"
        f"Judge: {judge_info}\n"
        f"API: {config.api_base}\n"
        f"Agents: parallel (vLLM continuous batching)\n"
        f"Max region hypotheses: {config.max_region_hypotheses}\n"
        f"Max country hypotheses: {config.max_country_hypotheses}\n"
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
            path = output["progressive_narrowing"]["path"]
            region = output["progressive_narrowing"]["confirmed_region"]
            print(f" {country} [Path {path}, {region}] ({elapsed:.1f}s)", file=sys.stderr)

        except Exception as e:
            elapsed = time.time() - img_start
            print(f" FAIL ({elapsed:.1f}s): {e}", file=sys.stderr)
            errors_total += 1

            err_result = {
                "image_path": str(image_path.resolve()),
                "error": f"{type(e).__name__}: {e}",
                "assessments": {},
                "progressive_narrowing": {},
                "hypothesis_evaluations": [],
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

    parser = argparse.ArgumentParser(description="Batch VLM Council")
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
