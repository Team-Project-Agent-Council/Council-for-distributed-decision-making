"""Batch processor - runs the pipeline on all images in a directory.

Output structure per image:
    <output_dir>/<image-stem>/
        result.json        # pipeline output
        bboxes.png         # bbox visualization
        crops/             # cropped detail images

Supports resume and --limit.

Usage:
    python -m vision_pipeline.batch Images/ results/
    python -m vision_pipeline.batch Images/ results/ --limit 5 --max-details 3
"""

import json
import sys
import time
from pathlib import Path

from vision_pipeline.config import PipelineConfig, load_config
from vision_pipeline.graph import build_graph
from vision_pipeline.visualize import draw_bboxes


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

INITIAL_STATE = {
    "image_path": "",
    "clean_image_path": None,
    "scene_description": "",
    "details": [],
    "has_details": False,
    "errors": [],
}


def run_batch(
    image_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    limit: int | None = None,
    clean_image_dir: Path | None = None,
) -> None:
    """Process all images in a directory through the pipeline."""
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    bbox_overview_dir = output_dir / "bbox_overview"
    bbox_overview_dir.mkdir(parents=True, exist_ok=True)

    # Check which are already done (for resume)
    done = {
        p.name
        for p in output_dir.iterdir()
        if p.is_dir() and (p / "result.json").exists()
    }

    remaining = [img for img in images if img.stem not in done]

    if limit is not None:
        remaining = remaining[:limit]

    print(
        f"Found {len(images)} images, {len(done)} already processed, "
        f"{len(remaining)} remaining.",
        file=sys.stderr,
    )

    if not remaining:
        print("All images already processed. Done.", file=sys.stderr)
        return

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

        # Per-image output directory
        img_output_dir = output_dir / image_path.stem
        img_output_dir.mkdir(parents=True, exist_ok=True)
        crop_dir = str(img_output_dir / "crops")

        try:
            img_config = PipelineConfig(
                ollama_host=config.ollama_host,
                vision_model=config.vision_model,
                text_model=config.text_model,
                grounding_model=config.grounding_model,
                max_details=config.max_details,
                crop_output_dir=crop_dir,
            )

            img_graph = build_graph(img_config)

            state = dict(INITIAL_STATE)
            state["image_path"] = str(image_path.resolve())

            # Look for a matching clean image (same stem, any supported extension)
            clean_image_path = None
            if clean_image_dir is not None:
                for ext in IMAGE_EXTENSIONS:
                    candidate = clean_image_dir / (image_path.stem + ext)
                    if candidate.is_file():
                        clean_image_path = str(candidate.resolve())
                        break
            state["clean_image_path"] = clean_image_path

            result = img_graph.invoke(state)

            # Save result
            with open(img_output_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Generate bbox visualization
            details = result.get("details", [])
            viz_path = str(bbox_overview_dir / f"{image_path.stem}.png")
            draw_bboxes(str(image_path.resolve()), details, viz_path)
            draw_bboxes(str(image_path.resolve()), details, str(img_output_dir / "bboxes.png"))

            elapsed = time.time() - img_start
            n_details = len([d for d in details if d.get("bbox")])
            status = "OK" if not result.get("errors") else f"WARN ({len(result['errors'])} errors)"
            print(f" {status} [{n_details} crops] ({elapsed:.1f}s)", file=sys.stderr)

            if result.get("errors"):
                errors_total += 1
                for err in result["errors"]:
                    print(f"    ! {err}", file=sys.stderr)

        except Exception as e:
            elapsed = time.time() - img_start
            print(f" FAIL ({elapsed:.1f}s): {e}", file=sys.stderr)
            errors_total += 1

            err_result = dict(INITIAL_STATE)
            err_result["image_path"] = str(image_path.resolve())
            err_result["errors"] = [f"Batch error: {type(e).__name__}: {e}"]
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

    parser = argparse.ArgumentParser(description="Batch process images through the vision pipeline")
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory for JSON results")
    parser.add_argument("--clean-image-dir", default=None, help="Directory with clean images (no bbox overlays) for scene description")
    parser.add_argument("--max-details", type=int, default=None, help="Override MAX_DETAILS")
    parser.add_argument("--limit", type=int, default=None, help="Process only N images")
    args = parser.parse_args()

    config = load_config()

    if args.max_details is not None:
        config = PipelineConfig(
            ollama_host=config.ollama_host,
            vision_model=config.vision_model,
            text_model=config.text_model,
            grounding_model=config.grounding_model,
            max_details=args.max_details,
            crop_output_dir=config.crop_output_dir,
        )

    run_batch(
        image_dir=Path(args.image_dir),
        output_dir=Path(args.output_dir),
        config=config,
        limit=args.limit,
        clean_image_dir=Path(args.clean_image_dir) if args.clean_image_dir else None,
    )


if __name__ == "__main__":
    main()
