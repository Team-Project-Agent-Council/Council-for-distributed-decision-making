"""Crop Tool node - crops the original image based on bounding boxes.

This is NOT an LLM node. It is pure Python image manipulation using Pillow.
Takes normalized bounding box coordinates and produces cropped image files.
"""

import os
import re
from pathlib import Path

from PIL import Image

from vision_pipeline.config import PipelineConfig
from vision_pipeline.state import PipelineState

# Padding added around each bounding box (fraction of box size)
BBOX_PADDING = 0.15


def _sanitize_filename(name: str) -> str:
    """Turn a detail name into a safe filename."""
    safe = re.sub(r"[^\w\s-]", "", name.lower())
    safe = re.sub(r"[\s]+", "_", safe).strip("_")
    return safe or "detail"


def crop_tool(
    state: PipelineState,
    config: PipelineConfig,
) -> dict:
    """Crop the original image for each detail that has a valid bounding box.

    Reads:  state["image_path"], state["details"] (each with bbox)
    Writes: state["details"] (enriched with crop_path field)
    """
    details = state.get("details", [])
    image_path = state["image_path"]

    if not details:
        return {"details": details}

    try:
        img = Image.open(image_path)
    except Exception as e:
        return {
            "details": details,
            "errors": [f"Crop Tool: cannot open image: {e}"],
        }

    width, height = img.size
    os.makedirs(config.crop_output_dir, exist_ok=True)

    updated = []
    errors = []
    seen_names: dict[str, int] = {}

    for detail in details:
        d = dict(detail)
        bbox = d.get("bbox")

        if bbox is None or len(bbox) != 4:
            d["crop_path"] = None
            updated.append(d)
            continue

        x1, y1, x2, y2 = bbox

        # Add padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = box_w * BBOX_PADDING
        pad_y = box_h * BBOX_PADDING

        x1 = max(0.0, x1 - pad_x)
        y1 = max(0.0, y1 - pad_y)
        x2 = min(1.0, x2 + pad_x)
        y2 = min(1.0, y2 + pad_y)

        # Convert normalized coords to pixel coords
        left = int(x1 * width)
        top = int(y1 * height)
        right = int(x2 * width)
        bottom = int(y2 * height)

        # Skip zero-area crops
        if right <= left or bottom <= top:
            d["crop_path"] = None
            errors.append(f"Crop Tool: zero-area crop for '{d['name']}'")
            updated.append(d)
            continue

        try:
            cropped = img.crop((left, top, right, bottom))

            # Generate unique filename
            base_name = _sanitize_filename(d["name"])
            count = seen_names.get(base_name, 0)
            seen_names[base_name] = count + 1
            if count > 0:
                base_name = f"{base_name}_{count}"

            crop_path = str(Path(config.crop_output_dir) / f"{base_name}.png")
            cropped.save(crop_path, "PNG")
            d["crop_path"] = crop_path

        except Exception as e:
            d["crop_path"] = None
            errors.append(f"Crop Tool: failed to crop '{d['name']}': {e}")

        updated.append(d)

    result: dict = {"details": updated}
    if errors:
        result["errors"] = errors
    return result
