"""Detail Extractor node - generates bounding box coordinates.

Uses Grounding DINO (primary) with Florence-2 as fallback to locate each
identified detail in the image with precise bounding boxes.
Includes deduplication: if two details get nearly identical bounding boxes
(IoU > threshold), only the first is kept.
"""

from vision_pipeline.config import PipelineConfig
from vision_pipeline.state import PipelineState
from vision_pipeline.grounding import GroundingDINODetector
from vision_pipeline.florence import FlorenceGrounder

# Module-level caches (loaded once, reused)
_dino: GroundingDINODetector | None = None
_florence: FlorenceGrounder | None = None

# Bounding boxes with IoU above this threshold are considered duplicates
IOU_DEDUP_THRESHOLD = 0.7


def _get_dino() -> GroundingDINODetector:
    """Get or create the Grounding DINO detector (singleton)."""
    global _dino
    if _dino is None:
        _dino = GroundingDINODetector()
    return _dino


def _get_florence() -> FlorenceGrounder:
    """Get or create the Florence-2 grounder (singleton, fallback only)."""
    global _florence
    if _florence is None:
        _florence = FlorenceGrounder()
    return _florence


def _iou(a: list[float], b: list[float]) -> float:
    """Compute Intersection over Union between two normalized bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def _deduplicate(details: list[dict]) -> list[dict]:
    """Remove details whose bounding boxes overlap too much (IoU > threshold).

    Keeps the first detail in the list for each overlapping group.
    Details without a bbox are always kept.
    """
    kept: list[dict] = []

    for d in details:
        bbox = d.get("bbox")
        if bbox is None:
            kept.append(d)
            continue

        is_dup = False
        for k in kept:
            k_bbox = k.get("bbox")
            if k_bbox is not None and _iou(bbox, k_bbox) > IOU_DEDUP_THRESHOLD:
                is_dup = True
                break

        if not is_dup:
            kept.append(d)

    return kept


def detail_extractor(
    state: PipelineState,
    config: PipelineConfig,
) -> dict:
    """Locate identified details in the image and produce bounding boxes.

    Uses Grounding DINO as primary model. Falls back to Florence-2 if DINO
    fails to find a bounding box for a detail.

    Reads:  state["image_path"], state["details"]
    Writes: state["details"] (enriched with bbox field, deduplicated)
    """
    details = state.get("details", [])
    image_path = state["image_path"]

    if not details:
        return {"details": details}

    dino = _get_dino()
    updated = []
    errors = []

    for detail in details:
        d = dict(detail)

        try:
            # Primary: Grounding DINO
            d["bbox"] = dino.ground_phrase(image_path, detail["name"])

            # Fallback: Florence-2 if DINO found nothing
            if d["bbox"] is None:
                florence = _get_florence()
                d["bbox"] = florence.ground_phrase(image_path, detail["name"])

        except Exception as e:
            d["bbox"] = None
            errors.append(
                f"Detail Extractor error for '{detail['name']}': {type(e).__name__}: {e}"
            )

        updated.append(d)

    # Deduplicate overlapping bounding boxes
    before_count = len([d for d in updated if d.get("bbox")])
    updated = _deduplicate(updated)
    after_count = len([d for d in updated if d.get("bbox")])

    if before_count > after_count:
        errors.append(
            f"Detail Extractor: removed {before_count - after_count} duplicate bbox(es)"
        )

    result: dict = {"details": updated}
    if errors:
        result["errors"] = errors
    return result
