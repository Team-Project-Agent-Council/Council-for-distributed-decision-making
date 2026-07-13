"""Grounding DINO model for precise open-set bounding box detection.

Loads IDEA-Research/grounding-dino-base from HuggingFace and provides
a simple interface for text-conditioned object detection.
The model is loaded once and reused across calls.
"""

import re

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Map of common GeoGuessr detail names to simpler queries Grounding DINO understands
QUERY_SIMPLIFICATIONS = {
    "utility pole": "pole",
    "power pole": "pole",
    "telephone pole": "pole",
    "street lamp": "street light",
    "street light": "street light",
    "license plate": "license plate",
    "licence plate": "license plate",
    "road sign": "sign",
    "traffic sign": "sign",
    "traffic barrier": "barrier",
    "traffic cone": "cone",
    "traffic light": "traffic light",
    "traffic signal": "traffic light",
    "manhole cover": "manhole",
    "fire hydrant": "fire hydrant",
    "crosswalk": "crosswalk",
    "pedestrian crossing": "crosswalk",
    "kilometer marker": "sign",
    "mile post": "sign",
    "satellite dish": "satellite dish",
    "guardrail": "guardrail",
    "bollard": "bollard",
}


def _simplify_query(phrase: str) -> list[str]:
    """Generate progressively simpler queries from a detail name.

    Returns a list of queries to try, from most specific to most generic.
    """
    original = phrase.lower().strip()
    queries = [original]

    # Check for known simplifications
    for pattern, simple in QUERY_SIMPLIFICATIONS.items():
        if pattern in original and simple != original:
            queries.append(simple)
            break

    # Strip adjectives: remove leading words like "red", "wooden", "metal", "large"
    # Keep only the last 1-2 words as the core object
    words = re.sub(r"[^a-z\s]", "", original).split()
    if len(words) > 2:
        queries.append(" ".join(words[-2:]))
    if len(words) > 1:
        queries.append(words[-1])

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique


class GroundingDINODetector:
    """Wraps Grounding DINO for text-conditioned object detection."""

    def __init__(self, model_name: str = "IDEA-Research/grounding-dino-base"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name
        ).to(self.device)

    def _detect(
        self,
        image: Image.Image,
        query: str,
        box_threshold: float,
        text_threshold: float,
    ) -> tuple[list[float], float] | None:
        """Run detection and return (bbox_pixels, score) or None."""
        w, h = image.size

        text = query.lower().strip()
        if not text.endswith("."):
            text += "."

        inputs = self.processor(
            images=image, text=text, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )

        if not results or len(results[0]["boxes"]) == 0:
            return None

        scores = results[0]["scores"]
        best_idx = scores.argmax().item()
        box = results[0]["boxes"][best_idx].tolist()
        score = scores[best_idx].item()

        return box, score

    def ground_phrase(
        self,
        image_path: str,
        phrase: str,
        box_threshold: float = 0.2,
        text_threshold: float = 0.2,
    ) -> list[float] | None:
        """Locate a phrase in an image and return normalized bbox [x1, y1, x2, y2].

        Tries progressively simpler queries if the original phrase doesn't match.

        Args:
            image_path: Path to the image file.
            phrase: Text description of the object to locate.
            box_threshold: Minimum confidence for box detection.
            text_threshold: Minimum confidence for text matching.

        Returns:
            Normalized bbox [x1, y1, x2, y2] (0..1) or None if not found.
        """
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        queries = _simplify_query(phrase)

        best_result = None
        best_score = 0.0

        for query in queries:
            result = self._detect(image, query, box_threshold, text_threshold)
            if result is not None:
                box, score = result
                if score > best_score:
                    best_result = box
                    best_score = score
                # If we got a good match, no need to try simpler queries
                if score > 0.4:
                    break

        if best_result is None:
            return None

        box = best_result

        # Normalize to 0..1
        x1 = box[0] / w
        y1 = box[1] / h
        x2 = box[2] / w
        y2 = box[3] / h

        # Clamp
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)

        if x1 >= x2 or y1 >= y2:
            return None

        # Reject full-image boxes
        area = (x2 - x1) * (y2 - y1)
        if area > 0.8:
            return None

        return [x1, y1, x2, y2]
