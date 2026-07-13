"""Florence-2 model for visual phrase grounding.

Loads Microsoft Florence-2-large from HuggingFace and provides
phrase grounding: text -> bounding box (fallback when DINO fails).

The model is loaded once and reused across calls.
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class FlorenceGrounder:
    """Wraps Florence-2 for grounding and detection tasks."""

    def __init__(self, model_name: str = "microsoft/Florence-2-large"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device)

    def _run_task(
        self, image: Image.Image, task: str, prompt: str = ""
    ) -> dict:
        """Run a Florence-2 task and return parsed results."""
        text = task + prompt

        inputs = self.processor(
            text=text, images=image, return_tensors="pt"
        ).to(self.device, self.dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        return self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

    def _normalize_bbox(
        self, bbox: list, w: int, h: int
    ) -> list[float] | None:
        """Normalize pixel bbox to 0..1 and validate."""
        if len(bbox) != 4:
            return None

        x1 = bbox[0] / w
        y1 = bbox[1] / h
        x2 = bbox[2] / w
        y2 = bbox[3] / h

        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)

        if x1 >= x2 or y1 >= y2:
            return None

        area = (x2 - x1) * (y2 - y1)
        if area > 0.8:
            return None

        return [x1, y1, x2, y2]

    def ground_phrase(
        self, image_path: str, phrase: str
    ) -> list[float] | None:
        """Locate a phrase in an image and return normalized bbox [x1, y1, x2, y2]."""
        image = Image.open(image_path).convert("RGB")
        task = "<CAPTION_TO_PHRASE_GROUNDING>"

        parsed = self._run_task(image, task, phrase)

        bboxes = parsed.get(task, {}).get("bboxes", [])
        if not bboxes:
            return None

        return self._normalize_bbox(bboxes[0], image.width, image.height)
