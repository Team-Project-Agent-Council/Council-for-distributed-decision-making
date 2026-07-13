"""Image encoding utilities."""

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def encode_image(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_string, mime_type)."""
    path = Path(image_path)
    mime = _MIME_TYPES.get(path.suffix.lower(), "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def build_vlm_message(b64: str, mime: str, text_prompt: str) -> HumanMessage:
    """Build a HumanMessage with an image and text prompt for a VLM."""
    return HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": text_prompt},
        ]
    )
