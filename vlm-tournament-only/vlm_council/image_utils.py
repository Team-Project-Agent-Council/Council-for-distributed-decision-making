"""Image encoding utilities."""

import base64
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from vlm_council.rag.keyed_lookup import Reference

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


def build_vlm_message_multi(
    primary: tuple[str, str],
    references: list[tuple[str, str, str]],
    text_prompt: str,
) -> HumanMessage:
    """Build a multi-image HumanMessage.

    Args:
        primary: (b64, mime) for the street-view image; placed first.
        references: list of (b64, mime, caption); each rendered as caption-text
            then image. Caption is short, e.g. "[ref] Austria/bollards".
        text_prompt: trailing text content with the question/instructions.
    """
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:{primary[1]};base64,{primary[0]}"}},
        {"type": "text", "text": "(image above: street view to identify)"},
    ]
    for b64, mime, caption in references:
        content.append({"type": "text", "text": caption})
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    content.append({"type": "text", "text": text_prompt})
    return HumanMessage(content=content)


def build_discussion_message(
    image_b64: str,
    image_mime: str,
    text_prompt: str,
    reference_images: list["Reference"] | None = None,
) -> HumanMessage:
    """Build the HumanMessage for an agent's discuss() call.

    If reference_images is non-empty, encodes each one and uses build_vlm_message_multi;
    otherwise falls back to the single-image build_vlm_message. References whose files
    cannot be read are skipped silently.
    """
    if not reference_images:
        return build_vlm_message(image_b64, image_mime, text_prompt)

    refs: list[tuple[str, str, str]] = []
    for ref in reference_images:
        try:
            b64, mime = encode_image(ref.image_path)
        except (OSError, FileNotFoundError):
            continue
        caption = f"[ref] {ref.country}/{ref.category}"
        refs.append((b64, mime, caption))

    if not refs:
        return build_vlm_message(image_b64, image_mime, text_prompt)

    return build_vlm_message_multi((image_b64, image_mime), refs, text_prompt)
