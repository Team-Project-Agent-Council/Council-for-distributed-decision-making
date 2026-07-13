from __future__ import annotations

import base64
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from council.llm import get_llm, get_thinking_prefix

_llm = get_llm("vision")

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class VisionOutput(BaseModel):
    general_description: str
    crop_descriptions: list[str]


def _encode_image(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    mime = _MIME_TYPES.get(path.suffix.lower(), "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def _parse_output(text: str) -> VisionOutput:
    """Parse JSON from model output, with fallback if wrapped in markdown code block."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text.strip())
        return VisionOutput(
            general_description=data.get("general_description", ""),
            crop_descriptions=data.get("crop_descriptions", []),
        )
    except json.JSONDecodeError:
        return VisionOutput(general_description=text, crop_descriptions=[])


async def run(image_path: str) -> VisionOutput:
    think = get_thinking_prefix("vision")
    b64, mime = _encode_image(image_path)
    response = await _llm.ainvoke([
        SystemMessage(content=(
            f"{think} You are analyzing a street-level image for GeoGuessr country identification. "
            "Respond with a JSON object with exactly two keys:\n"
            '- "general_description": string - complete scene description covering landscape, '
            "environment, infrastructure, weather, and notable visual features. Be VERY detailed here, that is VERY important, otherwise the whole system will fail!\n"
            '- "crop_descriptions": list of strings - specific observations, each focusing on one '
            "distinct element (signs with exact text, license plates, road markings, vegetation, "
            "buildings, vehicles).\n"
            "Respond with raw JSON only, no markdown."
        )),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
            ]
        ),
    ])
    return _parse_output(response.content)
