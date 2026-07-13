"""Pipeline state schema for the vision graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Detail(TypedDict):
    """A single notable detail identified in the scene."""

    name: str  # short label, e.g. "road sign"
    reason: str  # why it is worth examining more closely
    bbox: list[float] | None  # [x1, y1, x2, y2] normalized 0..1, set by Detail Extractor
    crop_path: str | None  # file path to cropped image, set by Crop Tool
    focused_description: str | None  # detailed description, set by Detail Focusser


class PipelineState(TypedDict):
    """State that flows through the entire LangGraph pipeline."""

    # Input (set once at invocation)
    image_path: str
    clean_image_path: str | None  # clean image without bounding box overlays, for scene description

    # Scene Parser output
    scene_description: str

    # Detail pipeline
    details: list[Detail]  # last-write-wins - each node returns the full updated list
    has_details: bool  # routing flag for the conditional edge

    # Error accumulation (operator.add appends across nodes)
    errors: Annotated[list[str], operator.add]
