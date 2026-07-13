"""Road Evidence Extractor, dedicated VLM call for structured road observations.

Replaces the regex-on-evidence approach in v10/v11/v12-original. The extractor
returns a strict JSON object that the prefilters consume directly:

    {
      "outside_color": "white" | "yellow" | "red" | "blue" | "none" | "unclear",
      "inside_color":  "white" | "yellow" | "red" | "blue" | "none" | "unclear",
      "driving_side":  "LEFT" | "RIGHT" | "UNCLEAR",
      "driving_side_basis": "oncoming_traffic" | "front_car_lane" |
                            "asymmetric_marking" | "plate_or_sign" | "none"
    }

Rules enforced in the prompt (and re-enforced after parsing):
- If driving_side_basis is "none", driving_side MUST be UNCLEAR.
- The camera may be REAR-mounted; the lane the camera-car drives in is then
  mirrored. The model is instructed not to infer driving side from the
  camera-car's lane alone, it must point to one of the listed bases.
- "outside" = edge line, "inside" = center line, matches the road_lines table
  format ("Outside: X | Inside: Y").

Output is plain dict, no langgraph nodes here. The graph node lives in
``vlm_council/road_evidence.py`` and stores the result under state
key ``road_evidence``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal

from langchain_core.messages import SystemMessage

from vlm_council.image_utils import build_vlm_message
from vlm_council.llm import get_vlm


_THINKING_ENABLED = os.environ.get("VLM_JUDGE_THINKING", "false").lower() in ("true", "1", "yes")
_THINK_PREFIX = "<|think|>\n" if _THINKING_ENABLED else ""


_VALID_COLORS = {"white", "yellow", "red", "blue", "none", "unclear"}
_VALID_SIDES = {"LEFT", "RIGHT", "UNCLEAR"}
_VALID_BASES = {
    "oncoming_traffic",
    "front_car_lane",
    "asymmetric_marking",
    "plate_or_sign",
    "none",
}


SYSTEM_PROMPT = """\
You are the Road Evidence Extractor for a GeoGuessr council. Your ONLY job is to read the image and return four structured observations about the road infrastructure. You do NOT propose countries.

You will report:

1. outside_color, color of the EDGE / outermost continuous road line.
2. inside_color, color of the CENTER line dividing the two driving directions.
3. driving_side, which side of the road traffic drives on.
4. driving_side_basis, the SINGLE strongest visual cue you used to decide driving_side.

Color values (lowercase, exactly one of):
- white, yellow, red, blue, none, unclear
  • Use "none" only if the road has no marking of that type at all.
  • Use "unclear" if the marking exists but the color cannot be determined.

Driving side values:
- LEFT, RIGHT, UNCLEAR

Driving-side basis values:
- oncoming_traffic, at least one oncoming vehicle is visible, on the opposite lane
- front_car_lane, the camera car is clearly mounted on the FRONT of the vehicle and you can see which lane it occupies
- asymmetric_marking, the road marking pattern itself is asymmetric in a way that determines side (e.g. yellow line only on one specific edge in a region's standard)
- plate_or_sign, a license plate or road sign with diagnostic positioning is visible
- none, none of the above is observable

CRITICAL RULES, READ CAREFULLY:

A. The Street View camera may be mounted on the REAR of the vehicle. In that case the lane the camera-car appears to drive in is MIRRORED in the image. You MUST NOT infer driving_side from the camera-car's apparent lane unless you are certain the camera is forward-facing (basis = "front_car_lane"). When in doubt, use a different basis or set basis = "none".

B. If your basis is "none", driving_side MUST be "UNCLEAR". No exceptions.

C. "outside" is the EDGE line of the lane (next to the shoulder), "inside" is the CENTER line that separates the two traffic directions. Do not confuse these.

D. Report only what you can ACTUALLY SEE. Do not guess based on what is "common" for any region.

Respond with JSON ONLY (no prose, no markdown):
{"outside_color": "<value>", "inside_color": "<value>", "driving_side": "<value>", "driving_side_basis": "<value>"}\
"""


def _strip_think_tags(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if m:
        return m.group(2).strip()
    m = re.search(r"<\|channel\>thought(.*?)<channel\|>(.*)", text, re.DOTALL)
    if m:
        return m.group(2).strip()
    m = re.search(r"</think>(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _parse(raw: str) -> dict | None:
    text = _strip_think_tags(raw).strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _coerce_color(v) -> str:
    if not isinstance(v, str):
        return "unclear"
    s = v.strip().lower()
    return s if s in _VALID_COLORS else "unclear"


def _coerce_side(v) -> str:
    if not isinstance(v, str):
        return "UNCLEAR"
    s = v.strip().upper()
    return s if s in _VALID_SIDES else "UNCLEAR"


def _coerce_basis(v) -> str:
    if not isinstance(v, str):
        return "none"
    s = v.strip().lower().replace(" ", "_").replace("-", "_")
    return s if s in _VALID_BASES else "none"


def _empty_evidence() -> dict:
    return {
        "outside_color": "unclear",
        "inside_color": "unclear",
        "driving_side": "UNCLEAR",
        "driving_side_basis": "none",
    }


async def extract(image_b64: str, image_mime: str, llm=None) -> dict:
    """Run the extractor and return a validated dict.

    On any error or unparseable response, returns an "all-unclear" payload, downstream filters then skip eliminations and rely on tournament context
    instead.
    """
    if llm is None:
        llm = get_vlm("road_evidence")

    text_prompt = (
        "Examine the road in the image and report the four required fields. "
        "Follow the rules carefully, especially the rear-camera caveat and "
        "the basis=none → UNCLEAR rule. Respond with JSON only."
    )
    msg = build_vlm_message(image_b64, image_mime, text_prompt)

    try:
        response = await llm.ainvoke([
            SystemMessage(content=_THINK_PREFIX + SYSTEM_PROMPT),
            msg,
        ])
    except Exception:  # noqa: BLE001
        return _empty_evidence()

    parsed = _parse(getattr(response, "content", "") or "")
    if not parsed:
        return _empty_evidence()

    out = {
        "outside_color": _coerce_color(parsed.get("outside_color")),
        "inside_color": _coerce_color(parsed.get("inside_color")),
        "driving_side": _coerce_side(parsed.get("driving_side")),
        "driving_side_basis": _coerce_basis(parsed.get("driving_side_basis")),
    }

    # Hard rule: basis=none ⇒ driving_side=UNCLEAR
    if out["driving_side_basis"] == "none":
        out["driving_side"] = "UNCLEAR"

    return out
