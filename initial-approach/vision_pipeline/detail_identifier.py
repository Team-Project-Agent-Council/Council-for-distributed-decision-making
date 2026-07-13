"""Detail Identifier node - identifies GeoGuessr-relevant details from the scene description.

Receives the scene description WITH object positions from the Scene Parser and
determines which specific elements warrant closer examination via cropping.
Uses a text LLM (no image needed - positions are described in text).
"""

import json
import re

from ollama import Client

from vision_pipeline.config import PipelineConfig
from vision_pipeline.state import Detail, PipelineState

DETAIL_IDENTIFIER_SYSTEM_PROMPT = """\
You are a GeoGuessr meta analyst. Given a description of a Google Street View scene,
select the objects most useful for identifying the country or region.

PRIORITY ORDER - pick higher-priority items first:

1. TEXT & SIGNS (highest value):
   - Any sign with readable text - transcribed text and script are the strongest clue
   - Road name signs, shop signs, banners, stickers
   - Sign backs with visible manufacturer markings

2. LICENSE PLATES:
   - Any visible plate on any vehicle - format, color, and characters are country-specific

3. ROAD INFRASTRUCTURE:
   - Traffic lights: lens count, arrow style, mounting (horizontal/vertical/hanging)
   - Road markings: line color (white vs yellow), crosswalk style, driving side
   - Kilometer markers, mile posts

4. POLES, BOLLARDS & BARRIERS:
   - Utility poles: material, crossbar shape, insulators
   - Bollards: color bands, shape, reflectors
   - Guardrails: profile shape (W-beam, thrie-beam, cable), end treatment
   - Street lamps: arm style, head shape

5. BUILDINGS & ARCHITECTURE:
   - Roof type and tile color
   - Window and balcony styles
   - Fence and wall materials

6. VEHICLES & TRANSIT:
   - Taxis: color, roof sign, livery - taxi designs are highly country-specific
   - Public buses: color scheme, route numbers, operator markings
   - Region-specific vehicles: tuk-tuks, rickshaws, jeepneys, matatus, boda-bodas
   - Motorcycles/scooters if distinctive (cargo style, rider setup)
   - Commercial vehicles with visible text, logos, or phone numbers
   - Vehicle brand/model if recognizable and regionally distinctive (Lada, Tata, Wuling)

7. CULTURAL ELEMENTS:
   - Murals and artwork on buildings (if they contain text, symbols, or distinctive artistic style)
   - Religious buildings or structures (mosque minarets, stupas, church towers, temples)
   - Shop fronts with distinctive signage or merchandise display
   - Distinctive street furniture (post boxes, phone booths, trash bins with visible markings)

DO NOT select:
- Generic scene features: "sky", "trees", "road", "buildings", "vegetation"
- People (privacy - do not crop individuals)
- Generic cars without visible plates AND without distinctive features
- Duplicate objects of the same type - pick ONE of each kind (e.g. one pole, not three)

NAMING RULES for the "name" field:
- Use a simple, grounding-friendly label: "sign", "pole", "license plate", "traffic light"
- Add ONE descriptor if it helps distinguish from other items: "red circular sign", "wooden pole"
- Do NOT use long phrases like "large metal rectangular informational road sign"
- The name must work as a search query to find the object in the image

OUTPUT FORMAT:
Return a JSON array. Each item has:
- "name": short grounding-friendly label (1-3 words)
- "reason": why this object helps identify the location (1 sentence)

If the scene has no GeoGuessr-relevant objects, return [].

Respond ONLY with the JSON array. No commentary, no markdown fences.
Example:
[{"name": "red circular sign", "reason": "Text in Cyrillic script indicates an Eastern European country."},
 {"name": "wooden pole", "reason": "T-shaped crossbar with ceramic insulators is typical for Southeast Asia."},
 {"name": "license plate", "reason": "Yellow plate with black text matches UK format."}]\
"""

DETAIL_IDENTIFIER_USER_PROMPT = """\
Scene description:

{scene_description}

Select the {max_details} most informative GeoGuessr metas from this description.
Prioritize text, signs, and plates over generic infrastructure.
Respond with a JSON array only.\
"""


def _parse_details_json(raw: str, max_details: int) -> list[Detail]:
    """Extract a list of Detail dicts from the LLM's raw response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```", "", cleaned).strip()

    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    details: list[Detail] = []
    for item in items[:max_details]:
        if isinstance(item, dict) and "name" in item:
            details.append(
                Detail(
                    name=str(item["name"]).strip(),
                    reason=str(item.get("reason", "")).strip(),
                    bbox=None,
                    crop_path=None,
                    focused_description=None,
                )
            )
    return details


def detail_identifier(
    state: PipelineState,
    client: Client,
    config: PipelineConfig,
) -> dict:
    """Identify GeoGuessr-relevant details from the scene description.

    Reads:  state["scene_description"]
    Writes: state["details"], state["has_details"]
    """
    description = state.get("scene_description", "")

    if not description:
        return {"details": [], "has_details": False}

    try:
        response = client.chat(
            model=config.text_model,
            messages=[
                {"role": "system", "content": DETAIL_IDENTIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": DETAIL_IDENTIFIER_USER_PROMPT.format(
                        scene_description=description,
                        max_details=config.max_details,
                    ),
                },
            ],
        )

        raw = response["message"]["content"]
        details = _parse_details_json(raw, config.max_details)

        return {
            "details": details,
            "has_details": len(details) > 0,
        }

    except Exception as e:
        return {
            "details": [],
            "has_details": False,
            "errors": [f"Detail Identifier error: {type(e).__name__}: {e}"],
        }
