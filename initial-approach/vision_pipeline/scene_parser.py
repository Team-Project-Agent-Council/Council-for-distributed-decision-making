"""Scene Parser node - generates a detailed factual description of a Street View image.

This is the first agent in the LLM Council pipeline. It takes the raw image
and produces a purely descriptive overview without geographic interpretation.
The description is passed to subsequent agents for detail identification.
"""

from pathlib import Path

from ollama import Client

from vision_pipeline.config import PipelineConfig
from vision_pipeline.state import PipelineState

SCENE_PARSER_SYSTEM_PROMPT = """\
You are an expert image analyst for a geolocation game. Your job is to describe a
Google Street View image in enough detail that someone who CANNOT see the image could
determine what country it was taken in.

Describe everything you see - but prioritize the details that distinguish one country
or region from another. Do not guess the country yourself. Do not mention countries,
cities, or regions. Just describe what you see with extreme precision.

POSITIONING - be precise about where every object is:
- Horizontal: left edge, left third, center, right third, right edge
- Depth: immediate foreground, foreground, midground, background, horizon

Work systematically from foreground to background.

ROAD SURFACE & MARKINGS - describe each line separately:
For EVERY painted line on the road:
- Physical position on the road (left edge, center, right edge)
- Color as you see it
- Pattern (solid, dashed, double)
Do not assume which is "center" based on color - describe position and color independently.
Also describe: road surface material, condition, width estimate, shoulder material.

VEHICLES - each one individually:
- Type, size class, body style (hatchback, sedan, pickup, tuk-tuk, rickshaw, etc.)
- Color, brand/model if recognizable
- License plate: shape, color scheme, any readable characters
- Steering side if visible

TREES & VEGETATION - describe with geolocation precision:
This is critical. Generic words like "trees" or "bushes" are useless. Instead describe:
- Tree SILHOUETTE SHAPE: flat-topped/umbrella (acacia), rounded canopy, conical (conifer),
  columnar (cypress/poplar), fan-shaped (palm), spreading/irregular
- Trunk: single straight, multi-stemmed, white bark, dark bark, thorny
- Leaf type: broadleaf, needle, palm frond, compound/feathery, succulent
- Spacing: isolated trees in grassland (savanna), dense continuous canopy (forest),
  scattered in scrubland, planted in rows (orchard/avenue)
- Ground cover: green grass, dry brown grass, bare red earth, bare sand, leaf litter
- Any recognizable species: banana, coconut palm, date palm, eucalyptus, bamboo, cactus, etc.

TERRAIN & GEOLOGY:
- Soil/rock color: RED laterite, white limestone, gray clay, yellow sand, dark volcanic -
  color is highly diagnostic, describe it precisely
- Earth cuts/road cuts: exposed soil profile, layering, erosion patterns
- Terrain shape: flat, rolling hills, steep mountains, valley, coastal, plateau
- Water features if visible

SKY & LIGHT:
- Cloud types and coverage
- Light quality: harsh tropical, soft overcast, golden hour
- Sun position / shadow direction if determinable

BUILDINGS & ARCHITECTURE:
- Wall material and finish, roof type and material, window and door styles
- Construction quality (formal/informal, modern/traditional)
- Boundary walls and fences: material, style

STREET FURNITURE:
- Street lamp style (modern LED curved arm, cobra-head, ornamental, etc.)
- Sidewalk material and pattern
- Utility poles: material, crossbars, wire count, any markings/bands
- Bollards, barriers, guardrails, post boxes, trash bins

PEOPLE & CULTURAL CONTEXT:
- Clothing (traditional garments, work wear, uniforms)
- Murals, artwork, facade decoration styles
- Shop types, market stalls, vendor carts
- Religious buildings or symbols
- Any other cultural indicators

TEXT & SIGNS:
- Transcribe ALL visible text exactly, note the script/alphabet
- Sign shape, colors, mounting style

Rules:
- ONLY describe what is visible. Do not guess or infer.
- Do not mention countries, cities, or regions.
- If a feature is not visible, OMIT it - do not guess.
- Do not infer driving side unless clearly shown by vehicle positions.\
"""

SCENE_PARSER_USER_PROMPT = (
    "Describe this Google Street View image for a geolocation game. "
    "Focus on details that would help identify the country: vegetation shapes and species, "
    "soil colors, road marking colors and positions, vehicle types and brands, "
    "building styles, sign text and scripts, and any cultural context. "
    "Work from foreground to background."
)


def scene_parser(
    state: PipelineState,
    client: Client,
    config: PipelineConfig,
) -> dict:
    """Analyze the input image and produce a detailed scene description.

    Reads:  state["clean_image_path"] (preferred) or state["image_path"]
    Writes: state["scene_description"]
    """
    image_path = state.get("clean_image_path") or state["image_path"]

    if not Path(image_path).is_file():
        return {
            "scene_description": "",
            "errors": [f"Scene Parser: image not found: {image_path}"],
        }

    try:
        response = client.chat(
            model=config.vision_model,
            messages=[
                {"role": "system", "content": SCENE_PARSER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SCENE_PARSER_USER_PROMPT,
                    "images": [image_path],
                },
            ],
        )
        return {"scene_description": response["message"]["content"]}

    except Exception as e:
        return {
            "scene_description": "",
            "errors": [f"Scene Parser error: {type(e).__name__}: {e}"],
        }
