"""Detail Focusser node - generates detailed descriptions of cropped regions.

Takes each cropped image and produces a focused description of the target object.
Includes integrated validation: if the crop does not contain the expected object,
the VLM responds with NOT_FOUND and the crop is discarded.

Requests are dispatched in parallel to leverage Ollama's OLLAMA_NUM_PARALLEL
batching on a single GPU.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ollama import Client

from vision_pipeline.config import PipelineConfig
from vision_pipeline.state import PipelineState

NOT_FOUND_MARKER = "NOT_FOUND"

DETAIL_FOCUSSER_SYSTEM_PROMPT = """\
You are an image description assistant. You will receive a cropped image
that should contain a specific object.

If the named object IS visible in the crop:
- Describe that object in maximum detail.
- Also describe any relevant context visible around the object in the crop
  (e.g. if the crop shows a vehicle, also note the road surface, nearby signs, etc.)
- Transcribe any visible text exactly, noting the script/alphabet.
- Do not interpret, speculate, or guess locations.
- Do not mention countries, cities, or regions.
- Do not guess features that are not clearly visible.

Describe based on the object type:
- SIGNS: text content (exact transcription), script, shape, colors, pictograms, mounting
  style, condition, size estimate, material (metal, wood, plastic)
- LICENSE PLATES: color scheme, character format, any visible characters, front vs rear,
  plate shape (rectangular wide, square, narrow), any stickers or emblems
- POLES: material (wood/concrete/metal), color, crossbar shape, insulators, wire count,
  any painted markings or bands
- STREET LAMPS: pole material and color, shape (straight/curved/ornamental), lamp head
  style (modern LED, globe, cobra-head, lantern), single or multi-arm
- BOLLARDS/BARRIERS: color bands, shape, reflectors, material, profile
- TRAFFIC LIGHTS: lens count, orientation, arrow indicators, housing color
- VEHICLES: type (car/SUV/truck/van/pickup/bus/motorcycle), size class (subcompact/compact/
  midsize/full-size), body style (hatchback/sedan/wagon/coupe/crossover/minivan), color,
  brand/model if recognizable, any text/logos/badges visible, steering side if visible,
  cargo setup, distinctive features (roof rack, taxi sign, livery, damage)
- MOTORCYCLES: type, brand if recognizable, cargo attachments, sidecar, rider setup
- BUSES/TAXIS: color scheme, livery, route number, operator name, roof sign
- ROAD MARKINGS: exact color, pattern (solid/dashed/dotted), width, condition, position
  on road surface (center, edge, crosswalk)
- BUILDINGS: wall material and finish (rendered, raw brick, stone, cladding), roof material
  and style, window type (shuttered, casement, sash), construction quality, any visible
  text or signage on facade
- WALLS/FENCES: material (concrete block, rendered, stone, brick, wood, metal), height,
  color, construction method, any decorative elements
- SIDEWALKS: surface material (concrete, tile, brick, cobblestone), pattern, color, width
- OTHER: colors, shape, material, condition, and any markings

If the named object is NOT visible in the crop:
- Respond with exactly: NOT_FOUND\
"""

DETAIL_FOCUSSER_USER_PROMPT = """\
Expected object: "{detail_name}"

If you can see a {detail_name} in this image, describe it in detail.
If not, respond with: NOT_FOUND\
"""


def _process_single_detail(
    client: Client,
    config: PipelineConfig,
    detail: dict,
) -> tuple[dict, str | None]:
    """Describe a single crop with integrated validation. Returns (updated_detail, error_or_none)."""
    d = dict(detail)
    crop_path = d.get("crop_path")

    if crop_path is None or not Path(crop_path).is_file():
        d["focused_description"] = None
        return d, None

    try:
        response = client.chat(
            model=config.vision_model,
            messages=[
                {"role": "system", "content": DETAIL_FOCUSSER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": DETAIL_FOCUSSER_USER_PROMPT.format(
                        detail_name=d["name"],
                    ),
                    "images": [crop_path],
                },
            ],
        )
        content = response["message"]["content"].strip()

        # Check if the VLM says the object is not in the crop
        if content.upper().startswith(NOT_FOUND_MARKER):
            d["focused_description"] = None
            d["bbox"] = None
            d["crop_path"] = None
            return d, f"Bbox validation failed for '{d['name']}': crop does not contain the expected object"

        d["focused_description"] = content
        return d, None

    except Exception as e:
        d["focused_description"] = None
        return d, f"Detail Focusser error for '{d['name']}': {type(e).__name__}: {e}"


def detail_focusser(
    state: PipelineState,
    client: Client,
    config: PipelineConfig,
) -> dict:
    """Produce detailed descriptions for each cropped detail image.

    Dispatches requests in parallel using threads. Each request combines
    validation and description in a single VLM call.
    Ollama batches concurrent requests on the GPU (OLLAMA_NUM_PARALLEL).

    Reads:  state["details"] (each with crop_path)
    Writes: state["details"] (enriched with focused_description field)
    """
    details = state.get("details", [])

    if not details:
        return {"details": details}

    # Separate croppable from non-croppable details
    croppable = []
    non_croppable = []
    for detail in details:
        if detail.get("crop_path") and Path(detail["crop_path"]).is_file():
            croppable.append(detail)
        else:
            d = dict(detail)
            d["focused_description"] = None
            non_croppable.append(d)

    if not croppable:
        return {"details": non_croppable}

    # Process croppable details in parallel
    updated = list(non_croppable)
    errors = []

    process_fn = lambda d: _process_single_detail(client, config, d)

    with ThreadPoolExecutor(max_workers=len(croppable)) as executor:
        future_to_detail = {
            executor.submit(process_fn, detail): detail
            for detail in croppable
        }

        for future in as_completed(future_to_detail):
            result_detail, error = future.result()
            updated.append(result_detail)
            if error:
                errors.append(error)

    # Restore original order
    name_order = [d["name"] for d in details]
    updated_by_name = {d["name"]: d for d in updated}
    ordered = [updated_by_name[name] for name in name_order if name in updated_by_name]

    result: dict = {"details": ordered}
    if errors:
        result["errors"] = errors
    return result
