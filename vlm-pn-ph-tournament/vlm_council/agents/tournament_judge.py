"""Tournament Judge, pairwise multimodal comparator for the v12 bracket.

Given the streetview, reference images for two candidate countries, the RAG
context (driving-side text, road-line text, recovery warnings), and per-country
specialist evidence, the judge picks ONE winner and explains why.

Output JSON contract:
    {"winner": "<country_a or country_b>",
     "reasoning": "<2-3 sentences>",
     "coordinates": "<lat>, <lon>"}     # always required on the FINAL match

Structured output is enforced server-side via vLLM's OpenAI-compatible
``response_format={"type": "json_schema", ...}``. The schema marks
``coordinates`` as required for the final match so the model cannot omit it.

Thinking is OFF by default (controlled by VLM_JUDGE_THINKING env var, same as
other judge calls, config.judge_thinking). Even when ON, we strip <think>
tags before parsing.
"""

from __future__ import annotations

import json
import os
import re

from langchain_core.messages import SystemMessage

from vlm_council.image_utils import build_vlm_message_multi, encode_image
from vlm_council.llm import get_vlm
from vlm_council.rag.keyed_lookup import Reference


_THINKING_ENABLED = os.environ.get("VLM_JUDGE_THINKING", "false").lower() in ("true", "1", "yes")
_THINK_PREFIX = "<|think|>\n" if _THINKING_ENABLED else ""

# Regex for "lat, lon", accepts signed decimals, optional whitespace.
_COORD_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


def _judge_schema(is_final: bool, country_a: str, country_b: str) -> dict:
    """Build a JSON schema for the judge response.

    Constrains ``winner`` to one of the two candidate countries, makes
    coordinates required for the final match.
    """
    properties = {
        "winner": {
            "type": "string",
            "enum": [country_a, country_b],
            "description": "The winning country, must be exactly country A or country B.",
        },
        "reasoning": {
            "type": "string",
            "description": "2-3 sentences citing specific visual matches.",
        },
    }
    required = ["winner", "reasoning"]
    if is_final:
        properties["coordinates"] = {
            "type": "string",
            "pattern": r"^-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?$",
            "description": "Latitude and longitude inside the winning country, formatted as 'lat, lon'.",
        }
        required.append("coordinates")

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _strip_think_tags(text: str) -> str:
    """Strip thinking-chain prefixes; return only the post-think response."""
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


def _parse_judge_json(raw: str) -> dict | None:
    text = _strip_think_tags(raw).strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    # Try whole text first (guided JSON returns clean output)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


SYSTEM_PROMPT = """\
You are the Tournament Judge of a GeoGuessr council. You are comparing TWO candidate countries head-to-head and must pick ONE winner. Both candidates are equally plausible going into this match, neither is favoured by ordering, label, or position. Evaluate the visual evidence symmetrically.

You see:
- The streetview image to identify (first image).
- Reference images. EACH reference image is captioned with the country it comes from in the form "[<Country Name>, <category>]" (e.g. "[Czech Republic, bollards]"). A reference image ONLY proves something about the country named in its own caption, never about any other candidate. If a candidate has zero reference images, you have NO visual reference evidence for that candidate; do not invent any.
- A RAG context block with driving-side observation, road-line observation, and any prefilter warnings.
- Per-country specialist evidence summary.

Decision rules:
1. Compare visual cues in the streetview against the reference images for each country (bollards, license plates, signs, road lines, sidewalks, etc.). Apply the same scrutiny to both.
2. STRICT attribution: if the streetview matches an image captioned "[Country X, ...]", that is evidence FOR Country X, never for the other candidate. Do not transfer a visual match from one country's references to the other.
3. If a RAG WARNING is present (e.g. "road-line check returned MISMATCH for ALL"), TREAT the regulatory observation as UNRELIABLE for that feature and rely on the other visual cues.
4. Without a warning, contradictions to driving-side / road-line observations are STRONG signals.
5. Pick the country whose reference images and infrastructure profile match the streetview MOST closely.
6. If references are missing or unhelpful for a side, fall back to specialist evidence and reasoning, do NOT borrow another country's references to support that side.
7. The order of "Country A" / "Country B" in the prompt is arbitrary, do NOT prefer one over the other for any reason other than visual evidence.

You MUST pick exactly one winner, no ties, no abstentions.

Respond with JSON only:
{"winner": "<exactly the name of the chosen country>", "reasoning": "<2-3 sentences citing specific visual matches and naming the country whose reference image matched>"}\
"""

FINAL_SYSTEM_PROMPT = """\
You are the Tournament Judge of a GeoGuessr council. This is the FINAL match of the bracket, pick the winner AND give coordinates. Both candidates are equally plausible going into this match, evaluate the visual evidence symmetrically and do NOT prefer one position over the other.

You see:
- The streetview image to identify (first image).
- Reference images. EACH reference image is captioned with the country it comes from in the form "[<Country Name>, <category>]" (e.g. "[Poland, bollards]"). A reference image ONLY proves something about the country named in its own caption, never about any other candidate. If a candidate has zero reference images, you have NO visual reference evidence for that candidate; do not invent any.
- A RAG context block with driving-side observation, road-line observation, and any prefilter warnings.
- Per-country specialist evidence summary.

Decision rules:
1. Compare visual cues in the streetview against the reference images for each country. Apply the same scrutiny to both.
2. STRICT attribution: if the streetview matches an image captioned "[Country X, ...]", that is evidence FOR Country X, never for the other candidate. Do not transfer a visual match from one country's references to the other.
3. If a RAG WARNING is present, the corresponding regulatory observation is UNRELIABLE, rely on other cues.
4. The order of "Country A" / "Country B" is arbitrary, do NOT prefer one over the other for any reason other than visual evidence.
5. Pick exactly ONE winner.
6. Estimate coordinates inside the winning country, using any visible terrain / settlement / landscape clues to refine the latitude and longitude. The coordinates field is REQUIRED, provide a real lat/lon point inside the winning country, never "0, 0".

Respond with JSON only:
{"winner": "<exactly the name of the chosen country>", "reasoning": "<2-3 sentences citing specific visual matches and naming the country whose reference image matched>", "coordinates": "<lat>, <lon>"}\
"""


def _refs_to_payload(
    refs: list[Reference],
) -> list[tuple[str, str, str]]:
    """Encode references into (b64, mime, caption) triples for build_vlm_message_multi.

    Caption embeds the country name directly so the judge cannot misattribute
    a visual match to the wrong candidate. Each image is "owned" by exactly
    the country in its caption.
    """
    out: list[tuple[str, str, str]] = []
    for r in refs:
        try:
            b64, mime = encode_image(r.image_path)
        except (OSError, FileNotFoundError):
            continue
        caption = f"[{r.country}, {r.category}]"
        out.append((b64, mime, caption))
    return out


def _format_specialist_evidence(state: dict, country_a: str, country_b: str) -> str:
    """Compact summary of which specialists rated each country and how.

    Reads hypothesis_evaluations and pulls all rows where hypothesis_id matches
    country_a or country_b. Falls back to reasoning lines from agent assessments.
    """
    lines: list[str] = []
    evals = state.get("hypothesis_evaluations", []) or []
    target_ids = {
        "country_" + country_a.lower().replace(" ", "_"): country_a,
        "country_" + country_b.lower().replace(" ", "_"): country_b,
    }
    for e in evals:
        hid = e.get("hypothesis_id", "")
        if hid in target_ids:
            country = target_ids[hid]
            lines.append(
                f"  [{e.get('agent_name', '?')}] {country}: {e.get('confidence', '?')}, {e.get('reasoning', '')}"
            )
    if lines:
        return "Specialist Evidence:\n" + "\n".join(lines)
    return "Specialist Evidence: (none recorded)"


def _format_rag_block(
    driving_side: str | None,
    road_line: str | None,
    warnings: list[str],
) -> str:
    parts = []
    parts.append(f"Driving side observed: {driving_side or 'UNCLEAR'}")
    parts.append(f"Road line observed: {road_line or '(none reported)'}")
    if warnings:
        parts.append("RAG WARNINGS (treat the affected observation as unreliable):")
        for w in warnings:
            parts.append(f"  - {w}")
    else:
        parts.append("RAG WARNINGS: (none)")
    return "\n".join(parts)


def build_match_message(
    image_b64: str,
    image_mime: str,
    country_a: str,
    country_b: str,
    refs_a: list[Reference],
    refs_b: list[Reference],
    driving_side: str | None,
    road_line: str | None,
    warnings: list[str],
    specialist_block: str,
    is_final: bool,
):
    """Assemble the multimodal HumanMessage for one tournament match."""
    refs_payload = _refs_to_payload(refs_a) + _refs_to_payload(refs_b)

    rag_block = _format_rag_block(driving_side, road_line, warnings)
    instruction_tail = (
        "Provide winner, reasoning, and coordinates as JSON."
        if is_final
        else "Provide winner and reasoning as JSON."
    )

    refs_summary = (
        f"Reference images provided: {len(refs_a)} from {country_a}, "
        f"{len(refs_b)} from {country_b}. "
        "Each image's caption names its country of origin, only attribute a "
        "visual match to the country named in its caption."
    )

    text = (
        f"Country A: {country_a}\n"
        f"Country B: {country_b}\n\n"
        f"{refs_summary}\n\n"
        f"{rag_block}\n\n"
        f"{specialist_block}\n\n"
        f"Compare the streetview against the reference images and decide. "
        f"{instruction_tail}"
    )
    return build_vlm_message_multi((image_b64, image_mime), refs_payload, text)


def _validate_coords(raw: str) -> str:
    """Return ``lat, lon`` string if parseable and non-zero, else empty.

    Guided JSON (strict schema with regex pattern) should produce a valid
    lat/lon ~always; this is just a guard against the rare parse failure or
    the literal 0,0 sentinel.
    """
    if not raw:
        return ""
    m = _COORD_RE.match(raw)
    if not m:
        return ""
    lat, lon = float(m.group(1)), float(m.group(2))
    if abs(lat) < 1e-6 and abs(lon) < 1e-6:
        return ""
    return f"{lat}, {lon}"


_CATEGORY_DESCRIPTIONS = {
    "bollards": "bollards, delineator posts, road barrier posts (plastic/metal/concrete posts along roads)",
    "license_plates": "vehicle license/number plates visible on cars, trucks, motorcycles",
    "signs_stop": "stop signs (octagonal red signs)",
    "signs_yield": "yield/give-way signs (inverted triangle signs)",
    "signs_chevrons": "chevron signs (arrow boards on curves showing direction)",
    "signs_pedestrian": "pedestrian crossing signs (signs showing walking person symbol)",
    "utility_poles": "utility/power/telephone poles (wooden, concrete, or metal poles carrying wires)",
    "traffic_lights": "traffic lights/signals (mounted at intersections)",
    "post_boxes": "post boxes / mailboxes (public mail collection boxes)",
    "sidewalks": "sidewalk/pavement/curb patterns (colored or textured walking surfaces)",
    "signs_speed": "speed limit signs",
    "signs_bus_stop": "bus stop signs or shelters",
    "signs_directions": "directional/destination signs (blue/green signs pointing to places)",
    "signs_railway_crossing": "railway/railroad crossing signs or barriers",
    "signs_back": "the back side of road signs (visible color, stickers, patterns)",
    "signs_posts": "sign posts/poles themselves (the metal or wooden pole holding the sign)",
    "signs_road_numbering": "road number signs (route markers, highway numbers)",
    "signs_animal_warning": "animal warning signs (deer, kangaroo, moose, cattle crossing)",
    "road_lines": "road line markings (center lines, edge lines, their colors)",
}

_FILTER_SYSTEM = (
    "You compare reference images against artifacts in a street-view image. "
    "Report ONLY clear, exact visual matches where specific details are identifiable in both. "
    "Be strict, when in doubt, it is NOT a match."
)

_FILTER_INSTRUCTION = (
    "\nCompare each numbered reference against the artifacts visible in the street-view "
    "image above.\n\n"
    "For each reference, decide: MATCH or NO MATCH.\n"
    "- MATCH = the artifact in the street-view and the reference are BOTH clearly "
    "identifiable and share the SAME shape, colors, pattern, and design.\n"
    "- NO MATCH = the artifact is different, OR either image is too blurry/small/distant "
    "to confirm specific details.\n\n"
    "If you cannot clearly see the distinguishing details in BOTH the reference AND the "
    "street-view image, it is NO MATCH. 'Looks similar' is NOT a match.\n\n"
    "Output format (single line):\n"
    "Matches: <comma-separated indices of matching references, or 'none'>"
)

_VERIFY_SYSTEM = (
    "You are a strict visual verification judge. A previous model claimed a reference image "
    "matches an artifact in a street-view photo. Verify if this is truly an exact match. "
    "Reject unless BOTH images are clear enough to confirm identical details."
)

_VERIFY_INSTRUCTION = (
    "\nA previous analysis claimed this reference matches an artifact in the street-view image. "
    "Verify this claim by looking closely at both.\n\n"
    "Check:\n"
    "1. Can you clearly identify the specific artifact in the street-view image?\n"
    "2. Can you clearly identify the details in the reference image?\n"
    "3. Do they share the SAME specific shape, colors, pattern, and design?\n\n"
    "If ANY of these are uncertain, the artifact is too small, blurry, distant, or "
    "the details don't actually match, the claim is REJECTED.\n\n"
    "Output:\n"
    "Verdict: CONFIRMED or REJECTED\n"
    "Reason: <one sentence>"
)


async def filter_visible_refs(
    image_b64: str,
    image_mime: str,
    country: str,
    refs: list[Reference],
    batch_size: int = 8,
    llm=None,
) -> list[Reference]:
    """Drop references whose feature category isn't visibly present in the streetview.

    Reimplementation of v10's ``run_evidence_filter``. Each batch sees the streetview
    plus up to ``batch_size`` numbered refs and returns the indices that visibly match.
    """
    if not refs:
        return []
    if llm is None:
        llm = get_vlm("judge")

    matched: list[Reference] = []
    for batch_start in range(0, len(refs), batch_size):
        batch = refs[batch_start:batch_start + batch_size]

        # Build message manually (v10 style): streetview first, then
        # "--- Reference images ---" header, then each ref label+image,
        # then the instruction at the end.
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
            {"type": "text", "text": f"\n--- Reference images from {country} ---"},
        ]
        valid_indices: list[int] = []
        for i, r in enumerate(batch):
            try:
                b64, mime = encode_image(r.image_path)
            except (OSError, FileNotFoundError):
                continue
            label = f"[{i}] [{r.category}]"
            if r.properties:
                label += f" ({', '.join(r.properties)})"
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            valid_indices.append(i)

        if not valid_indices:
            continue

        content.append({"type": "text", "text": _FILTER_INSTRUCTION})

        from langchain_core.messages import HumanMessage as _HM
        msg = _HM(content=content)
        try:
            response = await llm.ainvoke([SystemMessage(content=_FILTER_SYSTEM), msg])
        except Exception:
            matched.extend(batch)
            continue

        raw = _strip_think_tags(response.content).strip()
        for line in raw.split("\n"):
            if not line.lower().strip().startswith("matches:"):
                continue
            val = line.split(":", 1)[1].strip().lower()
            if val == "none" or not val:
                break
            for part in re.split(r"[,\s]+", val):
                part = part.strip()
                if part.isdigit():
                    idx = int(part)
                    if 0 <= idx < len(batch):
                        matched.append(batch[idx])
            break

    return matched


async def identify_visible_features(
    image_b64: str,
    image_mime: str,
    available_categories: list[str],
    db_bollard_props: dict[str, dict] | None = None,
    llm=None,
) -> dict:
    """Identify which RAG feature categories are visibly present in the streetview.

    Args:
        db_bollard_props: maps country → {"materials": [...], "colors": [...]} from the
            bollard_country_summary.json DB. When provided, the bollard prompt is constrained
            to only the material/color terms that actually appear in the DB for the candidate
            countries, so the model only reports values that can be matched against real refs.

    Returns dict with:
      categories: list[str], detected category names from available_categories
      bollard_properties: dict {"materials": [...], "colors": [...]}
      road_lines_observed: str
    """
    import re as _re
    if llm is None:
        llm = get_vlm("judge")

    # Compute DB-constrained material/color vocab for bollard prompt.
    # Union across all candidate countries so the model can detect any bollard
    # type that exists in the DB.
    if db_bollard_props:
        all_materials: set[str] = set()
        all_colors: set[str] = set()
        for props in db_bollard_props.values():
            all_materials.update(props.get("materials", []))
            all_colors.update(props.get("colors", []))
        mat_line = " / ".join(sorted(all_materials)) if all_materials else "Metal / Concrete / Plastic / Wood / Rock"
        col_line = " / ".join(sorted(all_colors)) if all_colors else "Red / White / Black / Blue / Yellow / Grey / Orange / Green / Brown / Pink"
    else:
        mat_line = "Metal / Concrete / Plastic / Wood / Rock"
        col_line = "Red / White / Black / Blue / Yellow / Grey / Orange / Green / Brown / Pink"

    cat_display = "\n".join(
        f"  - {c}: {_CATEGORY_DESCRIPTIONS.get(c, c)}"
        for c in available_categories
    )
    msg = build_vlm_message_multi(
        (image_b64, image_mime),
        [],
        (
            f"Which of these features are VISIBLE in this image?\n\n{cat_display}\n\n"
            "For BOLLARDS, also describe using ONLY these terms:\n"
            f"  Material: {mat_line}\n"
            f"  Colors: {col_line}\n\n"
            "Output format:\n"
            "Categories: <comma-separated list from above, or 'none'>\n"
            "Bollard: <Material> <Color1> <Color2> ... (only if bollards visible)\n"
            "Road_lines: <colors visible, e.g. 'yellow center, white edge'> (only if road lines visible)\n\n"
            "If NO features are visible:\nCategories: none"
        ),
    )
    system = SystemMessage(content=(
        "You examine a street-view image to identify visible man-made infrastructure. "
        "Be thorough, report EVERY category where you can see that type of feature."
    ))
    try:
        raw = _strip_think_tags((await llm.ainvoke([system, msg])).content)
    except Exception:
        return {"categories": [], "bollard_properties": {}, "road_lines_observed": ""}

    # Parse valid material/color terms from the actual DB vocab (not a hardcoded list).
    valid_materials = (
        {m for props in db_bollard_props.values() for m in props.get("materials", [])}
        if db_bollard_props
        else {"Metal", "Concrete", "Plastic", "Wood", "Rock"}
    )
    valid_colors = (
        {c for props in db_bollard_props.values() for c in props.get("colors", [])}
        if db_bollard_props
        else {"Red", "White", "Black", "Blue", "Yellow", "Grey", "Orange", "Green", "Brown", "Pink"}
    )

    result: dict = {"categories": [], "bollard_properties": {}, "road_lines_observed": ""}
    for line in raw.strip().split("\n"):
        low = line.lower().strip()
        if low.startswith("categories:"):
            cats_str = line.split(":", 1)[1].strip()
            if cats_str.lower() != "none":
                cats = [c.strip() for c in cats_str.split(",")]
                result["categories"] = [c for c in cats if c in available_categories]
        elif low.startswith("bollard:"):
            desc = line.split(":", 1)[1].strip()
            tokens = [t.strip().title() for t in _re.split(r"[,\s/]+", desc) if t.strip()]
            result["bollard_properties"] = {
                "materials": [t for t in tokens if t in valid_materials],
                "colors": [t for t in tokens if t in valid_colors],
            }
        elif low.startswith("road_lines:") or low.startswith("road lines:"):
            result["road_lines_observed"] = line.split(":", 1)[1].strip()
    return result


async def verify_ref_match(
    image_b64: str,
    image_mime: str,
    ref: Reference,
    llm=None,
) -> bool:
    """Verify a single claimed ref match against the streetview.

    Returns True only if CONFIRMED. Mirrors v10's run_evidence_verify.
    Fails open (returns True) on LLM error to avoid false drops.
    """
    if llm is None:
        llm = get_vlm("judge")

    label = f"[{ref.category}]"
    if ref.properties:
        label += f" ({', '.join(ref.properties)})"

    try:
        ref_b64, ref_mime = encode_image(ref.image_path)
    except (OSError, FileNotFoundError):
        return False

    payload = [(ref_b64, ref_mime, f"Reference to verify: {label}")]
    msg = build_vlm_message_multi((image_b64, image_mime), payload, _VERIFY_INSTRUCTION)
    try:
        raw = _strip_think_tags((await llm.ainvoke([SystemMessage(content=_VERIFY_SYSTEM), msg])).content)
    except Exception:
        return True  # fail-open

    for line in raw.strip().split("\n"):
        if line.lower().strip().startswith("verdict:"):
            return "confirmed" in line.split(":", 1)[1].lower()
    return False



async def judge_match(
    image_b64: str,
    image_mime: str,
    country_a: str,
    country_b: str,
    refs_a: list[Reference],
    refs_b: list[Reference],
    driving_side: str | None,
    road_line: str | None,
    warnings: list[str],
    specialist_block: str,
    is_final: bool,
    llm=None,
) -> dict:
    """Run one pairwise judge call. Returns a parsed dict (winner/reasoning[/coordinates]).

    Uses vLLM-guided JSON via ``response_format={"type": "json_schema", ...}``
    so the model cannot omit required fields. Falls back to country_a if the
    response is unparseable; coordinates fall back to empty string on the rare
    parse failure.
    """
    if llm is None:
        llm = get_vlm("judge")

    system = FINAL_SYSTEM_PROMPT if is_final else SYSTEM_PROMPT
    msg = build_match_message(
        image_b64=image_b64,
        image_mime=image_mime,
        country_a=country_a,
        country_b=country_b,
        refs_a=refs_a,
        refs_b=refs_b,
        driving_side=driving_side,
        road_line=road_line,
        warnings=warnings,
        specialist_block=specialist_block,
        is_final=is_final,
    )

    schema = _judge_schema(is_final, country_a, country_b)
    schema_name = "TournamentFinalDecision" if is_final else "TournamentMatchDecision"
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True,
        },
    }

    bound = llm.bind(response_format=response_format)
    response = await bound.ainvoke([SystemMessage(content=_THINK_PREFIX + system), msg])
    parsed = _parse_judge_json(response.content) or {}

    raw_winner = str(parsed.get("winner", "") or "").strip()
    # Strict normalization: case-insensitive compare against the two options.
    # If the model returns junk we surface it as winner="" instead of silently
    # defaulting to country_a, the caller decides how to break the tie.
    winner = ""
    if raw_winner.lower() == country_b.lower():
        winner = country_b
    elif raw_winner.lower() == country_a.lower():
        winner = country_a
    elif raw_winner:
        rl = raw_winner.lower()
        if country_b.lower() in rl and country_a.lower() not in rl:
            winner = country_b
        elif country_a.lower() in rl and country_b.lower() not in rl:
            winner = country_a

    out = {
        "winner": winner,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }
    if is_final:
        raw_coords = str(parsed.get("coordinates", "")).strip()
        out["coordinates"] = _validate_coords(raw_coords)
    return out


async def judge_match_symmetric(
    image_b64: str,
    image_mime: str,
    country_a: str,
    country_b: str,
    refs_a: list[Reference],
    refs_b: list[Reference],
    driving_side: str | None,
    road_line: str | None,
    warnings: list[str],
    specialist_block: str,
    is_final: bool,
    pool_rank_a: int,
    pool_rank_b: int,
    llm=None,
) -> dict:
    """Run the same pairwise comparison twice with swapped positions, then
    aggregate. This neutralises position-bias in the underlying judge.

    Both calls run concurrently (asyncio.gather). Outcomes:
      • Both runs agree on the winner → robust, return that winner.
      • Runs disagree (= judge picked by position, not evidence) → fall back
        to pool-rank (lower rank index = higher pool seed wins). On final
        match, take the coordinates from whichever run picked the tie-break
        winner; if neither did, fall back to forward-run coords or empty.
      • One run returned an empty winner (parse fail) → trust the other.
      • Both empty → fall back to pool-rank entirely.

    The returned dict includes ``forward`` and ``reverse`` sub-results plus an
    ``agreement`` flag so the eval pipeline can quantify residual bias.
    """
    import asyncio

    forward_task = asyncio.create_task(judge_match(
        image_b64=image_b64, image_mime=image_mime,
        country_a=country_a, country_b=country_b,
        refs_a=refs_a, refs_b=refs_b,
        driving_side=driving_side, road_line=road_line,
        warnings=warnings, specialist_block=specialist_block,
        is_final=is_final, llm=llm,
    ))
    reverse_task = asyncio.create_task(judge_match(
        image_b64=image_b64, image_mime=image_mime,
        country_a=country_b, country_b=country_a,
        refs_a=refs_b, refs_b=refs_a,
        driving_side=driving_side, road_line=road_line,
        warnings=warnings, specialist_block=specialist_block,
        is_final=is_final, llm=llm,
    ))
    forward, reverse = await asyncio.gather(forward_task, reverse_task)

    fwd_winner = forward.get("winner") or ""
    rev_winner = reverse.get("winner") or ""

    # Tie-break: pool-rank, lower index = higher seed
    pool_rank_winner = country_a if pool_rank_a <= pool_rank_b else country_b

    if fwd_winner and rev_winner and fwd_winner == rev_winner:
        winner = fwd_winner
        agreement = "agree"
    elif fwd_winner and not rev_winner:
        winner = fwd_winner
        agreement = "forward_only"
    elif rev_winner and not fwd_winner:
        winner = rev_winner
        agreement = "reverse_only"
    elif not fwd_winner and not rev_winner:
        winner = pool_rank_winner
        agreement = "both_empty"
    else:
        # Disagreement: judge is unsure, position-biased, or both
        winner = pool_rank_winner
        agreement = "disagree"

    # Reasoning: prefer the run whose winner matches the resolved winner;
    # fall back to forward if both runs disagree with the resolution.
    if forward.get("winner") == winner:
        reasoning = forward.get("reasoning", "")
    elif reverse.get("winner") == winner:
        reasoning = reverse.get("reasoning", "")
    else:
        reasoning = (
            forward.get("reasoning", "")
            or reverse.get("reasoning", "")
            or f"pool-rank tie-break: {winner}"
        )

    out = {
        "winner": winner,
        "reasoning": reasoning,
        "agreement": agreement,
        "forward": forward,
        "reverse": reverse,
    }
    if is_final:
        # Pick coords from whichever run chose the resolved winner; fall back
        # in order of preference. judge_match's _validate_coords returns ""
        # on bad input, so this surfaces "" cleanly to the caller.
        if forward.get("winner") == winner and forward.get("coordinates"):
            out["coordinates"] = forward.get("coordinates", "")
        elif reverse.get("winner") == winner and reverse.get("coordinates"):
            out["coordinates"] = reverse.get("coordinates", "")
        else:
            out["coordinates"] = (
                forward.get("coordinates", "")
                or reverse.get("coordinates", "")
                or ""
            )
    return out
