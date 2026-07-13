from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel

from council.llm import get_llm, get_thinking_prefix

_llm = get_llm("orchestrator")


class OrchestratorOutput(BaseModel):
    road_marking_extraction: str  # MUST quote verbatim: "CENTER line: [exact quote], EDGE lines: [exact quote]"
    linguistic_prompt: str
    landscape_prompt: str
    botanics_prompt: str
    regulatory_prompt: str
    infrastructure_prompt: str
    cultural_prompt: str


async def run(general_description: str, crop_descriptions: list[str]) -> OrchestratorOutput:
    crops_block = "\n".join(f"- Crop {i+1}: {c}" for i, c in enumerate(crop_descriptions))
    think = get_thinking_prefix("orchestrator")
    prompt = f"""\
        {think}
        You are an orchestrator preparing inputs for six specialist agents that identify countries
        from street-level image descriptions.

        You are the ONLY link between the source descriptions and the agents.
        The agents will NEVER see the original image - they rely ENTIRELY on what you write.
        Every detail you omit, swap, or distort is permanently lost and causes wrong predictions.

        --------------------------------------------------------------------
        STEP 1 - FILL road_marking_extraction FIRST (before any agent prompt)
        --------------------------------------------------------------------

        You MUST fill the road_marking_extraction field BEFORE writing any agent prompt.
        This forces you to locate and copy road marking facts from the source VERBATIM.

        Scan the source for any mention of road lines/markings. For each line mentioned,
        copy the EXACT phrase from the source that describes it. Do not rephrase or reinterpret.

        Format your extraction as:
          "Source says about center/dividing line: [exact quote].
           Source says about edge/shoulder lines: [exact quote].
           Full source sentence: [paste the complete sentence(s) containing these facts]."

        CRITICAL: the source text is authoritative. If the source says a color is on the
        "edge", it IS the edge - even if that seems unusual to you. Do NOT "correct" it.
        Your job is to COPY, not to interpret.

        After filling road_marking_extraction, use those EXACT quoted facts - unchanged -
        in the regulatory_prompt and infrastructure_prompt. Do not re-derive or reword them.

        --------------------------------------------------------------------
        STEP 2 - WRITE EACH AGENT'S PROMPT (using quoted facts from Step 1)
        --------------------------------------------------------------------

        Rules for ALL agent prompts:
        - COPY phrases from the source. Do not paraphrase or summarize.
        - Preserve every color, position, material, shape, size, count, and text transcription.
        - Preserve spatial labels: "center" stays "center", "edge" stays "edge",
          "left" stays "left", "foreground" stays "foreground".
        - Include ALL details. Redundancy across agents is fine; omission is not.
        - If unsure whether a detail is relevant to an agent, INCLUDE it.

        GENERAL DESCRIPTION:
        {general_description}

        CROP DESCRIPTIONS:
        {crops_block}

        -- Linguistic Agent --
        Output ONLY the raw text snippets visible on signs, posters, labels, or boards.
        No explanations - just the literal text. If multiple snippets, separate with comma.
        If no text is visible, say "No visible text."

        -- Landscape Agent --
        Pass through ALL terrain, geography, and environmental details verbatim.
        Include: terrain type, soil color/texture, vegetation density and spatial distribution,
        horizon features (hills, mesas, ridges - shape, distance, and position in scene),
        sky conditions, water features, elevation indicators.
        Preserve exact left/right/foreground/background positions of every feature.

        -- Botanics Agent --
        Pass through ALL vegetation details verbatim - every plant description, leaf shapes,
        sizes, growth habits, bark textures, flower/fruit details, identifiable species.
        Include spatial distribution (where in scene, density, canopy layers).

        -- Regulatory Agent --
        Pass through ALL regulatory details. For road markings, copy EXACTLY from your
        road_marking_extraction - do not reword. Use this format:
          CENTER line: [paste what you extracted about the center line]
          EDGE lines: [paste what you extracted about the edge lines]
        Also include: driving side, sign shapes/colors/text, license plate format/colors,
        traffic light config, utility pole types, barrier styles, crosswalk patterns.

        -- Infrastructure Agent --
        Pass through ALL vehicle descriptions (type, brand, color, position, plate details),
        road surface details (material, condition, width), lane markings copied EXACTLY from
        your road_marking_extraction with position labels preserved (center vs edge),
        building architecture (styles, materials, roof shapes, fence types, wall materials,
        compound layouts), utility infrastructure (pole material, wire count, crossbar style),
        street furniture (lamp styles, sidewalk materials, bollards, trash bins).

        -- Cultural Agent --
        Pass through ALL cultural and human context details - people's clothing styles
        (traditional garments, work wear, accessories), murals and artwork on buildings
        (subject matter, style, colors), shop and business types (pharmacy, food stall,
        market, repair shop), religious buildings or symbols, street vendors and their goods,
        decorative elements (flags, banners, shrines, statues), building decoration styles
        (painted facades, ornamental woodwork, balcony styles), and any other cultural
        indicators visible in the scene.
        If NO cultural or human context clues are present in the descriptions, set this to
        exactly "No cultural clues visible."

        Return all six prompts."""

    return await _llm.with_structured_output(OrchestratorOutput).ainvoke([HumanMessage(content=prompt)])
