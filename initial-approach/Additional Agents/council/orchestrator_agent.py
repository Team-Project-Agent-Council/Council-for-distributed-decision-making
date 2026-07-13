from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel

from council.llm import get_llm, get_thinking_prefix

_llm = get_llm("orchestrator")


class OrchestratorOutput(BaseModel):
    linguistic_prompt: str
    landscape_prompt: str
    botanics_prompt: str
    regulatory_prompt: str
    infrastructure_prompt: str
    climate_prompt: str


async def run(general_description: str, crop_descriptions: list[str]) -> OrchestratorOutput:
    crops_block = "\n".join(f"- Crop {i+1}: {c}" for i, c in enumerate(crop_descriptions))
    think = get_thinking_prefix("orchestrator")
    prompt = f"""\
        {think}
        You are an orchestrator preparing inputs for four specialist agents.

        Given the following image descriptions, write a specific prompt for each agent:

        GENERAL DESCRIPTION:
        {general_description}

        CROP DESCRIPTIONS:
        {crops_block}

        For the Linguistic Agent: output ONLY the raw text snippets visible on signs, posters, labels, or boards in the descriptions. No explanations, no framing - just the literal text as it would appear in the image. If multiple snippets, separate them with a comma.

        For the Landscape Agent: summarize all terrain, geography, infrastructure, climate, and environmental features so the agent can identify the landscape type or region.

        For the Botanics Agent: list all visible plant species, trees, shrubs, flowers, crops, or distinctive vegetation. Use scientific names where possible, otherwise common names. Focus on species that are geographically distinctive or regionally restricted rather than cosmopolitan weeds or grass.

        For the Regulatory Agent: describe all visible man-made infrastructure and regulatory elements - driving side (which side of the road vehicles drive on), road sign shapes and colors, center line color (yellow or white), license plate formats, utility pole types (wooden/concrete/steel), traffic light positions, guardrail styles, road surface markings, and any other infrastructure that varies by country regulation.

        For the Infrastructure Agent: describe all visible vehicles (types, brands, frequency), road surface quality, lane markings and widths, and building architecture (styles, materials, roof shapes, distinctive features).

        For the Climate Agent: describe all visible climate and environmental indicators - vegetation density and type, weather conditions (snow, rain, sunshine), seasonal cues (bare trees, dry grass, lush green), soil moisture, and light/sun angle.

        Return all six prompts."""

    return await _llm.with_structured_output(OrchestratorOutput).ainvoke([HumanMessage(content=prompt)])
