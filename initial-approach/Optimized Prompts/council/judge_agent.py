from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from council.llm import get_llm, get_thinking_prefix
from council.tools import wikidata_search, wikidata_sparql, geocode

_llm = get_llm("judge")


@tool
async def identify_country(linguistic: str, landscape: str, botanics: str, regulatory: str, rag_meta: str, infrastructure: str, cultural: str) -> str:
    """Name the single most likely country given findings from all specialist agents."""
    think = get_thinking_prefix("judge", "tool")
    response = await _llm.ainvoke([
        SystemMessage(content=(
            f"{think} You are a geography expert making a final country determination. "
            "Given evidence from seven specialist agents, name the single most likely country. "
            "RULES: (1) Pick a country that at least one agent predicted - do not invent your own. "
            "(2) If multiple agents agree on one country, prefer that country. "
            "(3) If agents disagree, follow the agent with the strongest region-specific evidence "
            "(readable text, license plate format, endemic species). "
            "(4) Do NOT default to United States or France without strong specific evidence. "
            "Respond with only the country name. Example: 'Austria'"
        )),
        HumanMessage(content=(
            f"Linguistic evidence: {linguistic}\n"
            f"Landscape evidence: {landscape}\n"
            f"Botanical evidence: {botanics}\n"
            f"Regulatory evidence: {regulatory}\n"
            f"Infrastructure evidence: {infrastructure}\n"
            f"Cultural evidence: {cultural}\n"
            f"RAG/meta candidates:\n{rag_meta}"
        )),
    ])
    return response.content.strip()


_llm_with_tools = _llm.bind_tools([identify_country, wikidata_search, wikidata_sparql, geocode])


async def run(
    linguistic_result: str,
    landscape_result: str,
    botanics_result: str,
    regulatory_result: str,
    rag_result: str,
    infrastructure_result: str,
    cultural_result: str,
    general_description: str = "",
    crop_descriptions: list[str] | None = None,
) -> AIMessage:
    """Call LLM - returns AIMessage with tool_calls for ToolNode to execute."""
    think = get_thinking_prefix("judge", "run")
    system = SystemMessage(content=(
        f"{think} You are a Judge Agent making the final country determination for a GeoGuessr image.\n\n"
        "You receive findings from specialist agents:\n"
        "- Linguistic Agent: language/script clues and their geographic implications\n"
        "- Landscape Agent: terrain, vegetation, and geographic clues\n"
        "- Botanics Agent: plant species distributions from GBIF/POWO\n"
        "- Regulatory Agent: road design, signs, markings, and infrastructure standards\n"
        "- Infrastructure Agent: vehicles, road surface, building architecture, and street furniture\n"
        "- Cultural Agent: clothing, murals, shops, religious buildings, colonial heritage, and cultural context\n"
        "- Meta Agent: RAG knowledge base candidates\n\n"
        "DECISION APPROACH:\n\n"
        "You receive evidence from multiple specialist agents. Each agent sees the same scene "
        "from a different angle. Your job is to find the country that makes ALL the evidence "
        "fit together - not just follow the most confident agent.\n\n"
        "Before deciding, ask yourself:\n"
        "- Which agents agree on the same REGION, even if they disagree on the exact country?\n"
        "- Are any agents providing evidence that CONTRADICTS another agent's top pick? "
        "If one agent's evidence about vegetation or terrain rules out a region that "
        "another agent suggested, that contradiction matters.\n"
        "- Are two agents agreeing because they found DIFFERENT supporting clues, or are they "
        "both interpreting the same feature (e.g. both reading road markings)? Independent "
        "agreement from different evidence types is a stronger signal.\n"
        "- Could an agent's confident prediction still be wrong? Road markings, for instance, "
        "are shared across many countries - a confident match to one country doesn't mean "
        "other countries don't use the same system.\n"
        "- If one agent's evidence genuinely rules out another agent's region, look for a "
        "third region that satisfies both constraints.\n\n"
        "Judge the evidence itself, not how confidently the agent presents it. "
        "A plant species with a narrow native range or text in a specific script is inherently "
        "more geographically specific than road marking colors or utility pole styles.\n\n"
        "RULE - No geographic bias:\n"
        "Do not favor or avoid any country. Do not default to the US, France, India, or UK "
        "when evidence is ambiguous.\n\n"
        "RULE 5 - Transparency:\n"
        "In your reasoning, explicitly list: (1) the key clues from each agent, "
        "(2) which country best explains ALL of them combined, (3) what specific evidence "
        "convinced you, (4) what alternatives you considered and why you rejected them.\n\n"
        "Use the available tools:\n"
        "1. wikidata_search - resolve any entity or property name to a Wikidata ID.\n"
        "2. wikidata_sparql - optionally verify or resolve conflicts between agents.\n"
        "3. geocode - get GPS coordinates for a REAL city or town name. "
        "Do NOT geocode vague descriptions like 'highlands' or 'mountains'. "
        "Use a real place: 'Bandung, Indonesia', 'Chiang Mai, Thailand', 'Zakopane, Poland'.\n"
        "4. identify_country - produce the final country determination.\n\n"
        "Always call geocode with a specific region or city (not just the country), "
        "then identify_country as your final action."
    ))
    human = HumanMessage(content=(
        f"Linguistic finding: {linguistic_result}\n"
        f"Landscape finding: {landscape_result}\n"
        f"Botanical finding: {botanics_result}\n"
        f"Regulatory finding: {regulatory_result}\n"
        f"Infrastructure finding: {infrastructure_result}\n"
        f"Cultural finding: {cultural_result}\n"
        f"RAG/meta candidates:\n{rag_result}"
    ))
    return await _llm_with_tools.ainvoke([system, human])


class JudgeOutput(BaseModel):
    country: str = Field(description="The single most likely country name")
    latitude: float = Field(description="GPS latitude of the predicted location")
    longitude: float = Field(description="GPS longitude of the predicted location")
    reasoning: str = Field(description="2-3 sentences citing which agents you followed and what evidence convinced you")


async def reason(messages: list[BaseMessage]) -> str:
    """Given messages including tool results, produce the final country answer."""
    think = get_thinking_prefix("judge", "reason")
    structured_llm = _llm.with_structured_output(JudgeOutput)
    result = await structured_llm.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} Based on all evidence and tool results above, determine the single most likely country, "
            "GPS coordinates, and a brief explanation. "
            "You MUST name which agent(s) you followed and what specific evidence convinced you. "
            "Use geocode results for coordinates if available; otherwise estimate from the regional clues."
        )),
    ])
    return f"Country: {result.country}\nCoordinates: {result.latitude}, {result.longitude}\nReasoning: {result.reasoning}"
