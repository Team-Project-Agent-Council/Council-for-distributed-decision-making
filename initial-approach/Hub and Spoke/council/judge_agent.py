from __future__ import annotations

from typing import Literal

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
            "RULES:\n"
            "(1) Think step by step. Consider ALL evidence before deciding.\n"
            "(2) Readable text on signs in a recognizable language is the strongest possible evidence. "
            "A single clearly readable sign can determine the country.\n"
            "(3) Official road sign standards (shape, color scheme) are strong evidence - "
            "different countries have distinct, standardized sign systems.\n"
            "(4) Landscape, vegetation, and architecture are supporting evidence - "
            "they can corroborate but should never override readable text or sign standards.\n"
            "(5) Do NOT default to any particular country without specific evidence.\n"
            "Respond with only the country name."
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
        "EVIDENCE HIERARCHY (strongest to weakest):\n"
        "Tier 1 - NEAR-CERTAIN (one clue can determine the country):\n"
        "  - Readable text in a recognizable language on official signs\n"
        "  - Country-specific road sign standards (shape + color scheme combinations are standardized per country)\n"
        "  - Clearly visible license plate format\n"
        "  - Named businesses, institutions, or brands unique to one country\n"
        "Tier 2 - STRONG (narrows to a region, needs corroboration):\n"
        "  - Road marking standards (color combinations and patterns vary by country)\n"
        "  - Driving side (left vs right)\n"
        "  - Endemic plant species with narrow native range\n"
        "  - Specific vehicle types dominant in certain regions\n"
        "Tier 3 - WEAK (supports but never determines alone):\n"
        "  - General landscape, terrain, vegetation type\n"
        "  - Building style, architecture\n"
        "  - Weather, climate indicators\n\n"
        "CRITICAL RULE: A single Tier 1 clue OUTWEIGHS any number of Tier 3 clues. "
        "Think carefully about the STRENGTH of each piece of evidence before weighing it.\n\n"
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
        "Do NOT geocode just a country name or its capital as a lazy default.\n"
        "   GEOCODING STRATEGY: Use ALL available clues to pinpoint the SPECIFIC region:\n"
        "   - If a sign mentions a city or town name, geocode THAT exact place\n"
        "   - If a recognizable company HQ or landmark is visible, geocode that location\n"
        "   - If road signs show directions to specific cities, geocode a point between them\n"
        "   - If regional clues narrow it down (climate, terrain, dialect), geocode a representative city in that region\n"
        "   - NEVER just geocode the capital unless you have specific evidence pointing there\n"
        "4. identify_country - produce the final country determination.\n\n"
        "Always call geocode with the most specific location you can determine from the evidence, "
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
    latitude: float = Field(description="GPS latitude of the predicted location - must be as precise as possible, not just the country center")
    longitude: float = Field(description="GPS longitude of the predicted location - must be as precise as possible, not just the country center")
    reasoning: str = Field(description="2-3 sentences citing which agents you followed and what evidence convinced you")


async def reason(messages: list[BaseMessage]) -> str:
    """Given messages including tool results, produce the final country answer."""
    think = get_thinking_prefix("judge", "reason")
    structured_llm = _llm.with_structured_output(JudgeOutput)
    result = await structured_llm.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} Based on all evidence and tool results above, determine the single most likely country, "
            "GPS coordinates, and a brief explanation.\n"
            "COORDINATES: Use the geocode results from above. If the geocode was for a specific city or region, "
            "use THOSE coordinates - do NOT replace them with the country center or capital.\n"
            "If no geocode was called, estimate coordinates for the most specific region you can identify "
            "from the evidence (e.g., if signs point to cities in southern Germany, pick coordinates in that area).\n"
            "You MUST name which agent(s) you followed and what specific evidence convinced you."
        )),
    ])
    return f"Country: {result.country}\nCoordinates: {result.latitude}, {result.longitude}\nReasoning: {result.reasoning}"


# -- deliberation (hub-and-spoke) ---------------------------------------------

class FollowUpRequest(BaseModel):
    agent: Literal["linguistic", "landscape", "botanics", "regulatory", "infrastructure", "cultural"] = Field(
        description="Which specialist agent to query"
    )
    question: str = Field(
        description="Confrontational follow-up question referencing other agents' claims"
    )


class DeliberationDecision(BaseModel):
    satisfied: bool = Field(
        description="True if evidence is sufficient for a final determination"
    )
    reasoning: str = Field(
        description="2-3 sentences on the current state of evidence"
    )
    follow_ups: list[FollowUpRequest] = Field(
        default_factory=list,
        description="Follow-up questions for specific agents. Empty if satisfied."
    )


async def deliberate(
    linguistic_result: str,
    landscape_result: str,
    botanics_result: str,
    regulatory_result: str,
    rag_result: str,
    infrastructure_result: str,
    cultural_result: str,
    deliberation_round: int,
    deliberation_history: list[str],
) -> DeliberationDecision:
    """Review all specialist results and decide whether to ask follow-ups or finalize."""
    think = get_thinking_prefix("judge", "deliberate")
    structured_llm = _llm.with_structured_output(DeliberationDecision)

    history_block = ""
    if deliberation_history:
        history_block = "\n\nDELIBERATION HISTORY (previous rounds):\n" + "\n".join(deliberation_history)

    system = SystemMessage(content=(
        f"{think} You are the Judge Agent in a hub-and-spoke deliberation council for GeoGuessr.\n\n"
        "You have received assessments from specialist agents. Your job is to determine whether "
        "the evidence is sufficient for a confident final country determination, or whether "
        "specific agents need to be challenged with follow-up questions.\n\n"
        "WHEN TO BE SATISFIED:\n"
        "- ONLY when ALL specialist agents rank the SAME country as their #1 candidate.\n"
        "- If even ONE agent has a different #1 country, you are NOT satisfied.\n"
        "- Check each agent's first-ranked country carefully before deciding.\n\n"
        "WHEN TO ASK FOLLOW-UPS:\n"
        "- ANY agent has a different #1 country than the majority\n"
        "- An agent's evidence could plausibly support multiple countries and you need disambiguation\n"
        "- Two agents contradict each other and you need to understand why\n\n"
        "HOW TO FORMULATE FOLLOW-UPS:\n"
        "- Be CONFRONTATIONAL: reference SPECIFIC claims from other agents by name.\n"
        "- Point out the EVIDENCE TIER: if one agent has Tier 1 evidence and another contradicts "
        "with only Tier 3, challenge the weaker agent to justify their position against the stronger evidence.\n"
        "- Challenge agents whose evidence is weak when it contradicts strong evidence from others\n"
        "- Ask agents to reconsider if their #1 pick conflicts with readable text or sign standards\n"
        "- Demand a clear position - no hedging, no 'it could be either'\n"
        "- Ask agents to THINK CAREFULLY and provide precise arguments for their position\n"
        "- Do NOT re-ask the same question from a previous round\n\n"
        f"Current deliberation round: {deliberation_round}\n"
        "Maximum 3 deliberation rounds. If this is round 3 or higher, you MUST set satisfied=True.\n"
        "Do not ask more than 3 agents in a single round."
        f"{history_block}"
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

    return await structured_llm.ainvoke([system, human])
