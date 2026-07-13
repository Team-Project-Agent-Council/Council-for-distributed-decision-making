from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from council.llm import get_llm, get_thinking_prefix
from council.tools import wikidata_search, wikidata_sparql
from geoguessr_rag.retriever import query_with_country_aggregation

_llm = get_llm("meta")


@tool
async def rag_search(description: str) -> str:
    """Query the GeoGuessr vector DB for country candidates based on a description."""
    results = await asyncio.to_thread(query_with_country_aggregation, description, n_results=50, top_countries=5)
    if not results:
        return "No results found."
    lines = []
    for cs in results:
        clues = " | ".join(r.text[:100] for r in cs.top_clues)
        lines.append(f"{cs.country_title} (score: {cs.total_score:.2f}): {clues}")
    return "\n".join(lines)


_llm_with_tools = _llm.bind_tools([rag_search, wikidata_search, wikidata_sparql])


async def run(general_description: str, crop_descriptions: list[str]) -> AIMessage:
    """Call LLM - returns AIMessage with tool_calls for ToolNode to execute."""
    crops_block = "\n".join(f"- {c}" for c in crop_descriptions)
    think = get_thinking_prefix("meta", "run")
    system = SystemMessage(content=(
        f"{think} You are a RAG Meta Agent. Use the available tools to identify the most likely country:\n"
        "1. rag_search - query the GeoGuessr knowledge base for country candidates matching the image description.\n"
        "2. wikidata_search - resolve any entity or property name to a Wikidata ID before writing SPARQL.\n"
        "3. wikidata_sparql - query Wikidata to verify or cross-check candidates using structured facts "
        "(language, script, driving side, currency, calling code, etc.).\n\n"
        "Workflow for Wikidata: wikidata_search (resolve IDs) -> wikidata_sparql (filter countries).\n"
        "Use all tools to gather evidence, then synthesize the findings."
    ))
    human = HumanMessage(content=f"General description: {general_description}\nCrop descriptions:\n{crops_block}")
    return await _llm_with_tools.ainvoke([system, human])


async def reason(messages: list[BaseMessage]) -> str:
    """Given messages including RAG tool results, produce the most likely country + reasoning."""
    think = get_thinking_prefix("meta", "reason")
    response = await _llm.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} Based on the search results above, provide a ranked list of candidate countries "
            "with reasoning for each. "
            "Format:\n"
            "1. <Country> (confidence: <high|medium|low>) - <reason this candidate scored highly>\n"
            "2. <Country> (confidence: <high|medium|low>) - <reason>\n"
            "List 2-5 candidates, most likely first."
        )),
    ])
    return response.content
