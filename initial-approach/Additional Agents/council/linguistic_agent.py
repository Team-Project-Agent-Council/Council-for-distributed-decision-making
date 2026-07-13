from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from council.llm import get_llm, get_thinking_prefix
from council.tools import wikidata_search, wikidata_sparql

_llm = get_llm("linguistic")


@tool
async def detect_language(text: str) -> str:
    """Detect the language or script of the given text. Returns the full language/script name."""
    think = get_thinking_prefix("linguistic", "tool")
    response = await _llm.ainvoke([
        SystemMessage(content=(
            f"{think} You are a language detection expert. Given a text snippet, identify the language "
            "and writing system. Respond with only the language name, nothing else. Example: 'German'"
        )),
        HumanMessage(content=text),
    ])
    return response.content.strip()


_llm_with_tools = _llm.bind_tools([detect_language, wikidata_search, wikidata_sparql])


async def run(prompt: str) -> AIMessage:
    """Call LLM - returns AIMessage with tool_calls for ToolNode to execute."""
    think = get_thinking_prefix("linguistic", "run")
    system = SystemMessage(content=(
        f"{think} You are a Linguistic Agent specializing in identifying countries from language clues. "
        "You receive text snippets extracted from signs, posters, and labels visible in a street-level image.\n\n"
        "Use the available tools to:\n"
        "1. detect_language - identify the language or script of a text snippet.\n"
        "2. wikidata_search - resolve a language, script, or property name to a Wikidata ID. "
        "Use kind='item' for languages/scripts, kind='property' for predicates.\n"
        "3. wikidata_sparql - query Wikidata for countries using the IDs from wikidata_search.\n\n"
        "Workflow: detect_language -> wikidata_search (resolve IDs) -> wikidata_sparql (filter countries).\n"
        "Gather evidence, then state the most likely country/region based on the linguistic clues."
    ))
    return await _llm_with_tools.ainvoke([system, HumanMessage(content=prompt)])


async def reason(messages: list[BaseMessage]) -> str:
    """Given messages including tool results, produce a linguistic country assessment."""
    think = get_thinking_prefix("linguistic", "reason")
    response = await _llm.ainvoke([
        *messages,
        HumanMessage(content=(
            f"{think} Based on all tool results above, provide a ranked list of candidate countries "
            "based solely on linguistic evidence. "
            "Format:\n"
            "1. <Country> (confidence: <high|medium|low>) - <reason this language/script points here>\n"
            "2. <Country> (confidence: <high|medium|low>) - <reason>\n"
            "List 2-5 candidates, most likely first."
        )),
    ])
    return response.content
