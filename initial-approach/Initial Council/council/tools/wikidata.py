from __future__ import annotations

import httpx
from langchain_core.tools import tool


@tool
async def wikidata_search(term: str, kind: str = "item") -> str:
    """Resolve a natural language term to Wikidata Q/P IDs.

    Call this BEFORE wikidata_sparql to get the correct numeric IDs for
    entities and properties - do not guess IDs.

    Args:
        term: The label to search for, e.g. "Portuguese", "Cyrillic", "drives on left".
        kind: "item" for entities (languages, scripts, countries, currencies).
              "property" for predicates (official language, writing system, currency).

    Returns a ranked list of matches with their ID, label, and description.
    Use the ID from the first relevant result in your SPARQL query.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": term,
                "language": "en",
                "type": "property" if kind == "property" else "item",
                "format": "json",
                "limit": 5,
            },
            headers={"User-Agent": "GeoGuessrCouncil/1.0 (research project)"},
        )
        r.raise_for_status()
    results = r.json().get("search", [])
    if not results:
        return f"No Wikidata entity found for '{term}'."
    lines = [
        f"{hit['id']}: {hit['label']} - {hit.get('description', 'no description')}"
        for hit in results
    ]
    return "\n".join(lines)


@tool
async def wikidata_sparql(query: str) -> str:
    """Execute a SPARQL SELECT query against the Wikidata public endpoint.

    Use wikidata_search first to resolve entity and property names to their
    Wikidata IDs, then build the SPARQL query with those IDs.

    Every query should include the label service so results are human-readable:
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }

    Sovereign states have wdt:P31 wd:Q6256 - add this to filter to countries only.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"},
            headers={"User-Agent": "GeoGuessrCouncil/1.0 (research project)"},
        )
        r.raise_for_status()
    bindings = r.json().get("results", {}).get("bindings", [])
    if not bindings:
        return "No results found for this query."
    rows = [" | ".join(v["value"] for v in b.values()) for b in bindings[:25]]
    return "\n".join(rows)
