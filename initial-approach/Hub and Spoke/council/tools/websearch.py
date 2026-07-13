from __future__ import annotations

import asyncio
import re

import httpx
from ddgs import DDGS
from langchain_core.tools import tool


@tool
async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo.

    Returns titles, snippets, and URLs. If a result looks highly relevant,
    follow up with fetch_page(url) to read its full content.

    Useful query types:
      "countries that drive on the left side of the road"
      "yellow center line road markings which countries"
      "EU road sign red circle speed limit design"
      "wooden utility poles countries North America"
      "blue EU license plate format countries"
      "overhead traffic lights vs side mounted countries"
      "road markings by country overview"

    Keep queries specific and factual for best results.
    """
    results = await asyncio.to_thread(
        lambda: list(DDGS().text(query, max_results=5))
    )
    if not results:
        return f"No results found for '{query}'."
    lines = []
    for r in results:
        lines.append(f"**{r['title']}**\n{r['body']}\nURL: {r['href']}")
    return "\n\n".join(lines)


@tool
async def fetch_page(url: str) -> str:
    """Fetch the full text content of a web page by URL.

    Use this after web_search when a result snippet looks highly relevant
    and you want to read the full article for detailed country-specific
    information (e.g. a road markings reference page, a GeoGuessr guide,
    a Wikipedia article on traffic signs by country).

    Returns the first 4000 characters of cleaned page text.
    Avoid fetching pages behind login walls or paywalls.
    """
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        r = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; GeoGuessrCouncil/1.0)"},
        )
        r.raise_for_status()
        html = r.text

    # strip scripts, styles, and tags; collapse whitespace
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return f"Could not extract text from {url}."
    return text[:4000] + ("\n\n[truncated]" if len(text) > 4000 else "")
