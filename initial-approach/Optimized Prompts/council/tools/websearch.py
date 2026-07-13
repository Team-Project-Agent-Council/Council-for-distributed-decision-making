from __future__ import annotations

import re

import httpx
from langchain_core.tools import tool

_SEARCH_URL = "https://html.duckduckgo.com/html/"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GeoGuessrCouncil/1.0)"}


@tool
async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo.

    Returns titles, snippets, and URLs. If a result looks highly relevant,
    follow up with fetch_page(url) to read its full content.

    Keep queries specific and factual for best results.
    """
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        try:
            r = await client.post(
                _SEARCH_URL,
                data={"q": query, "b": ""},
                headers=_HEADERS,
            )
            r.raise_for_status()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            return f"Search failed for '{query}': {exc}"

    html = r.text
    results: list[dict[str, str]] = []
    for m in re.finditer(
        r'<a rel="nofollow" class="result__a" href="(?P<url>[^"]+)"[^>]*>'
        r"(?P<title>.*?)</a>.*?"
        r'<a class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        html,
        re.DOTALL,
    ):
        title = re.sub(r"<[^>]+>", "", m.group("title")).strip()
        snippet = re.sub(r"<[^>]+>", "", m.group("snippet")).strip()
        url = m.group("url")
        if title and url:
            results.append({"title": title, "snippet": snippet, "url": url})
        if len(results) >= 5:
            break

    if not results:
        return f"No results found for '{query}'."
    lines = []
    for r in results:
        lines.append(f"**{r['title']}**\n{r['snippet']}\nURL: {r['url']}")
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
    async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
        r = await client.get(url, headers=_HEADERS)
        r.raise_for_status()
        html = r.text

    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return f"Could not extract text from {url}."
    return text[:4000] + ("\n\n[truncated]" if len(text) > 4000 else "")
