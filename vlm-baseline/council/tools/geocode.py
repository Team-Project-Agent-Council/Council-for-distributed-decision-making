from __future__ import annotations

import httpx
from langchain_core.tools import tool


@tool
async def geocode(place: str) -> str:
    """Get GPS coordinates (lat/lon) for a place name via OpenStreetMap Nominatim.

    Use this to produce a precise coordinate estimate for GeoGuessr.
    Always pass a REAL place name — a city, town, district, or province
    within the country. NOT a vague description like "highlands" or "mountains".

    Good examples:
      "Bandung, Indonesia"           → real city
      "Chiang Mai, Thailand"         → real city
      "Zakopane, Poland"             → real town
      "KwaZulu-Natal, South Africa"  → real province

    Bad examples (will fail):
      "Javanese highlands, Indonesia"  → not a place name
      "tropical mountain trail"        → description, not a place
    """
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place, "format": "json", "limit": 3},
            headers={"User-Agent": "GeoGuessrCouncil/1.0 (research project)"},
        )
        r.raise_for_status()
    results = r.json()
    if not results:
        return f"No coordinates found for '{place}'. Try a real city or town name instead."
    lines = [
        f"{res['display_name']}: lat={res['lat']}, lon={res['lon']}"
        for res in results
    ]
    return "\n".join(lines)
