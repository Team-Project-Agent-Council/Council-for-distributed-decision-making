from __future__ import annotations

import httpx
from langchain_core.tools import tool


@tool
async def geocode(place: str) -> str:
    """Get GPS coordinates (lat/lon) for a place name via OpenStreetMap Nominatim.

    Use this to produce a precise coordinate estimate for GeoGuessr.
    Always pass the most specific location you can determine - a region, city,
    or landmark within the country, NOT just the country name alone.
    Combining country + region or city yields far more accurate coordinates.

    Good examples (specific):
      "Chiang Mai, Thailand"         -> lat=18.8, lon=98.9
      "Bavaria, Germany"             -> lat=48.9, lon=11.4
      "Zakopane, Poland"             -> lat=49.3, lon=19.9
      "Cappadocia, Turkey"           -> lat=38.6, lon=34.8
      "KwaZulu-Natal, South Africa"  -> lat=-28.5, lon=30.9
      "Hollywood Boulevard, USA"     -> lat=34.1, lon=-118.3

    Bad example (too vague - avoid):
      "Thailand"  ->  centre of the country, useless for GeoGuessr
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
        return f"No coordinates found for '{place}'."
    lines = [
        f"{res['display_name']}: lat={res['lat']}, lon={res['lon']}"
        for res in results
    ]
    return "\n".join(lines)
