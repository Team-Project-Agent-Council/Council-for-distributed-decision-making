from __future__ import annotations

import httpx
from langchain_core.tools import tool


@tool
async def plant_search(description: str) -> str:
    """Search for plant or animal species by a textual description using iNaturalist.

    Use this when the exact species name is unknown but a visual description
    is available - e.g. from the vision agent output. Returns candidate species
    with their scientific names, which can then be passed to gbif_distribution
    or powo_distribution for geographic distribution lookup.

    Good inputs (descriptive, specific):
      "tall palm with feathery fronds and clusters of orange fruit"
      "large tree with umbrella canopy, thorns, and feathery bipinnate leaves"
      "giant cactus with ribbed columns and white flowers"
      "tree with white papery bark and silver-green narrow leaves"
      "large pink flower on succulent with thick leaves"

    The more morphological detail, the better the match.
    After getting candidate names, always follow up with gbif_distribution
    or powo_distribution to get the geographic range.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            "https://api.inaturalist.org/v1/taxa",
            params={
                "q": description,
                "rank": "species,genus",
                "is_active": "true",
                "per_page": 8,
            },
            headers={"User-Agent": "GeoGuessrCouncil/1.0 (research project)"},
        )
        r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return f"No species found matching '{description}'."
    lines = []
    for t in results:
        scientific = t.get("name", "")
        common = t.get("preferred_common_name", "-")
        group = t.get("iconic_taxon_name", "unknown")
        rank = t.get("rank", "")
        lines.append(f"{scientific} ({common}) [{rank}, {group}]")
    return (
        f"Candidate species for '{description}':\n"
        + "\n".join(lines)
        + "\n\nUse gbif_distribution or powo_distribution with the scientific name "
        "to look up the geographic range."
    )


@tool
async def gbif_distribution(species_name: str) -> str:
    """Look up the geographic distribution of a plant or animal species using GBIF
    (Global Biodiversity Information Facility).

    Given a species name (common or scientific), returns the countries where it
    is most commonly recorded. Use this to narrow down the location based on
    distinctive vegetation, trees, flowers, or fauna visible in the image.

    Examples of useful inputs:
      "Baobab tree"           -> Madagascar, Senegal, Tanzania, ...
      "Agave"                 -> Mexico, United States, ...
      "Eucalyptus"            -> Australia, ...
      "Araucaria araucana"    -> Chile, Argentina
      "Phoenix dactylifera"   -> Egypt, Saudi Arabia, ...
    """
    async with httpx.AsyncClient(timeout=15) as client:
        match_r = await client.get(
            "https://api.gbif.org/v1/species/match",
            params={"name": species_name, "verbose": "false"},
        )
        match_r.raise_for_status()
        match = match_r.json()

        if match.get("matchType") == "NONE":
            return f"No species found matching '{species_name}'."

        taxon_key = match.get("usageKey")
        canonical = match.get("canonicalName", species_name)

        occ_r = await client.get(
            "https://api.gbif.org/v1/occurrence/search",
            params={
                "taxonKey": taxon_key,
                "limit": 0,
                "facet": "country",
                "facetLimit": 15,
            },
        )
        occ_r.raise_for_status()
        data = occ_r.json()

    facets = data.get("facets", [])
    if not facets or not facets[0].get("counts"):
        return f"No country distribution data found for {canonical}."

    lines = [
        f"{c['name']}: {c['count']:,} occurrences"
        for c in facets[0]["counts"][:12]
    ]
    return f"{canonical} - top countries by occurrence:\n" + "\n".join(lines)


@tool
async def powo_distribution(plant_name: str) -> str:
    """Look up the native distribution of a plant species using POWO
    (Plants of the World Online, Kew Gardens).

    Returns the native geographic ranges (botanical regions and countries) for a
    plant species. More precise than GBIF for native vs. cultivated ranges.
    Only covers vascular plants.

    Examples:
      "Welwitschia mirabilis"  -> Angola, Namibia (endemic)
      "Coffea arabica"         -> Ethiopia, Yemen (native range)
      "Quercus robur"          -> Europe, Western Asia
    """
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://powo.science.kew.org/api/2/search",
            params={"q": plant_name, "f": "species_profile", "limit": 3},
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return f"No POWO entry found for '{plant_name}'."

    lines = []
    for res in results[:2]:
        name = res.get("name", plant_name)
        natives = res.get("distribution", {}).get("natives", [])
        if natives:
            regions = ", ".join(n.get("name", "") for n in natives[:15])
            lines.append(f"{name} - native range: {regions}")
        else:
            lines.append(f"{name} - no native distribution data available")
    return "\n".join(lines)
