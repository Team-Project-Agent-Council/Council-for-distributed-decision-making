"""Country → region mapping used by the eval pipeline.

Region taxonomy matches the one in agents/judge.py:
  Europe, East Asia, Southeast Asia, South Asia, Central Asia, Middle East,
  North Africa, Sub-Saharan Africa, North America, Central America & Caribbean,
  South America, Oceania.

``country_to_region(name_or_code)`` accepts ISO-3166 alpha-2, alpha-3, or a
country name (case-insensitive).  Returns None on lookup failure.
"""

from __future__ import annotations

_CODE2REGION: dict[str, str] = {
    # Europe
    "AL": "Europe", "AD": "Europe", "AT": "Europe", "BY": "Europe",
    "BE": "Europe", "BA": "Europe", "BG": "Europe", "HR": "Europe",
    "CY": "Europe", "CZ": "Europe", "DK": "Europe", "EE": "Europe",
    "FI": "Europe", "FR": "Europe", "DE": "Europe", "GR": "Europe",
    "HU": "Europe", "IS": "Europe", "IE": "Europe", "IT": "Europe",
    "XK": "Europe", "LV": "Europe", "LI": "Europe", "LT": "Europe",
    "LU": "Europe", "MT": "Europe", "MD": "Europe", "MC": "Europe",
    "ME": "Europe", "NL": "Europe", "MK": "Europe", "NO": "Europe",
    "PL": "Europe", "PT": "Europe", "RO": "Europe", "RU": "Europe",
    "SM": "Europe", "RS": "Europe", "SK": "Europe", "SI": "Europe",
    "ES": "Europe", "SE": "Europe", "CH": "Europe", "UA": "Europe",
    "GB": "Europe", "VA": "Europe",
    # East Asia
    "CN": "East Asia", "JP": "East Asia", "KR": "East Asia",
    "KP": "East Asia", "MN": "East Asia", "TW": "East Asia",
    "HK": "East Asia", "MO": "East Asia",
    # Southeast Asia
    "BN": "Southeast Asia", "KH": "Southeast Asia", "TL": "Southeast Asia",
    "ID": "Southeast Asia", "LA": "Southeast Asia", "MY": "Southeast Asia",
    "MM": "Southeast Asia", "PH": "Southeast Asia", "SG": "Southeast Asia",
    "TH": "Southeast Asia", "VN": "Southeast Asia",
    # South Asia
    "AF": "South Asia", "BD": "South Asia", "BT": "South Asia",
    "IN": "South Asia", "MV": "South Asia", "NP": "South Asia",
    "PK": "South Asia", "LK": "South Asia",
    # Central Asia
    "KZ": "Central Asia", "KG": "Central Asia", "TJ": "Central Asia",
    "TM": "Central Asia", "UZ": "Central Asia",
    # Middle East
    "BH": "Middle East", "IR": "Middle East", "IQ": "Middle East",
    "IL": "Middle East", "JO": "Middle East", "KW": "Middle East",
    "LB": "Middle East", "OM": "Middle East", "PS": "Middle East",
    "QA": "Middle East", "SA": "Middle East", "SY": "Middle East",
    "TR": "Middle East", "AE": "Middle East", "YE": "Middle East",
    "GE": "Middle East", "AM": "Middle East", "AZ": "Middle East",
    # North Africa
    "DZ": "North Africa", "EG": "North Africa", "LY": "North Africa",
    "MA": "North Africa", "MR": "North Africa", "SD": "North Africa",
    "TN": "North Africa", "SS": "North Africa",
    # Sub-Saharan Africa
    "AO": "Sub-Saharan Africa", "BJ": "Sub-Saharan Africa",
    "BW": "Sub-Saharan Africa", "BF": "Sub-Saharan Africa",
    "BI": "Sub-Saharan Africa", "CV": "Sub-Saharan Africa",
    "CM": "Sub-Saharan Africa", "CF": "Sub-Saharan Africa",
    "TD": "Sub-Saharan Africa", "KM": "Sub-Saharan Africa",
    "CG": "Sub-Saharan Africa", "CD": "Sub-Saharan Africa",
    "CI": "Sub-Saharan Africa", "DJ": "Sub-Saharan Africa",
    "GQ": "Sub-Saharan Africa", "ER": "Sub-Saharan Africa",
    "SZ": "Sub-Saharan Africa", "ET": "Sub-Saharan Africa",
    "GA": "Sub-Saharan Africa", "GM": "Sub-Saharan Africa",
    "GH": "Sub-Saharan Africa", "GN": "Sub-Saharan Africa",
    "GW": "Sub-Saharan Africa", "KE": "Sub-Saharan Africa",
    "LS": "Sub-Saharan Africa", "LR": "Sub-Saharan Africa",
    "MG": "Sub-Saharan Africa", "MW": "Sub-Saharan Africa",
    "ML": "Sub-Saharan Africa", "MU": "Sub-Saharan Africa",
    "MZ": "Sub-Saharan Africa", "NA": "Sub-Saharan Africa",
    "NE": "Sub-Saharan Africa", "NG": "Sub-Saharan Africa",
    "RW": "Sub-Saharan Africa", "ST": "Sub-Saharan Africa",
    "SN": "Sub-Saharan Africa", "SC": "Sub-Saharan Africa",
    "SL": "Sub-Saharan Africa", "SO": "Sub-Saharan Africa",
    "ZA": "Sub-Saharan Africa", "TZ": "Sub-Saharan Africa",
    "TG": "Sub-Saharan Africa", "UG": "Sub-Saharan Africa",
    "ZM": "Sub-Saharan Africa", "ZW": "Sub-Saharan Africa",
    # North America
    "CA": "North America", "MX": "North America", "US": "North America",
    "GL": "North America", "PM": "North America",
    # Central America & Caribbean
    "AI": "Central America & Caribbean", "AG": "Central America & Caribbean",
    "AW": "Central America & Caribbean", "BS": "Central America & Caribbean",
    "BB": "Central America & Caribbean", "BZ": "Central America & Caribbean",
    "BM": "Central America & Caribbean", "VG": "Central America & Caribbean",
    "KY": "Central America & Caribbean", "CR": "Central America & Caribbean",
    "CU": "Central America & Caribbean", "CW": "Central America & Caribbean",
    "DM": "Central America & Caribbean", "DO": "Central America & Caribbean",
    "SV": "Central America & Caribbean", "GD": "Central America & Caribbean",
    "GP": "Central America & Caribbean", "GT": "Central America & Caribbean",
    "HT": "Central America & Caribbean", "HN": "Central America & Caribbean",
    "JM": "Central America & Caribbean", "MQ": "Central America & Caribbean",
    "MS": "Central America & Caribbean", "AN": "Central America & Caribbean",
    "NI": "Central America & Caribbean", "PA": "Central America & Caribbean",
    "PR": "Central America & Caribbean", "BL": "Central America & Caribbean",
    "KN": "Central America & Caribbean", "LC": "Central America & Caribbean",
    "MF": "Central America & Caribbean", "VC": "Central America & Caribbean",
    "SX": "Central America & Caribbean", "TT": "Central America & Caribbean",
    "TC": "Central America & Caribbean", "VI": "Central America & Caribbean",
    # South America
    "AR": "South America", "BO": "South America", "BR": "South America",
    "CL": "South America", "CO": "South America", "EC": "South America",
    "FK": "South America", "GF": "South America", "GY": "South America",
    "PY": "South America", "PE": "South America", "SR": "South America",
    "UY": "South America", "VE": "South America",
    # Oceania
    "AU": "Oceania", "CK": "Oceania", "FJ": "Oceania", "PF": "Oceania",
    "GU": "Oceania", "KI": "Oceania", "MH": "Oceania", "FM": "Oceania",
    "NR": "Oceania", "NC": "Oceania", "NZ": "Oceania", "NU": "Oceania",
    "MP": "Oceania", "PW": "Oceania", "PG": "Oceania", "PN": "Oceania",
    "WS": "Oceania", "SB": "Oceania", "TK": "Oceania", "TO": "Oceania",
    "TV": "Oceania", "VU": "Oceania", "WF": "Oceania",
}

# Normalise alpha-3 → alpha-2 via pycountry (best-effort)
_A3_TO_A2: dict[str, str] = {}
try:
    import pycountry
    for _c in pycountry.countries:
        if hasattr(_c, "alpha_3") and hasattr(_c, "alpha_2"):
            _A3_TO_A2[_c.alpha_3.upper()] = _c.alpha_2.upper()
except ImportError:
    pass


def country_to_region(name_or_code: str | None) -> str | None:
    """Return the world region for a country name or ISO code, or None."""
    if not name_or_code:
        return None
    val = name_or_code.strip()

    # Try direct alpha-2 lookup (2-char)
    upper = val.upper()
    if upper in _CODE2REGION:
        return _CODE2REGION[upper]

    # Try alpha-3 → alpha-2 conversion
    if len(upper) == 3 and upper in _A3_TO_A2:
        a2 = _A3_TO_A2[upper]
        if a2 in _CODE2REGION:
            return _CODE2REGION[a2]

    # Try name lookup via pycountry
    try:
        import pycountry
        hit = (
            pycountry.countries.get(name=val)
            or pycountry.countries.get(common_name=val)
            or pycountry.countries.get(official_name=val)
            or pycountry.countries.lookup(val)
        )
        if hit:
            a2 = hit.alpha_2.upper()
            if a2 in _CODE2REGION:
                return _CODE2REGION[a2]
    except Exception:
        pass

    # Fuzzy: lowercase name contains a country name fragment
    lower = val.lower()
    try:
        import pycountry
        for c in pycountry.countries:
            if c.name.lower() == lower:
                a2 = c.alpha_2.upper()
                return _CODE2REGION.get(a2)
    except Exception:
        pass

    return None
