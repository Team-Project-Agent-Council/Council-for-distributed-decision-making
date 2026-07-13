from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from geoguessr_rag.config import EXCLUDED_SLUGS, PLONKIT_META_PATH

# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    embedding_text: str  # text with country header prepended (used for embedding)
    country_slug: str
    country_title: str
    country_code: str
    continent: str
    step_title: str
    category: str
    item_kind: str
    image_url: str = ""


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("vehicle", [
        "google car", "car meta", "generation 2", "generation 3", "generation 4",
        "gen 2", "gen 3", "gen 4", "snorkel", "roof rack", "camera gen",
        "trekker", "smallcam", "shitcam", "coverage was taken", "coverage car",
    ]),
    ("plate", [
        "licence plate", "license plate", "front plate", "rear plate",
        "number plate", " plates",
    ]),
    ("language", [
        "language", "alphabet", "script", "writing system", "cyrillic",
        "afrikaans", "swahili",
    ]),
    ("signage", [
        "signpost", "road sign", "speed sign", "warning sign", "direction sign",
        "exit sign", "street sign", "sign ",
    ]),
    ("road", [
        "road line", "centre line", "center line", "outer line", "road marking",
        "bollard", "guardrail", "chevron", "speed limit", "highway",
        "paved", "unpaved", "asphalt", "road surface", "curb", "rumble strip",
    ]),
    ("landscape", [
        "landscape", "terrain", "soil", "vegetation", "mountain",
        "hill", "desert", "arid", "tropical", "forest", "bush",
        "shrub", "grass", "savanna", "coastline", "elevation", "river",
        "eucalyptus", "palm tree",
    ]),
    ("infrastructure", [
        "building", "architecture", "bridge", "power line", "electricity",
        "telephone pole", "utility pole", "fence", "church", "mosque", "tower",
    ]),
]


def classify_category(text: str) -> str:
    lower = text.lower()
    for category, keywords in CATEGORY_RULES:
        for kw in keywords:
            if kw in lower:
                return category
    return "general"


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MULTI_SPACE = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _extract_text(item: dict) -> str | None:
    """Extract the text content from an item regardless of kind."""
    kind = item.get("kind")

    if kind == "tip":
        texts = item.get("data", {}).get("text", [])
        if texts:
            return "\n".join(texts)

    elif kind == "text":
        texts = item.get("text", [])
        if isinstance(texts, list) and texts:
            return "\n".join(texts)
        elif isinstance(texts, str) and texts:
            return texts

    elif kind == "centeredText":
        t = item.get("text", "")
        if t:
            return t if isinstance(t, str) else "\n".join(t)

    elif kind == "notes":
        texts = item.get("data", {}).get("text", [])
        if texts:
            return "\n".join(texts)

    return None


def load_and_chunk(meta_path: Path | None = None) -> list[Chunk]:
    path = meta_path or PLONKIT_META_PATH

    with open(path, encoding="utf-8") as f:
        data: dict = json.load(f)

    chunks: list[Chunk] = []

    for slug, entry in data.items():
        if slug in EXCLUDED_SLUGS:
            continue

        pub = entry.get("public", {})
        title = pub.get("title", slug)
        code = pub.get("code", "")
        continent = "|".join(pub.get("cat", []))

        for step_idx, step in enumerate(pub.get("steps", [])):
            step_title = step.get("title", f"step_{step_idx}")

            for item_idx, item in enumerate(step.get("items", [])):
                raw = _extract_text(item)
                if not raw:
                    continue

                text = clean_text(raw)
                if not text:
                    continue

                header = f"Country: {title} ({continent}). {step_title}."
                embedding_text = f"{header}\n{text}"

                image_url = ""
                if item.get("kind") == "tip":
                    image_url = item.get("data", {}).get("image", {}).get("imageUrl", "")

                chunk = Chunk(
                    chunk_id=f"{slug}__{step_idx}__{item_idx}",
                    text=text,
                    embedding_text=embedding_text,
                    country_slug=slug,
                    country_title=title,
                    country_code=code,
                    continent=continent,
                    step_title=step_title,
                    category=classify_category(text),
                    item_kind=item.get("kind", "unknown"),
                    image_url=image_url,
                )
                chunks.append(chunk)

    return chunks
