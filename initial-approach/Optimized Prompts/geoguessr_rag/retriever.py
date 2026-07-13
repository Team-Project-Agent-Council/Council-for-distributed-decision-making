from __future__ import annotations

import logging
from dataclasses import dataclass, field

import chromadb

from geoguessr_rag.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    country_slug: str
    country_title: str
    country_code: str
    continent: str
    step_title: str
    category: str
    distance: float
    similarity: float


@dataclass
class CountryScore:
    country_slug: str
    country_title: str
    country_code: str
    continent: str
    match_count: int = 0
    total_score: float = 0.0
    avg_similarity: float = 0.0
    top_clues: list[RetrievalResult] = field(default_factory=list)


def _get_collection() -> chromadb.Collection:
    db = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return db.get_collection(name=COLLECTION_NAME)


def _embed_query(text: str) -> list[float]:
    from geoguessr_rag.embedder import embed_query
    return embed_query(text)


def query(
    description: str,
    n_results: int = 10,
    where: dict | None = None,
    where_document: dict | None = None,
) -> list[RetrievalResult]:
    collection = _get_collection()
    query_embedding = _embed_query(description)

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    if where_document:
        kwargs["where_document"] = where_document

    results = collection.query(**kwargs)

    out: list[RetrievalResult] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        out.append(
            RetrievalResult(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                country_slug=meta["country_slug"],
                country_title=meta["country_title"],
                country_code=meta["country_code"],
                continent=meta["continent"],
                step_title=meta["step_title"],
                category=meta["category"],
                distance=dist,
                similarity=1.0 - dist,
            )
        )

    return out


def query_with_country_aggregation(
    description: str,
    n_results: int = 50,
    top_countries: int = 5,
    where: dict | None = None,
) -> list[CountryScore]:
    results = query(description, n_results=n_results, where=where)

    by_country: dict[str, CountryScore] = {}

    for r in results:
        slug = r.country_slug
        if slug not in by_country:
            by_country[slug] = CountryScore(
                country_slug=slug,
                country_title=r.country_title,
                country_code=r.country_code,
                continent=r.continent,
            )
        cs = by_country[slug]
        cs.match_count += 1
        cs.total_score += r.similarity
        if len(cs.top_clues) < 3:
            cs.top_clues.append(r)

    for cs in by_country.values():
        if cs.match_count > 0:
            cs.avg_similarity = cs.total_score / cs.match_count

    ranked = sorted(by_country.values(), key=lambda c: c.total_score, reverse=True)
    return ranked[:top_countries]
