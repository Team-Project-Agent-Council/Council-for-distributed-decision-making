from fastapi import FastAPI
from pydantic import BaseModel

from geoguessr_rag.retriever import query, query_with_country_aggregation

app = FastAPI(title="GeoGuessr RAG API", version="0.1.0")


class ClueResult(BaseModel):
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


class CountryResult(BaseModel):
    country_slug: str
    country_title: str
    country_code: str
    continent: str
    match_count: int
    total_score: float
    avg_similarity: float
    top_clues: list[ClueResult]


class SearchRequest(BaseModel):
    description: str
    n_results: int = 10
    category: str | None = None
    continent: str | None = None


class CountrySearchRequest(BaseModel):
    description: str
    n_results: int = 50
    top_countries: int = 5
    category: str | None = None
    continent: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=list[ClueResult])
def search_clues(req: SearchRequest):
    filters = {}
    if req.category:
        filters["category"] = req.category
    if req.continent:
        filters["continent"] = req.continent
    where = filters if len(filters) == 1 else {"$and": [{k: v} for k, v in filters.items()]} if filters else None
    return [
        ClueResult(**{k: getattr(r, k) for k in ClueResult.model_fields})
        for r in query(req.description, n_results=req.n_results, where=where)
    ]


@app.post("/search/countries", response_model=list[CountryResult])
def search_countries(req: CountrySearchRequest):
    filters = {}
    if req.category:
        filters["category"] = req.category
    if req.continent:
        filters["continent"] = req.continent
    where = filters if len(filters) == 1 else {"$and": [{k: v} for k, v in filters.items()]} if filters else None
    countries = query_with_country_aggregation(
        req.description, n_results=req.n_results, top_countries=req.top_countries, where=where
    )
    return [
        CountryResult(
            **{k: getattr(cs, k) for k in ("country_slug", "country_title", "country_code", "continent", "match_count", "total_score", "avg_similarity")},
            top_clues=[ClueResult(**{k: getattr(r, k) for k in ClueResult.model_fields}) for r in cs.top_clues],
        )
        for cs in countries
    ]
