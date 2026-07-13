"""Embedding via Ollama or vLLM, plus ChromaDB index builder.

Two backends:
  - ollama (default locally): qwen3-embedding:8b via Ollama's embed API
  - vllm (cluster): qwen3-embedding:8b via vLLM's /v1/embeddings endpoint

Set via EMBEDDING_PROVIDER env var ("ollama" or "vllm").
"""
from __future__ import annotations

import logging
import os

from geoguessr_rag.chunker import Chunk
from geoguessr_rag.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    OLLAMA_HOST,
)

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "ollama").lower()
VLLM_EMBEDDING_URL = os.environ.get("VLLM_EMBEDDING_URL", "http://127.0.0.1:8235")


def _embed_ollama(texts: list[str], batch_size: int) -> list[list[float]]:
    import ollama
    client = ollama.Client(host=OLLAMA_HOST)
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embed(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(resp.embeddings)
        if len(texts) > batch_size:
            logger.info("  Embedded batch %d-%d / %d", i + 1, min(i + batch_size, len(texts)), len(texts))
    return all_embeddings


def _embed_vllm(texts: list[str], batch_size: int, is_query: bool = False) -> list[list[float]]:
    import httpx
    url = f"{VLLM_EMBEDDING_URL.rstrip('/')}/v1/embeddings"
    model = os.environ.get("VLLM_EMBEDDING_MODEL", EMBEDDING_MODEL)

    # Qwen3-Embedding: queries need "Instruct: ...\nQuery:" prefix, documents are bare text
    if is_query:
        task = "Given visual features from a street-view image, retrieve geographic clues that match"
        texts = [f"Instruct: {task}\nQuery:{t}" for t in texts]

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = httpx.post(url, json={"model": model, "input": batch}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        batch_embs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        all_embeddings.extend(batch_embs)
        if len(texts) > batch_size:
            logger.info("  Embedded batch %d-%d / %d", i + 1, min(i + batch_size, len(texts)), len(texts))
    return all_embeddings


def embed_texts(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """Embed document texts using the configured provider (ollama or vllm)."""
    if EMBEDDING_PROVIDER == "vllm":
        return _embed_vllm(texts, batch_size, is_query=False)
    return _embed_ollama(texts, batch_size)


def embed_query(text: str) -> list[float]:
    """Embed a single query text."""
    if EMBEDDING_PROVIDER == "vllm":
        return _embed_vllm([text], batch_size=1, is_query=True)[0]
    import ollama
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.embed(model=EMBEDDING_MODEL, input=[text])
    return resp.embeddings[0]


# -- Legacy -------------------------------------------------------------------

def get_ollama_client():
    import ollama
    return ollama.Client(host=OLLAMA_HOST)


# -- Index builder ------------------------------------------------------------

def build_index(chunks: list[Chunk], force: bool = False) -> None:
    import chromadb
    logger.info("Opening ChromaDB at %s", CHROMA_DB_PATH)
    db = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    if force:
        try:
            db.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
        except Exception:
            pass

    collection = db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0 and not force:
        logger.info("Collection already has %d entries. Use --force to rebuild.", existing)
        return

    provider_info = f"vLLM @ {VLLM_EMBEDDING_URL}" if EMBEDDING_PROVIDER == "vllm" else f"Ollama @ {OLLAMA_HOST}"
    logger.info("Embedding %d chunks via %s (%s)...", len(chunks), EMBEDDING_MODEL, provider_info)
    embeddings = embed_texts([c.embedding_text for c in chunks])

    logger.info("Inserting into ChromaDB...")
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch_chunks = chunks[i : i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embeddings[i : i + EMBEDDING_BATCH_SIZE]

        collection.add(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c.text for c in batch_chunks],
            metadatas=[
                {
                    "country_slug": c.country_slug,
                    "country_title": c.country_title,
                    "country_code": c.country_code,
                    "continent": c.continent,
                    "step_title": c.step_title,
                    "category": c.category,
                    "item_kind": c.item_kind,
                }
                for c in batch_chunks
            ],
        )
        logger.info(
            "  Stored batch %d-%d / %d",
            i + 1,
            min(i + EMBEDDING_BATCH_SIZE, len(chunks)),
            len(chunks),
        )

    logger.info("Done! Collection '%s' now has %d entries.", COLLECTION_NAME, collection.count())
