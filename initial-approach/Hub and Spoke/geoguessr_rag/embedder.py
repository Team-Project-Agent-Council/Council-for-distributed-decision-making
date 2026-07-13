from __future__ import annotations

import logging

import chromadb
import ollama

from geoguessr_rag.chunker import Chunk
from geoguessr_rag.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    OLLAMA_HOST,
)

logger = logging.getLogger(__name__)


def get_ollama_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def embed_texts(
    client: ollama.Client,
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embed(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(resp.embeddings)
        logger.info(
            "  Embedded batch %d-%d / %d",
            i + 1,
            min(i + batch_size, len(texts)),
            len(texts),
        )
    return all_embeddings


def build_index(chunks: list[Chunk], force: bool = False) -> None:
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
        logger.info(
            "Collection already has %d entries. Use --force to rebuild.", existing
        )
        return

    logger.info("Embedding %d chunks via Ollama (%s)...", len(chunks), EMBEDDING_MODEL)
    oll = get_ollama_client()
    embeddings = embed_texts(oll, [c.embedding_text for c in chunks])

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
