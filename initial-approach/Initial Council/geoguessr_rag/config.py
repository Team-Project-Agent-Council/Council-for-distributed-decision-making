from pathlib import Path
import os

_env_root = os.environ.get("GEOGUESSR_RAG_ROOT")
PROJECT_ROOT = Path(_env_root) if _env_root else Path(__file__).resolve().parent.parent

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11436")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:8b")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "4096"))

# ChromaDB
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "geoguessr_clues"

# Data
PLONKIT_META_PATH = PROJECT_ROOT / "plonkit_meta.json"

# Embedding batch size
EMBEDDING_BATCH_SIZE = 50

# Non-country entries to skip
EXCLUDED_SLUGS = {"beginners-guide", "spillover-countries", "middle-earth"}
