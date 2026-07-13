"""Configuration loaded from environment variables, v12."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VLMCouncilConfig:
    """Immutable config for the VLM Council v12."""

    vlm_model: str
    api_base: str
    max_model_len: int
    gpu_memory_utilization: float
    judge_model: str     # Judge can use a separate model (e.g. Thinking variant)
    judge_thinking: bool # Enable thinking for the judge
    max_region_hypotheses: int     # Path B region candidates
    max_country_hypotheses: int    # Top-K countries before pre-filters (default 6)

    # RAG grounding
    rag_data_dir: str | None       # path to the vendored data/ tree
    rag_max_refs_per_round: int    # cap on reference images per evaluate/tournament step
    rag_max_refs_per_country: int  # cap per country

    # Tournament
    tournament_finalists: int      # max bracket size after pre-filters (default 4)


_REPO_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_config() -> VLMCouncilConfig:
    rag_data_dir = os.environ.get("VLM_DATA_DIR")
    if rag_data_dir is None and _REPO_DEFAULT_DATA_DIR.exists():
        rag_data_dir = str(_REPO_DEFAULT_DATA_DIR)

    return VLMCouncilConfig(
        vlm_model=os.environ.get("VLM_MODEL", "google/gemma-4-31b-it"),
        api_base=os.environ.get("VLM_API_BASE", "http://localhost:8000/v1"),
        max_model_len=int(os.environ.get("VLM_MAX_MODEL_LEN", "8192")),
        gpu_memory_utilization=float(os.environ.get("VLM_GPU_MEMORY_UTIL", "0.9")),
        judge_model=os.environ.get(
            "VLM_JUDGE_MODEL",
            os.environ.get("VLM_MODEL", "google/gemma-4-31b-it"),
        ),
        judge_thinking=os.environ.get("VLM_JUDGE_THINKING", "false").lower() in ("true", "1", "yes"),
        max_region_hypotheses=int(os.environ.get("VLM_MAX_REGION_HYPOTHESES", "4")),
        max_country_hypotheses=int(os.environ.get("VLM_MAX_COUNTRY_HYPOTHESES", "6")),
        rag_data_dir=rag_data_dir,
        rag_max_refs_per_round=int(os.environ.get("VLM_RAG_MAX_REFS_PER_ROUND", "6")),
        rag_max_refs_per_country=int(os.environ.get("VLM_RAG_MAX_REFS_PER_COUNTRY", "3")),
        tournament_finalists=int(os.environ.get("VLM_TOURNAMENT_FINALISTS", "4")),
    )
