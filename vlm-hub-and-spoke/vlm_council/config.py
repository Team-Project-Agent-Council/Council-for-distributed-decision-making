"""Configuration loaded from environment variables."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VLMCouncilConfig:
    """Immutable config for the VLM Council."""

    vlm_model: str
    api_base: str
    max_model_len: int
    gpu_memory_utilization: float
    max_discussion_rounds: int
    judge_model: str     # Judge can use a separate model (e.g. Thinking variant)
    judge_thinking: bool     # Enable thinking for the judge (Gemma 4: <|think|> token, same model)


def load_config() -> VLMCouncilConfig:
    return VLMCouncilConfig(
        vlm_model=os.environ.get("VLM_MODEL", "google/gemma-4-31b-it"),
        api_base=os.environ.get("VLM_API_BASE", "http://localhost:8000/v1"),
        max_model_len=int(os.environ.get("VLM_MAX_MODEL_LEN", "8192")),
        gpu_memory_utilization=float(os.environ.get("VLM_GPU_MEMORY_UTIL", "0.9")),
        max_discussion_rounds=int(os.environ.get("VLM_MAX_DISCUSSION_ROUNDS", "3")),
        judge_model=os.environ.get(
            "VLM_JUDGE_MODEL",
            os.environ.get("VLM_MODEL", "google/gemma-4-31b-it"),
        ),
        judge_thinking=os.environ.get("VLM_JUDGE_THINKING", "false").lower() in ("true", "1", "yes"),
    )
