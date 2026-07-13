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
    judge_thinking: bool
    image_token_budget: int


def load_config() -> VLMCouncilConfig:
    return VLMCouncilConfig(
        vlm_model=os.environ.get("VLM_MODEL", "google/gemma-4-31b-it"),
        api_base=os.environ.get("VLM_API_BASE", "http://localhost:8000/v1"),
        max_model_len=int(os.environ.get("VLM_MAX_MODEL_LEN", "16384")),
        gpu_memory_utilization=float(os.environ.get("VLM_GPU_MEMORY_UTIL", "0.9")),
        judge_thinking=os.environ.get("VLM_JUDGE_THINKING", "true").lower() in ("true", "1", "yes"),
        image_token_budget=int(os.environ.get("VLM_IMAGE_TOKEN_BUDGET", "1120")),
    )
