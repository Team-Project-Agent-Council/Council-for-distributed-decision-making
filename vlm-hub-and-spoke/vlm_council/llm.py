"""LLM helpers uses vLLM's OpenAI-compatible API via ChatOpenAI."""

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

_DEFAULTS = {
    "api_base": "http://localhost:8000/v1",
    "model": "google/gemma-4-31b-it",
}


def get_vlm(agent_name: str) -> ChatOpenAI:
    """Return a ChatOpenAI pointing at the vLLM server for the given agent.

    Supports per-agent overrides via VLM_<AGENT>_MODEL, VLM_<AGENT>_API_BASE, etc.
    """
    prefix = f"VLM_{agent_name.upper()}"
    model = os.environ.get(
        f"{prefix}_MODEL",
        os.environ.get("VLM_MODEL", _DEFAULTS["model"]),
    )
    api_base = os.environ.get(
        f"{prefix}_API_BASE",
        os.environ.get("VLM_API_BASE", _DEFAULTS["api_base"]),
    )

    kwargs = dict(
        model=model,
        temperature=0,
        openai_api_key="EMPTY",  # vLLM doesn't need a real key
        openai_api_base=api_base,
        max_retries=2,
    )

    return ChatOpenAI(**kwargs)
