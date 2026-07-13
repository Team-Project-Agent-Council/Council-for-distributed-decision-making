"""LLM helpers, uses vLLM's OpenAI-compatible API via ChatOpenAI."""

import os

from langchain_openai import ChatOpenAI


_DEFAULTS = {
    "api_base": "http://localhost:8000/v1",
    "model": "google/gemma-4-31b-it",
}


def get_vlm(agent_name: str, thinking: bool = False) -> ChatOpenAI:
    """Return a ChatOpenAI pointing at the vLLM server for the given agent.

    Args:
        agent_name: Name of the agent (used for per-agent env var overrides).
        thinking: If True, enables thinking mode for Gemma-4 (used for judge).
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
        openai_api_key="EMPTY",
        openai_api_base=api_base,
        max_retries=2,
    )

    if thinking:
        kwargs["model_kwargs"] = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}

    return ChatOpenAI(**kwargs)
