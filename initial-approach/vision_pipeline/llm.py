import os
from langchain_ollama import ChatOllama

_DEFAULTS = {
    "host": "http://localhost:11434",
    "model": "qwen3:32b",
    "num_ctx": "4096",
}


def get_llm(agent_name: str):
    prefix = agent_name.upper()
    model = os.environ.get(f"{prefix}_MODEL", _DEFAULTS["model"])
    host = os.environ.get(f"{prefix}_OLLAMA_HOST", os.environ.get("OLLAMA_HOST", _DEFAULTS["host"]))
    num_ctx = int(os.environ.get(f"{prefix}_NUM_CTX", _DEFAULTS["num_ctx"]))
    return ChatOllama(model=model, temperature=0, num_ctx=num_ctx, base_url=host)


def get_thinking_prefix(agent_name: str, call: str = "default") -> str:
    """Return /think or /no_think based on env var <AGENT>_THINK_<CALL>.
    call: name of the specific LLM call within the agent (e.g. 'tool', 'reason')
    Falls back to <AGENT>_THINK, then to /no_think.
    """
    prefix = agent_name.upper()
    key_specific = f"{prefix}_THINK_{call.upper()}"
    key_agent = f"{prefix}_THINK"
    value = os.environ.get(key_specific, os.environ.get(key_agent, "false"))
    return "/think" if value.lower() == "true" else "/no_think"
