"""VLM Council v12, comprehensive evaluation suite.

Two stages:
  - Stage 1 (deterministic, CPU): geo bias, per-agent metrics, cross-agent influence
  - Stage 2 (LLM-as-judge, GPU): role adherence, argument quality, assertiveness,
    cross-agent decision influence, graded by a judge LLM via vLLM

Entry point: ``python -m eval --help``
"""

__all__ = ["loader", "geo", "agents", "influence", "judge", "judge_aggregate", "report"]
