"""Local file-based tracer - structured JSONL trace without LangSmith.

Each council run writes one file:
    results/<location-id>/trace.jsonl

Every line is a self-contained JSON event:
    {"event": "llm_start"|"llm_end"|"tool_start"|"tool_end"|"chain_start"|"chain_end"|"error",
     "run_id": "...", "parent_run_id": "...|null",
     "name": "...", "ts": "<ISO-8601>",
     ... event-specific fields ...}

This gives you a full audit trail of every LLM call and tool call without
needing a LangSmith account.  Load with ``load_trace()`` for analysis.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sid(uid: UUID | None) -> str | None:
    return str(uid) if uid else None


class LocalTracer(BaseCallbackHandler):
    """Append one JSON line per event to a JSONL file.

    Thread-safe: uses a lock so concurrent tool calls in the same run
    don't interleave partial writes.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate/create on init so each run gets a fresh file
        path.write_text("")

    def _write(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    # -- LLM ----------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write({
            "event": "llm_start",
            "run_id": _sid(run_id),
            "parent_run_id": _sid(parent_run_id),
            "name": serialized.get("name", serialized.get("id", ["?"])[-1]),
            "ts": _now(),
            "prompts": prompts,
        })

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        outputs = [
            [{"text": g.text, "type": g.type} for g in gen]
            for gen in response.generations
        ]
        self._write({
            "event": "llm_end",
            "run_id": _sid(run_id),
            "parent_run_id": _sid(parent_run_id),
            "ts": _now(),
            "outputs": outputs,
            "llm_output": response.llm_output,
        })

    # -- Tools ---------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write({
            "event": "tool_start",
            "run_id": _sid(run_id),
            "parent_run_id": _sid(parent_run_id),
            "name": serialized.get("name", "?"),
            "ts": _now(),
            "input": input_str,
        })

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write({
            "event": "tool_end",
            "run_id": _sid(run_id),
            "parent_run_id": _sid(parent_run_id),
            "ts": _now(),
            "output": str(output),
        })

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write({
            "event": "tool_error",
            "run_id": _sid(run_id),
            "parent_run_id": _sid(parent_run_id),
            "ts": _now(),
            "error": str(error),
        })

    # -- Chains / Graph nodes ------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or serialized.get("name") or serialized.get("id", ["?"])[-1]
        self._write({
            "event": "chain_start",
            "run_id": _sid(run_id),
            "parent_run_id": _sid(parent_run_id),
            "name": name,
            "ts": _now(),
        })

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write({
            "event": "chain_end",
            "run_id": _sid(run_id),
            "parent_run_id": _sid(parent_run_id),
            "ts": _now(),
        })

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write({
            "event": "chain_error",
            "run_id": _sid(run_id),
            "parent_run_id": _sid(parent_run_id),
            "ts": _now(),
            "error": str(error),
        })


# ---------------------------------------------------------------------------
# Reader helper
# ---------------------------------------------------------------------------

def load_trace(path: Path) -> list[dict]:
    """Load a trace.jsonl file into a list of event dicts."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
