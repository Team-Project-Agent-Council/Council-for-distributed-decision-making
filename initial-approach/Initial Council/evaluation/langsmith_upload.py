"""Upload previously recorded trace.jsonl files to LangSmith.

The local traces written by LocalTracer use LangChain's standard callback
event schema.  This script reconstructs the run tree and pushes it to
LangSmith via the low-level Client API so that runs appear in the UI
exactly as if live tracing had been active.

Usage
-----
# Upload a single location
python -m evaluation.langsmith_upload \
    --location location-0e7d882b-f10c-4d6e-a526-46b4c64cc57b

# Upload every trace in the results directory
python -m evaluation.langsmith_upload --results-dir results

Required env vars
-----------------
LANGCHAIN_API_KEY   - LangSmith API key (ls__...)
LANGCHAIN_PROJECT   - project name to upload into (default: "council-eval")
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client

load_dotenv(override=True)

from evaluation.tracer import load_trace

# Maps our JSONL event names to LangSmith run types
_RUN_TYPE: dict[str, str] = {
    "chain_start": "chain",
    "llm_start": "llm",
    "tool_start": "tool",
}

_END_EVENTS = {"chain_end", "llm_end", "tool_end"}
_ERROR_EVENTS = {"chain_error", "llm_error", "tool_error"}


def upload_trace(trace_path: Path, project_name: str) -> str:
    """Upload one trace.jsonl to LangSmith and return the root-run URL.

    Always generates fresh run IDs so the same trace can be uploaded to
    multiple projects without hitting 409 conflicts.

    Args:
        trace_path:   Path to the trace.jsonl file.
        project_name: LangSmith project to upload into.

    Returns:
        URL string pointing to the root run in LangSmith.
    """
    events = load_trace(trace_path)
    if not events:
        raise ValueError(f"No events found in {trace_path}")

    client = Client()  # picks up LANGCHAIN_API_KEY from env

    # Remap every original run_id to a fresh UUID so we never conflict
    id_map: dict[str, uuid.UUID] = {}

    def _new_id(original: str) -> uuid.UUID:
        if original not in id_map:
            id_map[original] = uuid.uuid4()
        return id_map[original]

    root_run_id: str | None = None

    for ev in events:
        run_id = _new_id(ev["run_id"])
        parent_run_id = (
            _new_id(ev["parent_run_id"]) if ev.get("parent_run_id") else None
        )
        ts = datetime.datetime.fromisoformat(ev["ts"])
        event_name: str = ev["event"]

        if event_name in _RUN_TYPE:
            inputs = _extract_inputs(ev, event_name)
            extra: dict = {}
            if parent_run_id is None:
                # Tag the root run with the location so it's identifiable in the UI
                extra = {"metadata": {"location_id": trace_path.parent.name}}
            client.create_run(
                id=run_id,
                parent_run_id=parent_run_id,
                name=ev.get("name", "?"),
                run_type=_RUN_TYPE[event_name],
                inputs=inputs,
                start_time=ts,
                project_name=project_name,
                extra=extra,
            )
            if parent_run_id is None:
                root_run_id = str(run_id)

        elif event_name in _END_EVENTS:
            if event_name == "tool_end":
                outputs: Any = {"output": ev["output"]} if "output" in ev else {}
            else:
                outputs = ev.get("outputs")
                if isinstance(outputs, list):
                    outputs = {"generations": outputs}
                elif outputs is None:
                    outputs = {}
            client.update_run(run_id, end_time=ts, outputs=outputs)

        elif event_name in _ERROR_EVENTS:
            client.update_run(run_id, end_time=ts, error=ev.get("error", "unknown error"))

    if root_run_id is None:
        root_run_id = str(next(iter(id_map.values())))

    run = client.read_run(root_run_id)
    if run.app_path:
        return f"https://smith.langchain.com{run.app_path}"
    return f"https://smith.langchain.com/runs/{root_run_id}"


def _extract_inputs(ev: dict, event_name: str) -> dict:
    """Build the inputs dict appropriate for each start-event type."""
    if event_name == "llm_start":
        return {"prompts": ev.get("prompts", [])}
    if event_name == "tool_start":
        return {"input": ev.get("input", "")}
    # chain_start - inputs were not captured by LocalTracer
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Upload trace.jsonl files to LangSmith."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing location-<UUID> subdirectories (default: results)",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Upload a single location-<UUID> folder instead of all",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("LANGCHAIN_PROJECT", "eval-uploads"),
        help="LangSmith project name (default: $LANGCHAIN_PROJECT or 'eval-uploads')",
    )
    args = parser.parse_args()

    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("ERROR: LANGCHAIN_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    results_root = Path(args.results_dir)

    if args.location:
        paths = [results_root / args.location / "trace.jsonl"]
    else:
        paths = sorted(results_root.glob("*/trace.jsonl"))

    if not paths:
        print(f"No trace.jsonl files found under {results_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Uploading {len(paths)} trace(s) to project '{args.project}' ...")
    errors = 0
    for p in paths:
        if not p.exists():
            print(f"  SKIP {p.parent.name} - file not found")
            continue
        try:
            url = upload_trace(p, args.project)
            print(f"  OK   {p.parent.name}  ->  {url}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {p.parent.name} - {exc}", file=sys.stderr)
            errors += 1

    sys.exit(1 if errors else 0)
