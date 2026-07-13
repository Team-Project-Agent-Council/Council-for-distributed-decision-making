"""Run-lifecycle manager for the Progressive Narrowing demo.

Each `/api/demo/run` POST creates a `RunState` with:
  - asyncio.Queue receiving SSE-shaped events as the adapter progresses
  - background asyncio.Task driving the adapter
  - cached final result for the GET /runs/{id} fallback

The SSE endpoint subscribes to the queue and translates events into
`event: <type>\\ndata: <json>\\n\\n` frames.

Runs are kept in-memory (no DB persistence) and TTL-cleaned after 10 minutes
to bound memory. The dataset endpoint that backs the "Random" button reads
from the VLM-Council `Images/` folder + `georc_locations.csv` for ground
truth.
"""

from __future__ import annotations

import asyncio
import csv
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from council_adapters.progressive_narrowing_adapter import (
    run_progressive_narrowing,
)


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


@dataclass
class RunState:
    run_id: str
    image_url: str
    ground_truth: dict[str, Any] | None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    result: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None


_runs: dict[str, RunState] = {}
_RUN_TTL_SECONDS = 10 * 60


def _gc_runs() -> None:
    """Drop runs older than the TTL — best-effort cleanup."""
    now = time.time()
    stale = [
        rid
        for rid, s in _runs.items()
        if s.finished_at and (now - s.finished_at) > _RUN_TTL_SECONDS
    ]
    for rid in stale:
        _runs.pop(rid, None)


# ---------------------------------------------------------------------------
# Dataset (VLM-Council Images/ + georc_locations.csv)
# ---------------------------------------------------------------------------


# Resolve dataset path from `DEMO_DATASET_DIR`. The repo does not ship a
# bundled subset — the Street View images are too large to check in and
# too tied to a Google Maps licence to redistribute. Download the dataset
# from the project OneDrive (see the top-level README) and point this
# env var at the extracted folder to enable the "Random" button. Without
# it, `/api/demo/dataset/random` responds with a 404 and a hint.
_dataset_dir_env = os.environ.get("DEMO_DATASET_DIR", "").strip()
_DATASET_DIR: Path | None = Path(_dataset_dir_env) if _dataset_dir_env else None
_DATASET_CSV: Path | None = (
    _DATASET_DIR / "georc_locations.csv" if _DATASET_DIR else None
)


@dataclass(frozen=True)
class DatasetEntry:
    dataset_id: str  # filename without extension
    filename: str
    lat: float
    lng: float
    country_code: str


_dataset_cache: list[DatasetEntry] | None = None


def _load_dataset() -> list[DatasetEntry]:
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache
    # Guard against the unconfigured case: no DEMO_DATASET_DIR set means
    # the "Random" button is not usable. Return an empty list rather than
    # crashing so the router can respond with a helpful 404.
    if _DATASET_CSV is None or not _DATASET_CSV.exists():
        _dataset_cache = []
        return _dataset_cache
    out: list[DatasetEntry] = []
    with open(_DATASET_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out.append(
                    DatasetEntry(
                        dataset_id=Path(row["filename"]).stem,
                        filename=row["filename"],
                        lat=float(row["lat"]),
                        lng=float(row["lng"]),
                        country_code=row.get("country_code", ""),
                    )
                )
            except (KeyError, ValueError):
                continue
    _dataset_cache = out
    return _dataset_cache


def get_random_dataset_entry() -> DatasetEntry | None:
    entries = _load_dataset()
    if not entries:
        return None
    return random.choice(entries)


def find_dataset_entry(dataset_id: str) -> DatasetEntry | None:
    for e in _load_dataset():
        if e.dataset_id == dataset_id:
            return e
    return None


def read_dataset_image(entry: DatasetEntry) -> tuple[bytes, str]:
    if _DATASET_DIR is None:
        raise FileNotFoundError(
            "DEMO_DATASET_DIR is not set; cannot serve dataset images."
        )
    path = _DATASET_DIR / entry.filename
    data = path.read_bytes()
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/jpeg")
    return data, mime


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


async def start_run(
    image_bytes: bytes,
    mime: str,
    *,
    image_url: str,
    ground_truth: dict[str, Any] | None = None,
) -> RunState:
    """Create a new run and kick off the adapter as a background task.

    Returns the `RunState` so the router can immediately respond with
    `{ runId, imageUrl, groundTruth }` while the run keeps progressing in
    the background.
    """
    _gc_runs()

    run_id = f"demo-{uuid.uuid4().hex[:12]}"
    state = RunState(run_id=run_id, image_url=image_url, ground_truth=ground_truth)
    _runs[run_id] = state

    # Seed the queue with `run_started` so subscribers that connect after
    # the first event still see the initial payload.
    await state.queue.put({
        "type": "run_started",
        "ts": time.time(),
        "data": {
            "runId": run_id,
            "imageUrl": image_url,
            **({"groundTruth": ground_truth} if ground_truth else {}),
        },
    })

    async def emit(event_type: str, data: dict[str, Any]) -> None:
        await state.queue.put({"type": event_type, "ts": time.time(), "data": data})

    async def drive() -> None:
        try:
            result = await run_progressive_narrowing(image_bytes, mime, emit)
            state.result = result
        except Exception as exc:  # noqa: BLE001
            state.error = f"{type(exc).__name__}: {exc}"
            await state.queue.put({
                "type": "error",
                "ts": time.time(),
                "data": {"message": state.error},
            })
        finally:
            state.finished_at = time.time()
            await state.queue.put({"type": "done", "ts": time.time(), "data": {}})

    state.task = asyncio.create_task(drive(), name=f"demo-run-{run_id}")
    return state


def get_run(run_id: str) -> RunState | None:
    return _runs.get(run_id)


async def subscribe(run_id: str):
    """Async generator yielding event dicts until a `done` event is seen.

    A single subscriber per run is supported by the queue. If you need fan-out,
    wrap this in a broadcaster — for the demo a single `EventSource` is enough.
    """
    state = _runs.get(run_id)
    if state is None:
        return
    while True:
        evt = await state.queue.get()
        yield evt
        if evt["type"] == "done":
            return
