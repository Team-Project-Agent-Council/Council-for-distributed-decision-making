"""Async batch runner: invoke the council for each SampleRecord and persist results."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluation.graph import build_eval_graph
from evaluation.loader import SampleRecord
from evaluation.tracer import LocalTracer

# Build one graph (no vision node) shared across all evaluation runs
_graph = build_eval_graph()


@dataclass
class CouncilRunResult:
    location_id: str
    image_filename: str
    country_result: str
    linguistic_result: str
    landscape_result: str
    botanics_result: str
    regulatory_result: str
    infrastructure_result: str
    cultural_result: str
    rag_result: str
    error: str = ""


# Configurable output filename - set via evaluate.py --output-name
RESULT_FILENAME = "council_optimized_result.json"


def _council_result_path(record: SampleRecord) -> Path:
    return record.result_dir / RESULT_FILENAME


def _trace_path(record: SampleRecord) -> Path:
    return record.result_dir / "trace.jsonl"


def _is_done(record: SampleRecord) -> bool:
    path = _council_result_path(record)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return bool(data.get("country_result")) and not data.get("error")
    except Exception:
        return False


def _langsmith_active() -> bool:
    tracing = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true")
    api_key = bool(os.environ.get("LANGCHAIN_API_KEY"))
    return tracing and api_key


async def run_single(
    record: SampleRecord,
    run_name_prefix: str = "council-eval",
    verbose: bool = False,
) -> CouncilRunResult:
    """Run the council for one sample and persist the result JSON.

    Always writes a local trace.jsonl alongside council_result.json.
    If LangSmith env vars are set, traces are also sent there in parallel.
    """
    import time

    config: dict = {
        "run_name": f"{run_name_prefix}/{record.location_id}",
        "tags": ["evaluation"],
        "callbacks": [LocalTracer(_trace_path(record))],
    }
    langsmith_project = os.environ.get("LANGCHAIN_PROJECT")
    if langsmith_project:
        config["metadata"] = {"project": langsmith_project}

    loc_short = record.location_id[:25]

    if verbose:
        desc_preview = record.general_description[:80].replace("\n", " ")
        print(f"  [{loc_short}] start - {desc_preview!r}", flush=True)

    try:
        t0 = time.monotonic()

        state = {}
        async for chunk in _graph.astream(
            {
                "image_path": record.image_filename,
                "general_description": record.general_description,
                "crop_descriptions": record.crop_descriptions,
            },
            config=config,
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                state.update(update)
                if verbose:
                    elapsed_so_far = time.monotonic() - t0
                    print(f"  [{loc_short}]   {node_name} ({elapsed_so_far:.0f}s)", flush=True)

        elapsed = time.monotonic() - t0
        result = CouncilRunResult(
            location_id=record.location_id,
            image_filename=record.image_filename,
            country_result=state.get("country_result", ""),
            linguistic_result=state.get("linguistic_result", ""),
            landscape_result=state.get("landscape_result", ""),
            botanics_result=state.get("botanics_result", ""),
            regulatory_result=state.get("regulatory_result", ""),
            infrastructure_result=state.get("infrastructure_result", ""),
            cultural_result=state.get("cultural_result", ""),
            rag_result=state.get("rag_result", ""),
        )
        if verbose:
            country = ""
            for line in result.country_result.split("\n"):
                if line.lower().startswith("country"):
                    country = line.split(":", 1)[-1].strip().rstrip(".")
                    break
            print(f"  [{loc_short}] done in {elapsed:.0f}s -> {country or '???'}", flush=True)

    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        result = CouncilRunResult(
            location_id=record.location_id,
            image_filename=record.image_filename,
            country_result="",
            linguistic_result="",
            landscape_result="",
            botanics_result="",
            regulatory_result="",
            infrastructure_result="",
            cultural_result="",
            rag_result="",
            error=str(exc),
        )
        if verbose:
            print(f"  [{loc_short}] FAILED in {elapsed:.0f}s - {exc}", flush=True)

    _council_result_path(record).write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False)
    )
    return result


async def run_batch(
    records: list[SampleRecord],
    concurrency: int = 1,
    run_name_prefix: str = "council-eval",
    skip_done: bool = True,
    progress_callback=None,
    verbose: bool = False,
) -> list[CouncilRunResult]:
    """Run council over all records with bounded concurrency.

    Args:
        records: List of SampleRecords to process.
        concurrency: Max simultaneous council graph invocations.
        run_name_prefix: Prefix for LangSmith run names.
        skip_done: If True, skip records that already have council_result.json.
        progress_callback: Optional ``callable(done: int, total: int, result)``
            called after each record completes.
        verbose: If True, print per-node progress lines while each run executes.
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[CouncilRunResult] = []
    done_count = 0

    async def _bounded(record: SampleRecord) -> CouncilRunResult:
        nonlocal done_count
        async with sem:
            if skip_done and _is_done(record):
                # Load from disk rather than re-running
                data = json.loads(_council_result_path(record).read_text())
                known = CouncilRunResult.__dataclass_fields__
                filtered = {k: v for k, v in data.items() if k in known}
                for field in known:
                    filtered.setdefault(field, "")
                res = CouncilRunResult(**filtered)
            else:
                try:
                    res = await asyncio.wait_for(
                        run_single(record, run_name_prefix, verbose=verbose),
                        timeout=1200,  # 20 min max per image
                    )
                except asyncio.TimeoutError:
                    loc = record.location_id
                    if verbose:
                        print(f"  [{loc[:25]}] [x] TIMED OUT after 1200s", flush=True)
                    res = CouncilRunResult(
                        location_id=loc,
                        image_filename=record.image_filename,
                        country_result="",
                        linguistic_result="",
                        landscape_result="",
                        botanics_result="",
                        regulatory_result="",
                        infrastructure_result="",
                        cultural_result="",
                        rag_result="",
                        error="Timed out after 1200s",
                    )
                    _council_result_path(record).write_text(
                        json.dumps(asdict(res), indent=2, ensure_ascii=False)
                    )
        done_count += 1
        if progress_callback:
            progress_callback(done_count, len(records), res)
        return res

    tasks = [asyncio.create_task(_bounded(r)) for r in records]
    results = await asyncio.gather(*tasks)
    return list(results)
