"""Council evaluation pipeline CLI.

Usage
-----
Run the full pipeline (council + metrics):

    python evaluate.py

Skip already-processed images (default) and run with 2 parallel council calls:

    python evaluate.py --concurrency 2

Only compute metrics from existing council_result.json files (no LLM calls):

    python evaluate.py --skip-council

LangSmith tracing:
    Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in your environment (or .env).
    Each image gets its own named trace under the configured project.
"""

from __future__ import annotations

import os
if os.environ.get("CHROMA_SQLITE_PATCH"):
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import asyncio
import os
from pathlib import Path

import click
try:
    from dotenv import load_dotenv  # type: ignore[import]
    # override=False: bash exports take precedence over .env values
    # This is critical for cluster runs where run_eval_gpu.sh exports 127.0.0.1:11434
    load_dotenv(override=False)
except ImportError:
    pass


def _print_langsmith_status() -> None:
    tracing = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true")
    api_key = bool(os.environ.get("LANGCHAIN_API_KEY"))
    project = os.environ.get("LANGCHAIN_PROJECT", "(default)")
    click.echo("[Local trace] Writing trace.jsonl per image -> results/<location-id>/trace.jsonl")
    if tracing and api_key:
        click.echo(f"[LangSmith]   Tracing ON in parallel - project: {project}")
    elif tracing and not api_key:
        click.echo("[LangSmith]   LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY not set - cloud tracing skipped.")
    else:
        click.echo("[LangSmith]   Tracing OFF (set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY to enable)")


async def _run(
    results_dir: Path,
    mapping: Path,
    output: Path,
    concurrency: int,
    skip_council: bool,
    run_name: str,
    verbose: bool,
    output_name: str,
) -> None:
    from evaluation.loader import load_samples
    from evaluation.report import build_report, print_report, write_csv
    from evaluation.runner import CouncilRunResult, run_batch, _is_done, RESULT_FILENAME
    import evaluation.runner as runner_module

    # Set the output filename for this run
    runner_module.RESULT_FILENAME = output_name

    _print_langsmith_status()

    click.echo(f"\nLoading samples from {results_dir} ...")
    samples = load_samples(results_dir, mapping)
    click.echo(f"  Found {len(samples)} locations with pre-computed result.json\n")

    if not samples:
        click.echo("No samples found. Exiting.")
        return

    if skip_council:
        # Load existing council_result.json without running the council
        import json
        results: list[CouncilRunResult] = []
        missing = []
        for s in samples:
            path = s.result_dir / "council_result.json"
            if path.exists():
                data = json.loads(path.read_text())
                known = CouncilRunResult.__dataclass_fields__
                filtered = {k: v for k, v in data.items() if k in known}
                for field in known:
                    filtered.setdefault(field, "")
                results.append(CouncilRunResult(**filtered))
            else:
                missing.append(s.location_id)
        if missing:
            click.echo(
                f"[warn] {len(missing)} location(s) have no council_result.json and will be skipped:\n"
                + "\n".join(f"  {m}" for m in missing)
            )
    else:
        done_already = sum(1 for s in samples if _is_done(s))
        todo = len(samples) - done_already
        click.echo(
            f"Council runs: {todo} to run, {done_already} cached "
            f"(concurrency={concurrency})\n"
        )

        def _progress(done: int, total: int, res: CouncilRunResult) -> None:
            if res.error:
                click.echo(f"  [ERR] [{done:3d}/{total}] {res.location_id} - {res.error[:60]}")
            else:
                click.echo(f"  [OK]  [{done:3d}/{total}] {res.location_id}")

        results = await run_batch(
            samples,
            concurrency=concurrency,
            run_name_prefix=run_name,
            skip_done=True,
            progress_callback=_progress,
            verbose=verbose,
        )

    if not results:
        click.echo("No results to evaluate.")
        return

    click.echo("")
    rows = build_report(samples, results)
    print_report(rows)

    write_csv(rows, output)
    click.echo(f"\nResults written to {output}")


@click.command()
@click.option(
    "--results-dir",
    default="results",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory containing per-location result.json files.",
)
@click.option(
    "--mapping",
    default="location-image-mapping.csv",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Ground-truth CSV (image_filename, lat, lng).",
)
@click.option(
    "--output",
    default="evaluation_results.csv",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Where to write the evaluation CSV.",
)
@click.option(
    "--concurrency",
    default=1,
    show_default=True,
    type=click.IntRange(1),
    help="Max parallel council graph invocations.",
)
@click.option(
    "--skip-council",
    is_flag=True,
    default=False,
    help="Skip running the council; only compute metrics from existing council_result.json files.",
)
@click.option(
    "--run-name",
    default="council-eval",
    show_default=True,
    help="Prefix for LangSmith run names.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-node progress lines while each council run executes.",
)
@click.option(
    "--output-name",
    default="council_optimized_result.json",
    show_default=True,
    help="Filename for per-location result JSON (stored in each location dir).",
)
def main(
    results_dir: str,
    mapping: str,
    output: str,
    concurrency: int,
    skip_council: bool,
    run_name: str,
    verbose: bool,
    output_name: str,
) -> None:
    asyncio.run(
        _run(
            results_dir=Path(results_dir),
            mapping=Path(mapping),
            output=Path(output),
            concurrency=concurrency,
            skip_council=skip_council,
            run_name=run_name,
            verbose=verbose,
            output_name=output_name,
        )
    )


if __name__ == "__main__":
    main()
