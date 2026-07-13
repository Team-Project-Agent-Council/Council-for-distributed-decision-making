#!/usr/bin/env python3
"""Download the 500 GeoRC images from HuggingFace into a flat directory.

Dataset: https://huggingface.co/datasets/mohit-talreja/GeoRC

Run this script **locally** (on your laptop or any machine with outbound
internet access) and then rsync the resulting directory to the shared
`datasets` workspace on bwUniCluster. Compute nodes on the cluster do not
have direct outbound HTTPS, so downloading from a SLURM job is not
supported. See section 3 of the top-level README for the full workflow.

The dataset repo is organised as 100 sub-directories (one per
sample/place). Each sample holds 5 PNGs (5 views of the same place)
plus reasoning/metadata files. Every image counts as a separate
evaluation location in the ground-truth CSV, so this download gives
you 100 samples x 5 views = 500 evaluation images total. This script fetches
**only the PNGs** and writes them flat to the output directory, matching
the layout ``vlm_council.batch`` expects::

    Images/
        1NJsXTxIF9GGMDxC_1.png
        1NJsXTxIF9GGMDxC_2.png
        ...
        (500 files)

The ground-truth CSV (``georc_locations.csv``) is not built here; it is
uploaded separately alongside the images on the cluster.

Usage
-----
    python3 scripts/download_dataset.py                         # default: Images/
    python3 scripts/download_dataset.py --output-dir dataset/Images
    python3 scripts/download_dataset.py --workers 16            # more parallelism
    python3 scripts/download_dataset.py --dry-run               # just list what would be fetched

After the download, rsync the directory (plus your ground-truth CSV) to
the cluster's shared `datasets` workspace::

    rsync -avh --progress dataset/ <user>@uc3.scc.kit.edu:$(ssh <user>@uc3.scc.kit.edu 'ws_find datasets')/

Resume
------
Files that already exist and are non-empty are skipped, so a partial run
can be resumed by simply invoking the script again.

Requirements
------------
Python >= 3.10 (uses PEP 604 unions and PEP 585 generics), standard
library only. No ``datasets`` or ``huggingface_hub`` dependency.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HF_API = "https://huggingface.co/api/datasets/mohit-talreja/GeoRC"
HF_RAW = "https://huggingface.co/datasets/mohit-talreja/GeoRC/resolve/main"

# 100 HF-repo sub-directories (each = 1 sample/place), 5 PNG rounds each.
# 100 samples x 5 rounds = 500 evaluation images in total.
ROUNDS_PER_LOCATION = 5

# Retry / rate-limit knobs.
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0
HTTP_TIMEOUT_SECONDS = 60


def _http_get(url: str, *, binary: bool = False) -> bytes | str:
    """GET with retries and a small backoff. Raises after the final attempt."""
    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "geobench-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as r:
                data = r.read()
            return data if binary else data.decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as e:
            last_error = e
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def _list_location_ids() -> list[str]:
    """Return the 100 top-level directory names in the dataset repo.

    Each directory is one sample (one place, 5 views).
    """
    tree = json.loads(_http_get(f"{HF_API}/tree/main"))
    return sorted(entry["path"] for entry in tree if entry.get("type") == "directory")


def _download_image(location_id: str, round_index: int, output_dir: Path) -> tuple[str, bool]:
    """Download a single PNG. Returns (filename, was_downloaded)."""
    filename = f"{location_id}_{round_index}.png"
    dest = output_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return filename, False
    url = f"{HF_RAW}/{location_id}/{filename}"
    payload = _http_get(url, binary=True)
    # Atomic write via .tmp so a crashed download doesn't leave a half file.
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return filename, True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the 500 GeoRC images from HuggingFace into a flat directory."
    )
    parser.add_argument(
        "--output-dir",
        default="Images",
        help="Directory to write PNGs into (default: Images/).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel HTTP workers (default: 8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List everything that would be downloaded, but don't write files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target directory: {output_dir}")
    print("Listing locations from HuggingFace ...")
    location_ids = _list_location_ids()
    print(f"Found {len(location_ids)} samples "
          f"({len(location_ids) * ROUNDS_PER_LOCATION} evaluation images total).")

    if args.dry_run:
        print("\n[dry-run] would download:")
        for lid in location_ids[:3]:
            for r in range(1, ROUNDS_PER_LOCATION + 1):
                print(f"  {lid}/{lid}_{r}.png")
        print(f"  ... and {len(location_ids) * ROUNDS_PER_LOCATION - 15} more")
        return 0

    print(f"\nDownloading images with {args.workers} workers ...")
    tasks = [
        (lid, r) for lid in location_ids for r in range(1, ROUNDS_PER_LOCATION + 1)
    ]
    downloaded = 0
    skipped = 0
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_image, lid, r, output_dir): (lid, r)
            for lid, r in tasks
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            lid, r = futures[fut]
            try:
                _, was_downloaded = fut.result()
                if was_downloaded:
                    downloaded += 1
                else:
                    skipped += 1
            except Exception as e:
                errors += 1
                print(f"  ERR  {lid}_{r}.png: {e}", file=sys.stderr)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} (downloaded={downloaded}, skipped={skipped}, "
                      f"errors={errors})")

    print(f"\nDone. {downloaded} new, {skipped} already present, {errors} errors.")
    if errors:
        print("Re-run the script to retry failed files.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
