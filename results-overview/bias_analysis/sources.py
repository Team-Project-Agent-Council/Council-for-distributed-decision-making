"""Registry of approaches to pool into the cross-approach bias analysis.

Each source points at a directory of per-image ``result.json`` folders plus the
ground-truth CSV for that run. Paths are resolved relative to the repo root
(this package lives under ``results-overview/bias_analysis/``), so the analysis
runs from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# This file lives at <repo>/results-overview/bias_analysis/sources.py, so the
# repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_OVERVIEW = REPO_ROOT / "results-overview"


@dataclass(frozen=True)
class Source:
    name: str          # display name used in plots/report
    results_dir: Path  # dir of <image_id>/result.json folders
    gt_csv: Path       # ground-truth georc_locations.csv


def _council(overview_name: str, gt_approach_dir: str) -> Source:
    return Source(
        name=overview_name,
        results_dir=_OVERVIEW / overview_name / "council_run",
        gt_csv=REPO_ROOT / gt_approach_dir / "Images" / "georc_locations.csv",
    )


SOURCES: list[Source] = [
    _council("VLM Progressive Narrowing with Parallel Hypotheses",
             "vlm-progressive-narrowing-with-parallel-hypotheses"),
    _council("VLM PN + PH + Tournament", "vlm-pn-ph-tournament"),
    _council("VLM Tournament Only", "vlm-tournament-only"),
    _council("VLM Hub and Spoke", "vlm-hub-and-spoke"),
    _council("VLM Global Context Reguess", "vlm-global-context-reguess"),
    _council("VLM Debate", "vlm-debate"),
    Source(
        name="Baseline (single VLM)",
        results_dir=REPO_ROOT / "vlm-baseline" / "results",
        gt_csv=REPO_ROOT / "vlm-baseline" / "Images" / "georc_locations.csv",
    ),
]
