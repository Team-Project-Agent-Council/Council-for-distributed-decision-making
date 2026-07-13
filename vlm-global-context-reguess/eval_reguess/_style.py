"""Unified plot style for the eval_reguess evaluation suite."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class _Style:
    primary: str = "#4C72B0"
    success: str = "#55A868"
    error: str = "#C44E52"
    warning: str = "#DD8452"
    neutral: str = "#8172B2"
    cycle: tuple[str, ...] = (
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B2", "#937860", "#DA8BC3",
    )


STYLE = _Style()


def setup_plot_style() -> None:
    """Apply unified rcParams. Idempotent, safe to call repeatedly."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(color=list(STYLE.cycle)),
    })


# --- Helpers needed by heatmap.py -----------------------------------------

def figsize_single() -> tuple[float, float]:
    return (6.0, 4.0)


def figsize_wide() -> tuple[float, float]:
    return (10.0, 6.0)


def add_figure_caption(ax, caption: str, n: int | None = None) -> None:
    """Add a "Figure N: ..." caption below the axis (survives tight_layout)."""
    text = f"Figure {n}: {caption}" if n is not None else caption
    fig = ax.get_figure()
    fig.text(
        0.5, 0.005, text,
        ha="center", va="bottom", fontsize=9, style="italic", color="#555555",
        wrap=True,
    )


# --- Vector output (SVG + PDF alongside every PNG) -------------------------
# Every plot module calls setup_plot_style() then fig.savefig(<path>.png).
# We patch Figure.savefig once so any .png write also emits sibling .svg/.pdf
# with the same basename. This covers all existing modules plus heatmap.py
# without touching individual call sites.

import os as _os

from matplotlib.figure import Figure as _Figure

_VECTOR_FORMATS = ("svg", "pdf")
_orig_savefig = _Figure.savefig


def _savefig_with_vectors(self, fname, *args, **kwargs):
    result = _orig_savefig(self, fname, *args, **kwargs)
    # Only mirror real filesystem targets ending in .png (skip buffers/streams).
    if isinstance(fname, (str, _os.PathLike)):
        path = _os.fspath(fname)
        if path.lower().endswith(".png"):
            base = path[:-4]
            vector_kwargs = dict(kwargs)
            vector_kwargs.pop("dpi", None)  # DPI is irrelevant for vector output
            for ext in _VECTOR_FORMATS:
                try:
                    _orig_savefig(self, f"{base}.{ext}", *args, **vector_kwargs)
                except Exception:
                    pass  # never let vector export break the primary PNG
    return result


if getattr(_Figure.savefig, "_emits_vectors", False) is False:
    _savefig_with_vectors._emits_vectors = True
    _Figure.savefig = _savefig_with_vectors
