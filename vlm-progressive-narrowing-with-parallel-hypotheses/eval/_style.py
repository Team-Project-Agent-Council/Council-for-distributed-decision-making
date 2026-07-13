"""Shared plot style for the eval suite."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class _Style:
    primary: str = "#2196F3"
    secondary: str = "#9C27B0"
    success: str = "#4CAF50"
    warning: str = "#FF9800"
    error: str = "#F44336"
    neutral: str = "#9E9E9E"
    accent: str = "#00BCD4"


STYLE = _Style()


def setup_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
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
