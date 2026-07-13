"""Unified plot style for the eval suite.

Single source of truth for matplotlib styling so every figure produced by
``eval/`` shares fonts, colors, sizes, and DPI. Existing plot modules call
``setup_plot_style()`` once at the top of each ``plot_*`` function, then
reach for ``STYLE.primary`` etc. in place of hardcoded color names.

Color palette is a Tableau-derived 7-color set. Semantic aliases
(``primary``, ``success``, ``error``, ``warning``, ``neutral``) keep call
sites readable.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class _Palette:
    primary: str = "#4C72B0"      # blue, default bar/line color
    success: str = "#55A868"      # green, correct/positive
    error: str = "#C44E52"        # red, wrong/negative
    warning: str = "#DD8452"      # orange, attention
    neutral: str = "#8C8C8C"      # gray, baselines, axes
    secondary: str = "#937860"    # brown, second-axis comparisons
    accent: str = "#8172B3"       # purple, highlight
    cycle: tuple[str, ...] = (
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3",
    )


STYLE = _Palette()


_DPI_DEFAULT = 200
_FIGSIZE_SINGLE = (6.0, 4.0)
_FIGSIZE_WIDE = (10.0, 6.0)


def setup_plot_style() -> None:
    """Apply the unified rcParams. Idempotent, safe to call repeatedly."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "Computer Modern Roman", "serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": _DPI_DEFAULT,
        "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(color=list(STYLE.cycle)),
    })


def figsize_single() -> tuple[float, float]:
    return _FIGSIZE_SINGLE


def figsize_wide() -> tuple[float, float]:
    return _FIGSIZE_WIDE


def add_figure_caption(ax: plt.Axes, caption: str, n: int | None = None) -> None:
    """Add a "Figure N: ..." caption below the axis.

    The caption is set via ``fig.suptitle`` placement at the bottom so it
    survives ``tight_layout``. ``n`` is optional, when given, prefixes
    "Figure {n}: ".
    """
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
