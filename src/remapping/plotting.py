"""Plotting helpers: colours, style defaults, figure-saving utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_ROOT


class BehaviorColors:
    """Colour constants for behavioural variables."""

    def __init__(self):
        self.position = {"x": "#3498db", "y": "#9b59b6"}
        self.speed = "#f39c12"
        self.angle = "#27ae60"
        self.heatmaps = {
            "xy_position": "viridis",
            "angles": sns.husl_palette(h=0.5, s=2, l=0.8, as_cmap=True),
            "neu_activity": "inferno",
        }
        self.angular_position = sns.husl_palette(h=0.5, s=2, l=0.8, as_cmap=True)

    def get_position_color(self, axis: str = "x") -> str:
        return self.position.get(axis, "#888888")

    def get_speed_color(self) -> str:
        return self.speed

    def get_angle_color(self) -> str:
        return self.angle

    def get_heatmap_cmap(self, data_type: str = "xy_position"):
        return self.heatmaps.get(data_type, "viridis")


class PlotStyle:
    """Standard figure style for A4 paper (min 6–8 pt text).

    Figure widths:
        FULL_WIDTH  = 6.7"  (single-column, ~170 mm usable on A4)
        HALF_WIDTH  = 3.3"  (two panels side-by-side)
        THIRD_WIDTH = 2.2"  (three panels)
    """

    FULL_WIDTH = 6.7   # inches — single-column A4 with 20 mm margins
    HALF_WIDTH = 3.3
    THIRD_WIDTH = 2.2

    def __init__(self):
        plt.style.use("ggplot")
        plt.rcParams.update({
            # Font sizes (all ≥ 7 pt for A4 readability)
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 9,
            "figure.titlesize": 11,
            # Lines & markers
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            # Axes
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Ticks
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            # Grid
            "grid.linewidth": 0.3,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Font
            "font.size": 8,
            "font.family": "sans-serif",
        })


def get_figures_path(*subdirs: str) -> Path:
    """Return ``FIGURES_ROOT / subdirs``, creating directories as needed.

    Example::

        path = get_figures_path("1.embeddings", "pca_rings")
        fig.savefig(path / "ring_overlay.pdf", bbox_inches="tight")
    """
    p = FIGURES_ROOT
    for s in subdirs:
        p = p / s
    p.mkdir(parents=True, exist_ok=True)
    return p
