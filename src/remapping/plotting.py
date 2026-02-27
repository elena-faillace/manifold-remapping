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
    """Apply a standard ``ggplot``-based style on instantiation."""

    def __init__(self, title_fontsize: int = 16):
        self.title_fontsize = title_fontsize
        plt.style.use("ggplot")
        plt.rcParams.update({
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
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
