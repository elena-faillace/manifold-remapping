"""Manifold Remapping — mice head-direction analysis package."""

from .dataset import (
    MiceDataset,
    Animals,
    GROUP_ORDER,
    SESSION_ORDER,
    EXPERIMENT_ORDER,
    SESSION_TYPE_MAP,
    EXPERIMENT_TYPE_ORDER,
    EXPERIMENT_TYPE_COLORS,
    COLORS_EXPERIMENTS,
    SESSION_COLORS,
    COLORS_GROUPS,
    COLORS_SUBJECTS,
)
from .config import DATA_ROOT, FIGURES_ROOT

__all__ = [
    "MiceDataset",
    "Animals",
    "GROUP_ORDER",
    "SESSION_ORDER",
    "EXPERIMENT_ORDER",
    "SESSION_TYPE_MAP",
    "EXPERIMENT_TYPE_ORDER",
    "EXPERIMENT_TYPE_COLORS",
    "COLORS_EXPERIMENTS",
    "SESSION_COLORS",
    "COLORS_GROUPS",
    "COLORS_SUBJECTS",
    "DATA_ROOT",
    "FIGURES_ROOT",
]
