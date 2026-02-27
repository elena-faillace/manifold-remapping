"""Manifold Remapping — mice head-direction analysis package."""

from .dataset import MiceDataset, Animals, MiceDataType
from .config import DATA_ROOT, FIGURES_ROOT

__all__ = [
    "MiceDataset",
    "Animals",
    "MiceDataType",
    "DATA_ROOT",
    "FIGURES_ROOT",
]
