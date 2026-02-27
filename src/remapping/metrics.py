"""Evaluation metrics for comparing embeddings and predictions."""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation handling constant arrays (returns 0.0)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalization: str = "range",
) -> float:
    """Normalized RMSE.

    Args:
        normalization: ``'range'`` (max - min) or ``'std'``.
    """
    err = rmse(y_true, y_pred)
    if normalization == "range":
        denom = np.ptp(y_true)
    elif normalization == "std":
        denom = np.std(y_true)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    if denom == 0:
        return 0.0
    return float(err / denom)


def rsa_spearman(mat1: np.ndarray, mat2: np.ndarray) -> float:
    """Representational Similarity Analysis via Spearman rank correlation.

    Computes pairwise Euclidean distances within each matrix, then
    correlates the two distance vectors.

    Args:
        mat1, mat2: (n, k) arrays with the same number of rows.

    Returns:
        Spearman r between the upper-triangle distance vectors.
    """
    d1 = pdist(mat1, metric="euclidean")
    d2 = pdist(mat2, metric="euclidean")
    rho, _ = spearmanr(d1, d2)
    return float(rho)
