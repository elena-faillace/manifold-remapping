"""Signal-processing helpers for tuning-curve and firing-rate data."""

import numpy as np


def smooth_tuning_curves_circularly(
    tuning_curves: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """Circular moving-average smoothing of tuning curves.

    Args:
        tuning_curves: (n_points, n_neurons) array.
        kernel_size: width of the uniform kernel.

    Returns:
        Smoothed array with same shape as input.
    """
    n_points = tuning_curves.shape[0]
    kernel = np.ones(kernel_size) / kernel_size
    pad = kernel_size // 2

    smoothed = []
    for i in range(tuning_curves.shape[1]):
        padded = np.pad(tuning_curves[:, i], pad_width=pad, mode="wrap")
        conv = np.convolve(padded, kernel, mode="valid")
        smoothed.append(conv)

    # Trim if padding added 1 extra point
    if len(smoothed[0]) == n_points + 1:
        smoothed = [s[:n_points] for s in smoothed]

    return np.array(smoothed).T


def get_tuning_curves(
    firing_rates: np.ndarray,
    phi: np.ndarray,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Standalone version of ``MiceDataset.get_tuning_curves``.

    Bins firing rates by head-direction angle.

    Args:
        firing_rates: (T, N).
        phi: (T,) angles in degrees.
        n_points: number of angular bins.

    Returns:
        ring_neural: (n_points, N).
        phi_bins: (n_points,) bin centres.
    """
    from .dataset import MiceDataset
    return MiceDataset.get_tuning_curves(firing_rates, phi, n_points)


def average_by_phi_bin(
    embedding: np.ndarray,
    phi: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a T×K trajectory into angular bins.

    Args:
        embedding: (T, K) array (e.g. PCA scores).
        phi: (T,) angles in degrees.
        n_bins: number of bins spanning [0, 360).

    Returns:
        binned: (n_bins, K) mean embedding per bin.
        bin_centres: (n_bins,) angle centres.
    """
    phi_mod = phi % 360
    dphi = 360 / n_bins
    idx = np.clip(np.floor(phi_mod / dphi).astype(int), 0, n_bins - 1)

    binned = np.zeros((n_bins, embedding.shape[1]))
    counts = np.zeros(n_bins, dtype=int)
    for i in range(len(phi)):
        binned[idx[i]] += embedding[i]
        counts[idx[i]] += 1

    mask = counts > 0
    binned[mask] /= counts[mask, None]

    bin_centres = np.arange(n_bins) * dphi
    return binned, bin_centres
