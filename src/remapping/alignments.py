"""Alignment methods: Canonical Correlation Analysis and Procrustes."""

import logging

import numpy as np
from scipy.linalg import qr, svd, inv


def canoncorr(
    X: np.ndarray,
    Y: np.ndarray,
    fullReturn: bool = False,
) -> np.ndarray | tuple:
    """Canonical Correlation Analysis — line-by-line port from MATLAB ``canoncorr``.

    Args:
        X: (n, p1) observation matrix.
        Y: (n, p2) observation matrix.
        fullReturn: if True return ``(A, B, r, U, V)``; otherwise just ``r``.

    Returns:
        r: canonical correlations (always returned).
        A, B: canonical coefficients (if *fullReturn*).
        U, V: canonical scores (if *fullReturn*).
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        logging.warning("Not enough samples, might cause problems")

    X = X - np.mean(X, 0)
    Y = Y - np.mean(Y, 0)

    Q1, T11, perm1 = qr(X, mode="economic", pivoting=True, check_finite=True)
    rankX = int(np.sum(
        np.abs(np.diagonal(T11))
        > np.finfo(type(np.abs(T11[0, 0]))).eps * max(n, p1)
    ))
    if rankX == 0:
        logging.error("stats:canoncorr:BadData = X")
    elif rankX < p1:
        logging.warning("stats:canoncorr:NotFullRank = X")
        Q1 = Q1[:, :rankX]
        T11 = T11[:rankX, :rankX]

    Q2, T22, perm2 = qr(Y, mode="economic", pivoting=True, check_finite=True)
    rankY = int(np.sum(
        np.abs(np.diagonal(T22))
        > np.finfo(type(np.abs(T22[0, 0]))).eps * max(n, p2)
    ))
    if rankY == 0:
        logging.error("stats:canoncorr:BadData = Y")
    elif rankY < p2:
        logging.warning("stats:canoncorr:NotFullRank = Y")
        Q2 = Q2[:, :rankY]
        T22 = T22[:rankY, :rankY]

    d = min(rankX, rankY)
    L, D, M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver="gesdd")
    M = M.T

    A = inv(T11) @ L[:, :d] * np.sqrt(n - 1)
    B = inv(T22) @ M[:, :d] * np.sqrt(n - 1)
    r = D[:d]
    r[r >= 1] = 1
    r[r <= 0] = 0

    if not fullReturn:
        return r

    # Restore full-size coefficients in correct order
    stackedA = np.vstack((A, np.zeros((p1 - rankX, d))))
    newA = np.zeros(stackedA.shape)
    for i, j in enumerate(perm1):
        newA[j, :] = stackedA[i, :]

    stackedB = np.vstack((B, np.zeros((p2 - rankY, d))))
    newB = np.zeros(stackedB.shape)
    for i, j in enumerate(perm2):
        newB[j, :] = stackedB[i, :]

    U = X @ newA
    V = Y @ newB
    return newA, newB, r, U, V


def procrustes(
    X: np.ndarray,
    Y: np.ndarray,
    scaling: bool = True,
    reflection: str | bool = "best",
) -> tuple[float, np.ndarray, dict]:
    """Procrustes alignment (port of MATLAB ``procrustes``).

    Finds translation, rotation (+ optional scaling/reflection) of *Y* to
    best match *X* in least-squares sense.

    Args:
        X: (n, m) target coordinates.
        Y: (n, m) input coordinates.
        scaling: allow scaling component.
        reflection: ``'best'``, ``True``, or ``False``.

    Returns:
        d: normalised residual sum of squared errors.
        Z: transformed Y.
        tform: dict with ``'rotation'``, ``'scale'``, ``'translation'``.
    """
    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2).sum()
    ssY = (Y0 ** 2).sum()

    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros((n, m - my))), axis=1)

    A = X0.T @ Y0
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = V @ U.T

    if reflection != "best":
        have_reflection = np.linalg.det(T) < 0
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = V @ U.T

    traceTA = s.sum()

    if scaling:
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * (Y0 @ T) + muX
    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * (Y0 @ T) + muX

    if my < m:
        T = T[:my, :]
    c = muX - b * (muY @ T)

    tform = {"rotation": T, "scale": b, "translation": c}
    return d, Z, tform
