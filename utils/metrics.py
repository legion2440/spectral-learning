"""Numerical evaluation metrics shared by PCA, SVD and bonus experiments."""

from __future__ import annotations

import numpy as np

from utils.matrix_operations import as_float_matrix


def reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    """Return MSE and relative Frobenius reconstruction error."""
    left = as_float_matrix(original, name="original", min_samples=1)
    right = as_float_matrix(reconstructed, name="reconstructed", min_samples=1)
    if left.shape != right.shape:
        raise ValueError("original and reconstructed matrices must have the same shape")

    residual = left - right
    mse = float(np.mean(residual**2))
    denominator = float(np.linalg.norm(left, ord="fro"))
    relative = float(np.linalg.norm(residual, ord="fro") / denominator) if denominator else 0.0
    return {"mse": mse, "relative_frobenius_error": relative}


def variance_retained(explained_variance_ratio: np.ndarray) -> float:
    """Return the fraction of total variance represented by retained dimensions."""
    ratios = np.asarray(explained_variance_ratio, dtype=float)
    if ratios.ndim != 1 or ratios.size == 0 or not np.isfinite(ratios).all():
        raise ValueError("explained_variance_ratio must be a finite non-empty vector")
    return float(ratios.sum())
