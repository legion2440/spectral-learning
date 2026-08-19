"""Reusable matrix validation and component-selection helpers."""

from __future__ import annotations

import numpy as np


def as_float_matrix(
    X: np.ndarray,
    *,
    name: str = "X",
    min_samples: int = 2,
) -> np.ndarray:
    """Return a validated two-dimensional finite float matrix."""
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if matrix.shape[0] < min_samples:
        raise ValueError(f"{name} must contain at least {min_samples} sample(s)")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def choose_component_count(
    *,
    max_components: int,
    cumulative_variance: np.ndarray,
    n_components: int | None,
    variance_threshold: float | None,
) -> int:
    """Choose an explicit component count or the minimum count meeting a variance target."""
    if max_components < 1:
        raise ValueError("max_components must be positive")

    if n_components is not None:
        if isinstance(n_components, bool) or not isinstance(n_components, int):
            raise TypeError("n_components must be an integer")
        if not 1 <= n_components <= max_components:
            raise ValueError(
                f"n_components must be between 1 and {max_components}, got {n_components}"
            )
        return n_components

    threshold = 0.95 if variance_threshold is None else float(variance_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("variance_threshold must be in the interval (0, 1]")

    cumulative = np.asarray(cumulative_variance, dtype=float)
    if cumulative.ndim != 1 or cumulative.size == 0:
        raise ValueError("cumulative_variance must be a non-empty vector")
    return min(int(np.searchsorted(cumulative, threshold, side="left") + 1), max_components)
