"""Feature-level interpretation helpers for PCA and SVD components."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def top_component_features(
    components: np.ndarray,
    feature_names: Sequence[str],
    *,
    top_n: int = 5,
) -> list[dict[str, object]]:
    """Return the strongest signed feature loadings for each retained component."""
    matrix = np.asarray(components, dtype=float)
    names = [str(name) for name in feature_names]
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("Component matrix and feature names are inconsistent")
    if top_n < 1:
        raise ValueError("top_n must be positive")

    result: list[dict[str, object]] = []
    for component_index, row in enumerate(matrix, start=1):
        indices = np.argsort(np.abs(row))[::-1][: min(top_n, len(row))]
        loadings = [
            {"feature": names[index], "loading": float(row[index])}
            for index in indices
        ]
        result.append({"component": component_index, "top_features": loadings})
    return result
