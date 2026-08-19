"""K-means evaluation utilities for original and reduced representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from utils.matrix_operations import as_float_matrix


@dataclass(frozen=True)
class ClusteringResult:
    """Cluster assignments and serializable quality metrics."""

    labels: np.ndarray
    metrics: dict[str, float]


def evaluate_kmeans(
    X: np.ndarray,
    *,
    n_clusters: int = 6,
    reference_labels: np.ndarray | None = None,
    random_state: int = 42,
) -> ClusteringResult:
    """Fit K-means and calculate internal plus optional external separability metrics."""
    matrix = as_float_matrix(X)
    n_samples = matrix.shape[0]
    if not 2 <= n_clusters < n_samples:
        raise ValueError("n_clusters must be at least 2 and smaller than sample count")

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = model.fit_predict(matrix)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("K-means produced fewer than two clusters")

    metrics = {
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
    }

    if reference_labels is not None:
        reference = np.asarray(reference_labels)
        if reference.ndim != 1 or len(reference) != n_samples:
            raise ValueError("reference_labels must have one value per sample")
        metrics["adjusted_rand_index"] = float(adjusted_rand_score(reference, labels))
        metrics["normalized_mutual_information"] = float(
            normalized_mutual_info_score(reference, labels)
        )

    return ClusteringResult(labels=labels, metrics=metrics)
