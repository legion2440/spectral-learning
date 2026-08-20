"""Singular Value Decomposition dimensionality reduction with explicit logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

import numpy as np

from utils.matrix_operations import (
    as_float_matrix,
    canonicalize_component_signs,
    choose_component_count,
)


class SVDFromScratch:
    """Centered SVD reducer built around ``numpy.linalg.svd``."""

    model_type = "svd"

    def __init__(
        self,
        n_components: int | None = None,
        variance_threshold: float | None = 0.95,
    ) -> None:
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self._fitted = False

    def fit(self, X: np.ndarray) -> Self:
        """Fit a centered SVD and choose the retained right-singular vectors."""
        matrix = as_float_matrix(X)
        n_samples, n_features = matrix.shape

        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

        variances = (singular_values**2) / (n_samples - 1)
        total_variance = float(variances.sum())
        if total_variance <= 0.0:
            raise ValueError("SVD requires data with non-zero total variance")

        all_ratios = variances / total_variance
        cumulative = np.cumsum(all_ratios)
        count = choose_component_count(
            max_components=len(singular_values),
            cumulative_variance=cumulative,
            n_components=self.n_components,
            variance_threshold=self.variance_threshold,
        )

        self.n_samples_seen_ = n_samples
        self.n_features_in_ = n_features
        self.n_components_ = count
        self.components_ = canonicalize_component_signs(vt[:count])
        self.singular_values_ = singular_values[:count]
        self.all_singular_values_ = singular_values
        self.explained_variance_ = variances[:count]
        self.explained_variance_ratio_ = all_ratios[:count]
        self.all_explained_variance_ = variances
        self.all_explained_variance_ratio_ = all_ratios
        self.cumulative_explained_variance_ = cumulative
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project samples onto retained right-singular vectors."""
        self._require_fitted()
        matrix = as_float_matrix(X, min_samples=1)
        self._validate_feature_count(matrix)
        return (matrix - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and reduce the same matrix."""
        return self.fit(X).transform(X)

    def inverse_transform(self, reduced: np.ndarray) -> np.ndarray:
        """Reconstruct samples from the retained latent dimensions."""
        self._require_fitted()
        matrix = as_float_matrix(reduced, min_samples=1)
        if matrix.shape[1] != self.n_components_:
            raise ValueError(
                f"Expected {self.n_components_} reduced components, got {matrix.shape[1]}"
            )
        return matrix @ self.components_ + self.mean_

    def save(self, path: str | Path) -> Path:
        """Persist fitted SVD parameters without pickle."""
        self._require_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_type": self.model_type,
            "n_components": self.n_components,
            "variance_threshold": self.variance_threshold,
            "n_components_": self.n_components_,
            "n_features_in_": self.n_features_in_,
            "n_samples_seen_": self.n_samples_seen_,
        }
        np.savez_compressed(
            output,
            metadata=np.array(json.dumps(metadata)),
            mean=self.mean_,
            components=self.components_,
            singular_values=self.singular_values_,
            all_singular_values=self.all_singular_values_,
            explained_variance=self.explained_variance_,
            explained_variance_ratio=self.explained_variance_ratio_,
            all_explained_variance=self.all_explained_variance_,
            all_explained_variance_ratio=self.all_explained_variance_ratio_,
            cumulative_explained_variance=self.cumulative_explained_variance_,
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a model produced by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            if metadata.get("model_type") != cls.model_type:
                raise ValueError("The artifact is not an SVD model")
            model = cls(
                n_components=metadata.get("n_components"),
                variance_threshold=metadata.get("variance_threshold"),
            )
            model.n_components_ = int(metadata["n_components_"])
            model.n_features_in_ = int(metadata["n_features_in_"])
            model.n_samples_seen_ = int(metadata["n_samples_seen_"])
            model.mean_ = payload["mean"].copy()
            model.components_ = payload["components"].copy()
            model.singular_values_ = payload["singular_values"].copy()
            model.all_singular_values_ = payload["all_singular_values"].copy()
            model.explained_variance_ = payload["explained_variance"].copy()
            model.explained_variance_ratio_ = payload[
                "explained_variance_ratio"
            ].copy()
            model.all_explained_variance_ = payload[
                "all_explained_variance"
            ].copy()
            model.all_explained_variance_ratio_ = payload[
                "all_explained_variance_ratio"
            ].copy()
            model.cumulative_explained_variance_ = payload[
                "cumulative_explained_variance"
            ].copy()
            model._fitted = True
            return model

    def _validate_feature_count(self, matrix: np.ndarray) -> None:
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {matrix.shape[1]}"
            )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SVD model is not fitted")
