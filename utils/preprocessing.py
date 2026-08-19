"""Persistable numerical preprocessing with schema validation and train-only statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self, Sequence

import numpy as np
import pandas as pd


class TabularPreprocessor:
    """Median-impute and standardize a fixed feature schema."""

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, frame: pd.DataFrame, feature_names: Sequence[str]) -> Self:
        """Fit imputation and scaling statistics from training data only."""
        names = [str(name) for name in feature_names]
        if not names:
            raise ValueError("At least one feature is required")
        if len(set(names)) != len(names):
            raise ValueError("Feature names must be unique")

        matrix = self._numeric_frame(frame, names)
        medians = matrix.median(axis=0, skipna=True)
        if medians.isna().any():
            bad = medians[medians.isna()].index.tolist()
            raise ValueError(f"Features contain no finite values: {bad}")

        imputed = matrix.fillna(medians).to_numpy(dtype=float)
        means = imputed.mean(axis=0)
        scales = imputed.std(axis=0, ddof=0)
        constant_mask = np.isclose(scales, 0.0)
        safe_scales = scales.copy()
        safe_scales[constant_mask] = 1.0

        self.feature_names_ = names
        self.medians_ = medians.to_numpy(dtype=float)
        self.mean_ = means
        self.scale_ = safe_scales
        self.constant_features_ = [
            name for name, is_constant in zip(names, constant_mask, strict=True) if is_constant
        ]
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply fitted imputation and standardization to a frame with the same schema."""
        self._require_fitted()
        matrix = self._numeric_frame(frame, self.feature_names_)
        for index, name in enumerate(self.feature_names_):
            matrix[name] = matrix[name].fillna(self.medians_[index])
        values = matrix.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Preprocessed feature matrix contains non-finite values")
        return (values - self.mean_) / self.scale_

    def fit_transform(self, frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
        """Fit preprocessing statistics and transform the same frame."""
        return self.fit(frame, feature_names).transform(frame)

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        """Undo standardization for a finite matrix in the fitted feature order."""
        self._require_fitted()
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"Expected a matrix with {len(self.feature_names_)} features"
            )
        if not np.isfinite(values).all():
            raise ValueError("Matrix contains NaN or infinite values")
        return values * self.scale_ + self.mean_

    def save(self, path: str | Path) -> Path:
        """Store schema and statistics as portable JSON."""
        self._require_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": self.feature_names_,
            "medians": self.medians_.tolist(),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "constant_features": self.constant_features_,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load preprocessing state from :meth:`save`."""
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to load preprocessor {source}: {exc}") from exc

        model = cls()
        model.feature_names_ = [str(name) for name in payload["feature_names"]]
        model.medians_ = np.asarray(payload["medians"], dtype=float)
        model.mean_ = np.asarray(payload["mean"], dtype=float)
        model.scale_ = np.asarray(payload["scale"], dtype=float)
        model.constant_features_ = [str(name) for name in payload["constant_features"]]
        expected = len(model.feature_names_)
        if not all(len(array) == expected for array in (model.medians_, model.mean_, model.scale_)):
            raise ValueError("Preprocessor artifact has inconsistent vector sizes")
        if not np.isfinite(model.medians_).all() or not np.isfinite(model.mean_).all():
            raise ValueError("Preprocessor artifact contains non-finite statistics")
        if not np.isfinite(model.scale_).all() or np.any(model.scale_ <= 0):
            raise ValueError("Preprocessor artifact contains invalid scales")
        model._fitted = True
        return model

    @staticmethod
    def _numeric_frame(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        numeric = pd.DataFrame(index=frame.index)
        for name in names:
            series = pd.to_numeric(frame[name], errors="coerce")
            numeric[name] = series.replace([np.inf, -np.inf], np.nan)
        if numeric.empty:
            raise ValueError("Feature frame is empty")
        return numeric

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted")
