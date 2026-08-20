"""Dataset loading and lightweight cleaning for numerical spectral workflows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    """A cleaned source frame plus numerical features and an optional target."""

    frame: pd.DataFrame
    features: pd.DataFrame
    target: np.ndarray | None
    feature_names: list[str]
    target_name: str | None


def _validate_raw_header(source: Path) -> None:
    """Detect duplicate source headers before pandas can mangle their names."""
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(65536)
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unable to read dataset {source}: {exc}") from exc

    if not sample.strip():
        raise ValueError("Dataset is empty")

    try:
        dialect = csv.Sniffer().sniff(sample)
        header = next(csv.reader(StringIO(sample), dialect))
    except (csv.Error, StopIteration):
        return

    names = [str(value).strip() for value in header]
    if len(set(names)) != len(names):
        raise ValueError("Dataset contains duplicate column names")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV-like table with automatic delimiter detection."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    if not source.is_file():
        raise ValueError(f"Dataset path is not a file: {source}")

    _validate_raw_header(source)
    try:
        frame = pd.read_csv(source, sep=None, engine="python")
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        csv.Error,
    ) as exc:
        raise ValueError(f"Unable to read dataset {source}: {exc}") from exc

    if frame.empty:
        raise ValueError("Dataset is empty")
    frame.columns = [str(column).strip() for column in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Dataset contains duplicate column names")
    return frame


def load_dataset(
    path: str | Path,
    *,
    target: str | None = "quality",
    drop_duplicates: bool = True,
) -> DatasetBundle:
    """Load a dataset and select all numeric-compatible columns except the target."""
    frame = read_table(path)
    if drop_duplicates:
        frame = frame.drop_duplicates().reset_index(drop=True)

    target_values: np.ndarray | None = None
    target_name: str | None = None
    if target is not None:
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' is missing")
        numeric_target = pd.to_numeric(frame[target], errors="coerce")
        valid_target = numeric_target.notna() & np.isfinite(numeric_target.to_numpy(dtype=float))
        frame = frame.loc[valid_target].reset_index(drop=True)
        numeric_target = numeric_target.loc[valid_target].reset_index(drop=True)
        if frame.empty:
            raise ValueError("No rows remain after removing invalid target values")
        target_values = numeric_target.to_numpy()
        target_name = target

    candidate_columns = [column for column in frame.columns if column != target]
    numeric_columns: list[str] = []
    converted: dict[str, pd.Series] = {}
    for column in candidate_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            numeric_columns.append(column)
            converted[column] = series.replace([np.inf, -np.inf], np.nan)

    if not numeric_columns:
        raise ValueError("Dataset does not contain usable numerical feature columns")

    features = pd.DataFrame(converted, index=frame.index)
    return DatasetBundle(
        frame=frame,
        features=features,
        target=target_values,
        feature_names=numeric_columns,
        target_name=target_name,
    )


def basic_dataset_report(bundle: DatasetBundle) -> dict[str, object]:
    """Return serializable dataset diagnostics used by experiment metadata."""
    missing = bundle.features.isna().sum()
    return {
        "rows": int(len(bundle.features)),
        "features": int(bundle.features.shape[1]),
        "feature_names": list(bundle.feature_names),
        "missing_values": {name: int(value) for name, value in missing.items()},
        "duplicate_rows_after_cleaning": int(bundle.frame.duplicated().sum()),
        "target": bundle.target_name,
    }
