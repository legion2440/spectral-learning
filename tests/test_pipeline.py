from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.pca_model import PCAFromScratch
from models.svd_model import SVDFromScratch
from utils.clustering import evaluate_kmeans
from utils.data_loader import read_table
from utils.preprocessing import TabularPreprocessor
from workflows import _reconstruction_curve


def test_end_to_end_reduction_and_clustering() -> None:
    rng = np.random.default_rng(21)
    first = rng.normal(loc=-2.0, scale=0.4, size=(30, 4))
    second = rng.normal(loc=2.0, scale=0.4, size=(30, 4))
    frame = pd.DataFrame(np.vstack((first, second)), columns=list("abcd"))
    reference = np.array([0] * 30 + [1] * 30)

    matrix = TabularPreprocessor().fit_transform(frame, frame.columns)
    for reducer in (PCAFromScratch(n_components=2), SVDFromScratch(n_components=2)):
        reduced = reducer.fit_transform(matrix)
        result = evaluate_kmeans(reduced, n_clusters=2, reference_labels=reference)
        assert reduced.shape == (60, 2)
        assert result.metrics["silhouette"] > 0.5


def test_reconstruction_curve_uses_nested_train_fitted_basis() -> None:
    rng = np.random.default_rng(33)
    train = rng.normal(size=(100, 5)) * np.array([5.0, 3.0, 2.0, 1.0, 0.4])
    test = rng.normal(size=(40, 5)) * np.array([5.0, 3.0, 2.0, 1.0, 0.4])

    curve = _reconstruction_curve(train, test)
    pca_train = np.asarray(curve["pca_train_mse"], dtype=float)
    svd_train = np.asarray(curve["svd_train_mse"], dtype=float)
    pca_test = np.asarray(curve["pca_test_mse"], dtype=float)
    svd_test = np.asarray(curve["svd_test_mse"], dtype=float)

    assert np.all(np.diff(pca_train) <= 1e-10)
    assert np.all(np.diff(pca_test) <= 1e-10)
    assert np.allclose(pca_train, svd_train, atol=1e-10, rtol=1e-10)
    assert np.allclose(pca_test, svd_test, atol=1e-10, rtol=1e-10)


def test_read_table_rejects_duplicate_source_headers(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.csv"
    source.write_text("a,a,b\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column names"):
        read_table(source)


def test_read_table_rejects_empty_csv_cleanly(tmp_path: Path) -> None:
    source = tmp_path / "empty.csv"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset is empty"):
        read_table(source)
