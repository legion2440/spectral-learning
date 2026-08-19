import numpy as np

from models.pca_model import PCAFromScratch
from utils.metrics import reconstruction_metrics


def _matrix() -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(80, 3))
    return np.column_stack((base, base[:, 0] * 2.0 + base[:, 1] * 0.2))


def test_pca_shapes_and_sorted_variance() -> None:
    matrix = _matrix()
    model = PCAFromScratch(n_components=2).fit(matrix)
    reduced = model.transform(matrix)
    assert reduced.shape == (80, 2)
    assert model.components_.shape == (2, 4)
    assert np.all(np.diff(model.all_explained_variance_) <= 1e-12)


def test_pca_reconstruction_improves_with_more_components() -> None:
    matrix = _matrix()
    low = PCAFromScratch(n_components=1).fit(matrix)
    high = PCAFromScratch(n_components=3).fit(matrix)
    low_error = reconstruction_metrics(matrix, low.inverse_transform(low.transform(matrix)))["mse"]
    high_error = reconstruction_metrics(matrix, high.inverse_transform(high.transform(matrix)))["mse"]
    assert high_error <= low_error


def test_pca_save_load_roundtrip(tmp_path) -> None:
    matrix = _matrix()
    model = PCAFromScratch(variance_threshold=0.9).fit(matrix)
    path = model.save(tmp_path / "pca.npz")
    loaded = PCAFromScratch.load(path)
    assert np.allclose(model.transform(matrix), loaded.transform(matrix))
