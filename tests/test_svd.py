import numpy as np

from models.svd_model import SVDFromScratch
from utils.metrics import reconstruction_metrics


def _matrix() -> np.ndarray:
    rng = np.random.default_rng(11)
    left = rng.normal(size=(90, 2))
    return np.column_stack((left, left[:, 0] - 0.4 * left[:, 1], rng.normal(scale=0.05, size=90)))


def test_svd_shapes_and_variance() -> None:
    matrix = _matrix()
    model = SVDFromScratch(n_components=2).fit(matrix)
    reduced = model.transform(matrix)
    assert reduced.shape == (90, 2)
    assert model.components_.shape == (2, 4)
    assert np.isclose(model.all_explained_variance_ratio_.sum(), 1.0)


def test_svd_reconstruction_improves_with_rank() -> None:
    matrix = _matrix()
    low = SVDFromScratch(n_components=1).fit(matrix)
    high = SVDFromScratch(n_components=3).fit(matrix)
    low_error = reconstruction_metrics(matrix, low.inverse_transform(low.transform(matrix)))["mse"]
    high_error = reconstruction_metrics(matrix, high.inverse_transform(high.transform(matrix)))["mse"]
    assert high_error <= low_error


def test_svd_save_load_roundtrip(tmp_path) -> None:
    matrix = _matrix()
    model = SVDFromScratch(variance_threshold=0.9).fit(matrix)
    path = model.save(tmp_path / "svd.npz")
    loaded = SVDFromScratch.load(path)
    assert np.allclose(model.transform(matrix), loaded.transform(matrix))
