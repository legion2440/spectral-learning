import numpy as np

from models.pca_model import PCAFromScratch
from models.svd_model import SVDFromScratch
from utils.matrix_operations import canonicalize_component_signs
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


def test_component_sign_canonicalization_is_idempotent() -> None:
    components = np.array(
        [
            [-0.2, 0.9, 0.1],
            [0.3, -0.4, -0.8],
            [-1.0, 0.2, 0.1],
        ]
    )
    canonical = canonicalize_component_signs(components)
    assert np.allclose(canonical, canonicalize_component_signs(canonical))
    pivots = np.argmax(np.abs(canonical), axis=1)
    assert np.all(canonical[np.arange(len(canonical)), pivots] >= 0.0)


def test_pca_and_svd_components_align_on_separated_spectrum() -> None:
    rng = np.random.default_rng(19)
    matrix = rng.normal(size=(300, 4)) * np.array([7.0, 4.0, 2.0, 0.5])
    pca = PCAFromScratch(n_components=4).fit(matrix)
    svd = SVDFromScratch(n_components=4).fit(matrix)
    assert np.allclose(pca.components_, svd.components_, atol=1e-8, rtol=1e-8)
