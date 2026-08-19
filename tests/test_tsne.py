import numpy as np

from models.tsne_model import TSNEFromScratch


def test_tsne_returns_finite_embedding() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(24, 4))
    model = TSNEFromScratch(perplexity=5, n_iter=250, learning_rate=50, random_state=3)
    embedding = model.fit_transform(matrix)
    assert embedding.shape == (24, 2)
    assert np.isfinite(embedding).all()
    assert np.isfinite(model.kl_divergence_)
