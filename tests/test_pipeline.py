import numpy as np
import pandas as pd

from models.pca_model import PCAFromScratch
from models.svd_model import SVDFromScratch
from utils.clustering import evaluate_kmeans
from utils.preprocessing import TabularPreprocessor


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
