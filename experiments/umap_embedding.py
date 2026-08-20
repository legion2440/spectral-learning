"""Optional UMAP experiment for nonlinear two-dimensional visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.artifacts import create_run_directory, write_json
from utils.clustering import evaluate_kmeans
from utils.data_loader import load_dataset
from utils.preprocessing import TabularPreprocessor
from utils.visualization import plot_embedding_2d


def run_umap_embedding(
    input_path: str | Path,
    *,
    target: str | None = "quality",
    output_root: str | Path = "artifacts",
    max_samples: int = 5000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Path:
    """Run UMAP on a deterministic exploratory sample and save its artifacts."""
    try:
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError(
            "UMAP support is optional. Install it with "
            "'python -m pip install -r requirements-extra.txt'."
        ) from exc

    bundle = load_dataset(input_path, target=target)
    preprocessor = TabularPreprocessor().fit(bundle.features, bundle.feature_names)
    matrix = preprocessor.transform(bundle.features)

    if max_samples < 3:
        raise ValueError("max_samples must be at least 3")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if not 0.0 <= min_dist <= 1.0:
        raise ValueError("min_dist must be in [0, 1]")

    count = min(max_samples, len(matrix))
    if n_neighbors >= count:
        raise ValueError("n_neighbors must be smaller than the sampled row count")

    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(matrix), size=count, replace=False))
    sample = matrix[indices]

    model = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        transform_seed=random_state,
    )
    embedding = np.asarray(model.fit_transform(sample), dtype=float)

    if bundle.target is not None:
        labels = bundle.target[indices]
        label_name = "Target"
    else:
        labels = evaluate_kmeans(
            sample,
            n_clusters=min(6, count - 1),
            random_state=random_state,
        ).labels
        label_name = "Cluster"

    run_dir = create_run_directory(output_root, "umap")
    pd.DataFrame(embedding, columns=["umap_1", "umap_2"]).to_csv(
        run_dir / "embedding.csv", index=False
    )
    write_json(
        run_dir / "metrics.json",
        {
            "sample_size": count,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": "euclidean",
            "random_state": random_state,
        },
    )
    plot_embedding_2d(
        embedding,
        labels,
        run_dir / "umap_2d.png",
        title="UMAP nonlinear embedding",
        label_name=label_name,
        x_label="UMAP 1",
        y_label="UMAP 2",
    )
    return run_dir
