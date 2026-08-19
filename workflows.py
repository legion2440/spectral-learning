"""End-to-end workflows for training, comparison, inference and experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.image_compression import run_image_compression
from experiments.signal_denoising import run_signal_denoising
from models.pca_model import PCAFromScratch
from models.svd_model import SVDFromScratch
from models.tsne_model import TSNEFromScratch
from utils.artifacts import create_run_directory, write_json
from utils.clustering import evaluate_kmeans
from utils.data_loader import basic_dataset_report, load_dataset, read_table
from utils.interpretation import top_component_features
from utils.metrics import reconstruction_metrics, variance_retained
from utils.preprocessing import TabularPreprocessor
from utils.visualization import (
    plot_component_loadings,
    plot_correlation_heatmap,
    plot_cumulative_variance,
    plot_embedding_2d,
    plot_embedding_3d,
    plot_feature_distributions,
    plot_reconstruction_curve,
    plot_target_distribution,
)

Reducer = PCAFromScratch | SVDFromScratch


def _build_reducer(
    method: str,
    *,
    n_components: int | None,
    variance_threshold: float,
) -> Reducer:
    if method == "pca":
        return PCAFromScratch(n_components=n_components, variance_threshold=variance_threshold)
    if method == "svd":
        return SVDFromScratch(n_components=n_components, variance_threshold=variance_threshold)
    raise ValueError(f"Unsupported method: {method}")


def _train_indices(n_samples: int, train_fraction: float, random_state: int) -> np.ndarray:
    if n_samples < 3:
        raise ValueError("At least three samples are required")
    if not 0.5 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0]")
    if train_fraction == 1.0:
        return np.arange(n_samples)
    train_size = max(2, min(n_samples - 1, int(round(n_samples * train_fraction))))
    rng = np.random.default_rng(random_state)
    return np.sort(rng.permutation(n_samples)[:train_size])


def _prepare_data(
    input_path: str | Path,
    *,
    target: str | None,
    train_fraction: float,
    random_state: int,
) -> tuple[Any, TabularPreprocessor, np.ndarray, np.ndarray]:
    bundle = load_dataset(input_path, target=target)
    train_indices = _train_indices(len(bundle.features), train_fraction, random_state)
    preprocessor = TabularPreprocessor().fit(
        bundle.features.iloc[train_indices], bundle.feature_names
    )
    all_matrix = preprocessor.transform(bundle.features)
    train_matrix = all_matrix[train_indices]
    return bundle, preprocessor, all_matrix, train_matrix


def _labels_for_plot(bundle: Any, cluster_labels: np.ndarray) -> np.ndarray:
    return bundle.target if bundle.target is not None else cluster_labels


def train_workflow(
    input_path: str | Path,
    *,
    method: str,
    target: str | None = "quality",
    output_root: str | Path = "artifacts",
    n_components: int | None = None,
    variance_threshold: float = 0.95,
    n_clusters: int = 6,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> Path:
    """Fit one reducer, evaluate it, and save a reusable transform bundle."""
    bundle, preprocessor, matrix, train_matrix = _prepare_data(
        input_path,
        target=target,
        train_fraction=train_fraction,
        random_state=random_state,
    )
    reducer = _build_reducer(
        method, n_components=n_components, variance_threshold=variance_threshold
    ).fit(train_matrix)
    reduced = reducer.transform(matrix)
    reconstructed = reducer.inverse_transform(reduced)

    original_clusters = evaluate_kmeans(
        matrix,
        n_clusters=n_clusters,
        reference_labels=bundle.target,
        random_state=random_state,
    )
    reduced_clusters = evaluate_kmeans(
        reduced,
        n_clusters=n_clusters,
        reference_labels=bundle.target,
        random_state=random_state,
    )

    run_dir = create_run_directory(output_root, method)
    reducer.save(run_dir / f"{method}_model.npz")
    preprocessor.save(run_dir / "preprocessor.json")

    metrics = {
        "method": method,
        "dataset": basic_dataset_report(bundle),
        "train_fraction": train_fraction,
        "random_state": random_state,
        "selected_components": reducer.n_components_,
        "variance_retained": variance_retained(reducer.explained_variance_ratio_),
        "reconstruction": reconstruction_metrics(matrix, reconstructed),
        "clustering_original": original_clusters.metrics,
        "clustering_reduced": reduced_clusters.metrics,
        "constant_features": preprocessor.constant_features_,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "component_interpretation.json",
        top_component_features(reducer.components_, bundle.feature_names),
    )

    reduced_frame = pd.DataFrame(
        reduced,
        columns=[f"component_{index}" for index in range(1, reduced.shape[1] + 1)],
    )
    if bundle.target is not None and bundle.target_name is not None:
        reduced_frame[bundle.target_name] = bundle.target
    if "wine_type" in bundle.frame.columns:
        reduced_frame["wine_type"] = bundle.frame["wine_type"].to_numpy()
    reduced_frame.to_csv(run_dir / "reduced.csv", index=False)

    plot_target = _labels_for_plot(bundle, reduced_clusters.labels)
    plot_feature_distributions(bundle.features, run_dir / "feature_distributions.png")
    plot_correlation_heatmap(bundle.features, run_dir / "feature_correlation.png")
    if bundle.target is not None:
        plot_target_distribution(bundle.target, run_dir / "target_distribution.png", title="Target distribution")
    plot_cumulative_variance(
        {method.upper(): reducer.cumulative_explained_variance_},
        run_dir / "cumulative_variance.png",
        threshold=variance_threshold if n_components is None else None,
    )
    if reduced.shape[1] >= 2:
        plot_embedding_2d(reduced, plot_target, run_dir / "embedding_2d.png", title=f"{method.upper()} reduced space")
    if reduced.shape[1] >= 3:
        plot_embedding_3d(reduced, plot_target, run_dir / "embedding_3d.png", title=f"{method.upper()} reduced space")
    plot_component_loadings(
        reducer.components_, bundle.feature_names, run_dir / "component_loadings.png"
    )
    return run_dir


def _reconstruction_curve(matrix: np.ndarray) -> tuple[list[int], list[float], list[float]]:
    max_components = min(matrix.shape)
    counts = list(range(1, max_components + 1))
    pca_errors: list[float] = []
    svd_errors: list[float] = []
    for count in counts:
        pca = PCAFromScratch(n_components=count).fit(matrix)
        pca_reconstructed = pca.inverse_transform(pca.transform(matrix))
        pca_errors.append(reconstruction_metrics(matrix, pca_reconstructed)["mse"])

        svd = SVDFromScratch(n_components=count).fit(matrix)
        svd_reconstructed = svd.inverse_transform(svd.transform(matrix))
        svd_errors.append(reconstruction_metrics(matrix, svd_reconstructed)["mse"])
    return counts, pca_errors, svd_errors


def compare_workflow(
    input_path: str | Path,
    *,
    target: str | None = "quality",
    output_root: str | Path = "artifacts",
    n_components: int | None = None,
    variance_threshold: float = 0.95,
    n_clusters: int = 6,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> Path:
    """Fit PCA and SVD on the same train-only preprocessing state and compare outcomes."""
    bundle, preprocessor, matrix, train_matrix = _prepare_data(
        input_path,
        target=target,
        train_fraction=train_fraction,
        random_state=random_state,
    )
    pca = PCAFromScratch(
        n_components=n_components, variance_threshold=variance_threshold
    ).fit(train_matrix)
    svd = SVDFromScratch(
        n_components=n_components, variance_threshold=variance_threshold
    ).fit(train_matrix)
    pca_reduced = pca.transform(matrix)
    svd_reduced = svd.transform(matrix)
    pca_reconstructed = pca.inverse_transform(pca_reduced)
    svd_reconstructed = svd.inverse_transform(svd_reduced)

    original_clusters = evaluate_kmeans(
        matrix,
        n_clusters=n_clusters,
        reference_labels=bundle.target,
        random_state=random_state,
    )
    pca_clusters = evaluate_kmeans(
        pca_reduced,
        n_clusters=n_clusters,
        reference_labels=bundle.target,
        random_state=random_state,
    )
    svd_clusters = evaluate_kmeans(
        svd_reduced,
        n_clusters=n_clusters,
        reference_labels=bundle.target,
        random_state=random_state,
    )

    run_dir = create_run_directory(output_root, "compare")
    preprocessor.save(run_dir / "preprocessor.json")
    pca.save(run_dir / "pca_model.npz")
    svd.save(run_dir / "svd_model.npz")

    metrics = {
        "dataset": basic_dataset_report(bundle),
        "train_fraction": train_fraction,
        "random_state": random_state,
        "original_clustering": original_clusters.metrics,
        "pca": {
            "selected_components": pca.n_components_,
            "variance_retained": variance_retained(pca.explained_variance_ratio_),
            "reconstruction": reconstruction_metrics(matrix, pca_reconstructed),
            "clustering": pca_clusters.metrics,
        },
        "svd": {
            "selected_components": svd.n_components_,
            "variance_retained": variance_retained(svd.explained_variance_ratio_),
            "reconstruction": reconstruction_metrics(matrix, svd_reconstructed),
            "clustering": svd_clusters.metrics,
        },
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "component_interpretation.json",
        {
            "pca": top_component_features(pca.components_, bundle.feature_names),
            "svd": top_component_features(svd.components_, bundle.feature_names),
        },
    )

    plot_feature_distributions(bundle.features, run_dir / "feature_distributions.png")
    plot_correlation_heatmap(bundle.features, run_dir / "feature_correlation.png")
    if bundle.target is not None:
        plot_target_distribution(bundle.target, run_dir / "target_distribution.png", title="Target distribution")
    plot_cumulative_variance(
        {
            "PCA": pca.cumulative_explained_variance_,
            "SVD": svd.cumulative_explained_variance_,
        },
        run_dir / "variance_comparison.png",
        threshold=variance_threshold if n_components is None else None,
    )

    pca_plot_labels = _labels_for_plot(bundle, pca_clusters.labels)
    svd_plot_labels = _labels_for_plot(bundle, svd_clusters.labels)
    if pca_reduced.shape[1] >= 2:
        plot_embedding_2d(pca_reduced, pca_plot_labels, run_dir / "pca_2d.png", title="PCA reduced space")
    if svd_reduced.shape[1] >= 2:
        plot_embedding_2d(svd_reduced, svd_plot_labels, run_dir / "svd_2d.png", title="SVD reduced space")
    if pca_reduced.shape[1] >= 3:
        plot_embedding_3d(pca_reduced, pca_plot_labels, run_dir / "pca_3d.png", title="PCA reduced space")
    if svd_reduced.shape[1] >= 3:
        plot_embedding_3d(svd_reduced, svd_plot_labels, run_dir / "svd_3d.png", title="SVD reduced space")
    plot_component_loadings(pca.components_, bundle.feature_names, run_dir / "pca_loadings.png")
    plot_component_loadings(svd.components_, bundle.feature_names, run_dir / "svd_loadings.png")

    counts, pca_errors, svd_errors = _reconstruction_curve(train_matrix)
    plot_reconstruction_curve(
        counts, pca_errors, svd_errors, run_dir / "reconstruction_curve.png"
    )

    if "wine_type" in bundle.frame.columns:
        for wine_type in sorted(bundle.frame["wine_type"].dropna().unique()):
            mask = bundle.frame["wine_type"].to_numpy() == wine_type
            if int(mask.sum()) < 2:
                continue
            safe_name = str(wine_type).replace("/", "_").replace("\\", "_")
            if pca_reduced.shape[1] >= 2:
                plot_embedding_2d(
                    pca_reduced[mask],
                    pca_clusters.labels[mask],
                    run_dir / f"pca_{safe_name}_clusters.png",
                    title=f"PCA clusters — {wine_type} wine",
                )
            if svd_reduced.shape[1] >= 2:
                plot_embedding_2d(
                    svd_reduced[mask],
                    svd_clusters.labels[mask],
                    run_dir / f"svd_{safe_name}_clusters.png",
                    title=f"SVD clusters — {wine_type} wine",
                )
    return run_dir


def load_reducer(path: str | Path) -> Reducer:
    """Load PCA or SVD by reading the safe JSON metadata stored inside the NPZ."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
    model_type = metadata.get("model_type")
    if model_type == "pca":
        return PCAFromScratch.load(source)
    if model_type == "svd":
        return SVDFromScratch.load(source)
    raise ValueError(f"Unsupported model type in artifact: {model_type}")


def transform_workflow(
    input_path: str | Path,
    *,
    model_path: str | Path,
    preprocessor_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Apply a saved train-time preprocessing/model bundle to new CSV data."""
    frame = read_table(input_path)
    preprocessor = TabularPreprocessor.load(preprocessor_path)
    reducer = load_reducer(model_path)
    matrix = preprocessor.transform(frame)
    reduced = reducer.transform(matrix)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        reduced,
        columns=[f"component_{index}" for index in range(1, reduced.shape[1] + 1)],
    ).to_csv(output, index=False)
    return output


def nonlinear_workflow(
    input_path: str | Path,
    *,
    target: str | None = "quality",
    output_root: str | Path = "artifacts",
    max_samples: int = 1000,
    perplexity: float = 30.0,
    iterations: int = 750,
    random_state: int = 42,
) -> Path:
    """Run optional exact NumPy t-SNE on a bounded exploratory sample."""
    bundle = load_dataset(input_path, target=target)
    preprocessor = TabularPreprocessor().fit(bundle.features, bundle.feature_names)
    matrix = preprocessor.transform(bundle.features)
    if max_samples < 3:
        raise ValueError("max_samples must be at least 3")
    count = min(max_samples, len(matrix))
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(matrix), size=count, replace=False))
    sample = matrix[indices]
    if perplexity >= count:
        raise ValueError("perplexity must be smaller than the sampled row count")

    model = TSNEFromScratch(
        perplexity=perplexity,
        n_iter=iterations,
        random_state=random_state,
    )
    embedding = model.fit_transform(sample)
    labels = (
        bundle.target[indices]
        if bundle.target is not None
        else evaluate_kmeans(sample, n_clusters=min(6, count - 1), random_state=random_state).labels
    )

    run_dir = create_run_directory(output_root, "tsne")
    pd.DataFrame(embedding, columns=["tsne_1", "tsne_2"]).to_csv(
        run_dir / "embedding.csv", index=False
    )
    write_json(
        run_dir / "metrics.json",
        {
            "sample_size": count,
            "perplexity": perplexity,
            "iterations": iterations,
            "kl_divergence": model.kl_divergence_,
            "random_state": random_state,
        },
    )
    plot_embedding_2d(embedding, labels, run_dir / "tsne_2d.png", title="Exact NumPy t-SNE")
    return run_dir


__all__ = [
    "compare_workflow",
    "load_reducer",
    "nonlinear_workflow",
    "run_image_compression",
    "run_signal_denoising",
    "train_workflow",
    "transform_workflow",
]
