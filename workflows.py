"""End-to-end workflows for training, comparison, inference and experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    plot_metric_sweep,
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


def _split_indices(
    n_samples: int,
    train_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    train = _train_indices(n_samples, train_fraction, random_state)
    test_mask = np.ones(n_samples, dtype=bool)
    test_mask[train] = False
    return train, np.flatnonzero(test_mask)


def _prepare_data(
    input_path: str | Path,
    *,
    target: str | None,
    train_fraction: float,
    random_state: int,
) -> tuple[Any, TabularPreprocessor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bundle = load_dataset(input_path, target=target)
    train_indices, test_indices = _split_indices(
        len(bundle.features), train_fraction, random_state
    )
    preprocessor = TabularPreprocessor().fit(
        bundle.features.iloc[train_indices], bundle.feature_names
    )
    all_matrix = preprocessor.transform(bundle.features)
    train_matrix = all_matrix[train_indices]
    return bundle, preprocessor, all_matrix, train_matrix, train_indices, test_indices


def _labels_for_plot(bundle: Any, cluster_labels: np.ndarray) -> np.ndarray:
    return bundle.target if bundle.target is not None else cluster_labels


def _reconstruction_report(
    matrix: np.ndarray,
    reconstructed: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    """Return overall, train and held-out reconstruction diagnostics."""
    report: dict[str, Any] = reconstruction_metrics(matrix, reconstructed)
    train_metrics = reconstruction_metrics(
        matrix[train_indices], reconstructed[train_indices]
    )
    report["train"] = train_metrics

    if len(test_indices):
        test_metrics = reconstruction_metrics(
            matrix[test_indices], reconstructed[test_indices]
        )
        report["test"] = test_metrics
        report["test_minus_train_mse"] = float(
            test_metrics["mse"] - train_metrics["mse"]
        )
        report["test_to_train_mse_ratio"] = (
            float(test_metrics["mse"] / train_metrics["mse"])
            if train_metrics["mse"] > 0.0
            else None
        )
    else:
        report["test"] = None
        report["test_minus_train_mse"] = None
        report["test_to_train_mse_ratio"] = None
    return report


def _reconstruct_with_prefix(model: Reducer, matrix: np.ndarray, count: int) -> np.ndarray:
    components = model.components_[:count]
    centered = matrix - model.mean_
    return (centered @ components.T) @ components + model.mean_


def _reconstruction_curve(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray | None = None,
) -> dict[str, list[int] | list[float] | None]:
    """Evaluate nested prefixes of one train-fitted PCA/SVD basis on train and test."""
    max_components = min(train_matrix.shape)
    counts = list(range(1, max_components + 1))
    pca = PCAFromScratch(n_components=max_components).fit(train_matrix)
    svd = SVDFromScratch(n_components=max_components).fit(train_matrix)

    pca_train: list[float] = []
    svd_train: list[float] = []
    pca_test: list[float] | None = [] if test_matrix is not None and len(test_matrix) else None
    svd_test: list[float] | None = [] if test_matrix is not None and len(test_matrix) else None

    for count in counts:
        pca_train.append(
            reconstruction_metrics(
                train_matrix, _reconstruct_with_prefix(pca, train_matrix, count)
            )["mse"]
        )
        svd_train.append(
            reconstruction_metrics(
                train_matrix, _reconstruct_with_prefix(svd, train_matrix, count)
            )["mse"]
        )
        if pca_test is not None and svd_test is not None and test_matrix is not None:
            pca_test.append(
                reconstruction_metrics(
                    test_matrix, _reconstruct_with_prefix(pca, test_matrix, count)
                )["mse"]
            )
            svd_test.append(
                reconstruction_metrics(
                    test_matrix, _reconstruct_with_prefix(svd, test_matrix, count)
                )["mse"]
            )

    return {
        "component_counts": counts,
        "pca_train_mse": pca_train,
        "svd_train_mse": svd_train,
        "pca_test_mse": pca_test,
        "svd_test_mse": svd_test,
    }


def _clustering_sweep(
    representations: Mapping[str, np.ndarray],
    reference_labels: np.ndarray,
    *,
    min_clusters: int = 2,
    max_clusters: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate clustering stability across k without choosing a best value post hoc."""
    if not representations:
        raise ValueError("At least one representation is required")
    sample_count = len(next(iter(representations.values())))
    upper = min(max_clusters, sample_count - 1)
    if upper < min_clusters:
        raise ValueError("Not enough samples for the requested clustering sweep")

    cluster_counts = list(range(min_clusters, upper + 1))
    results: dict[str, list[dict[str, float | int]]] = {}
    for name, matrix in representations.items():
        if len(matrix) != sample_count:
            raise ValueError("All clustering sweep representations must have equal row counts")
        entries: list[dict[str, float | int]] = []
        for cluster_count in cluster_counts:
            evaluated = evaluate_kmeans(
                matrix,
                n_clusters=cluster_count,
                reference_labels=reference_labels,
                random_state=random_state,
            )
            entries.append({"k": cluster_count, **evaluated.metrics})
        results[name] = entries
    return {"cluster_counts": cluster_counts, "representations": results}


def _wine_type_separability(
    bundle: Any,
    matrix: np.ndarray,
    train_matrix: np.ndarray,
    *,
    random_state: int,
) -> dict[str, Any] | None:
    """Measure whether red/white separability survives an explicit 11D-to-2D reduction."""
    if "wine_type" not in bundle.frame.columns or min(train_matrix.shape) < 2:
        return None

    labels = bundle.frame["wine_type"].fillna("<missing>").astype(str).to_numpy()
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or len(classes) >= len(matrix):
        return None

    cluster_count = int(len(classes))
    pca_2d = PCAFromScratch(n_components=2).fit(train_matrix).transform(matrix)
    svd_2d = SVDFromScratch(n_components=2).fit(train_matrix).transform(matrix)
    return {
        "reference": "wine_type",
        "class_counts": {
            str(name): int(count) for name, count in zip(classes, counts, strict=True)
        },
        "n_clusters": cluster_count,
        "dimensions": {
            "original": int(matrix.shape[1]),
            "pca": 2,
            "svd": 2,
        },
        "original": evaluate_kmeans(
            matrix,
            n_clusters=cluster_count,
            reference_labels=labels,
            random_state=random_state,
        ).metrics,
        "pca_2d": evaluate_kmeans(
            pca_2d,
            n_clusters=cluster_count,
            reference_labels=labels,
            random_state=random_state,
        ).metrics,
        "svd_2d": evaluate_kmeans(
            svd_2d,
            n_clusters=cluster_count,
            reference_labels=labels,
            random_state=random_state,
        ).metrics,
    }


def _component_alignment(pca: PCAFromScratch, svd: SVDFromScratch) -> dict[str, Any]:
    shared = min(pca.n_components_, svd.n_components_)
    pca_components = pca.components_[:shared]
    svd_components = svd.components_[:shared]
    numerator = np.sum(pca_components * svd_components, axis=1)
    denominator = np.linalg.norm(pca_components, axis=1) * np.linalg.norm(
        svd_components, axis=1
    )
    cosines = numerator / np.clip(denominator, 1e-15, None)
    return {
        "paired_components": shared,
        "cosine_similarity": [float(value) for value in cosines],
    }


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
    (
        bundle,
        preprocessor,
        matrix,
        train_matrix,
        train_indices,
        test_indices,
    ) = _prepare_data(
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
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "random_state": random_state,
        "selected_components": reducer.n_components_,
        "variance_retained": variance_retained(reducer.explained_variance_ratio_),
        "reconstruction": _reconstruction_report(
            matrix, reconstructed, train_indices, test_indices
        ),
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
        plot_target_distribution(
            bundle.target, run_dir / "target_distribution.png", title="Target distribution"
        )
    plot_cumulative_variance(
        {method.upper(): reducer.cumulative_explained_variance_},
        run_dir / "cumulative_variance.png",
        threshold=variance_threshold if n_components is None else None,
    )
    if reduced.shape[1] >= 2:
        plot_embedding_2d(
            reduced,
            plot_target,
            run_dir / "embedding_2d.png",
            title=f"{method.upper()} reduced space",
        )
    if reduced.shape[1] >= 3:
        plot_embedding_3d(
            reduced,
            plot_target,
            run_dir / "embedding_3d.png",
            title=f"{method.upper()} reduced space",
        )
    plot_component_loadings(
        reducer.components_, bundle.feature_names, run_dir / "component_loadings.png"
    )
    return run_dir


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
    (
        bundle,
        preprocessor,
        matrix,
        train_matrix,
        train_indices,
        test_indices,
    ) = _prepare_data(
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

    quality_sweep = None
    if bundle.target is not None:
        quality_sweep = _clustering_sweep(
            {
                "original": matrix,
                "pca": pca_reduced,
                "svd": svd_reduced,
            },
            bundle.target,
            random_state=random_state,
        )

    wine_type_separability = _wine_type_separability(
        bundle,
        matrix,
        train_matrix,
        random_state=random_state,
    )

    run_dir = create_run_directory(output_root, "compare")
    preprocessor.save(run_dir / "preprocessor.json")
    pca.save(run_dir / "pca_model.npz")
    svd.save(run_dir / "svd_model.npz")

    metrics = {
        "dataset": basic_dataset_report(bundle),
        "train_fraction": train_fraction,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "random_state": random_state,
        "original_clustering": original_clusters.metrics,
        "component_alignment": _component_alignment(pca, svd),
        "pca": {
            "selected_components": pca.n_components_,
            "variance_retained": variance_retained(pca.explained_variance_ratio_),
            "reconstruction": _reconstruction_report(
                matrix, pca_reconstructed, train_indices, test_indices
            ),
            "clustering": pca_clusters.metrics,
        },
        "svd": {
            "selected_components": svd.n_components_,
            "variance_retained": variance_retained(svd.explained_variance_ratio_),
            "reconstruction": _reconstruction_report(
                matrix, svd_reconstructed, train_indices, test_indices
            ),
            "clustering": svd_clusters.metrics,
        },
        "quality_clustering_sweep": quality_sweep,
        "wine_type_separability": wine_type_separability,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "component_interpretation.json",
        {
            "pca": top_component_features(pca.components_, bundle.feature_names),
            "svd": top_component_features(svd.components_, bundle.feature_names),
        },
    )
    if quality_sweep is not None:
        write_json(run_dir / "quality_clustering_sweep.json", quality_sweep)
    if wine_type_separability is not None:
        write_json(run_dir / "wine_type_separability.json", wine_type_separability)

    plot_feature_distributions(bundle.features, run_dir / "feature_distributions.png")
    plot_correlation_heatmap(bundle.features, run_dir / "feature_correlation.png")
    if bundle.target is not None:
        plot_target_distribution(
            bundle.target, run_dir / "target_distribution.png", title="Target distribution"
        )
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
        plot_embedding_2d(
            pca_reduced,
            pca_plot_labels,
            run_dir / "pca_2d.png",
            title="PCA reduced space",
        )
    if svd_reduced.shape[1] >= 2:
        plot_embedding_2d(
            svd_reduced,
            svd_plot_labels,
            run_dir / "svd_2d.png",
            title="SVD reduced space",
        )
    if pca_reduced.shape[1] >= 3:
        plot_embedding_3d(
            pca_reduced,
            pca_plot_labels,
            run_dir / "pca_3d.png",
            title="PCA reduced space",
        )
    if svd_reduced.shape[1] >= 3:
        plot_embedding_3d(
            svd_reduced,
            svd_plot_labels,
            run_dir / "svd_3d.png",
            title="SVD reduced space",
        )
    plot_component_loadings(
        pca.components_, bundle.feature_names, run_dir / "pca_loadings.png"
    )
    plot_component_loadings(
        svd.components_, bundle.feature_names, run_dir / "svd_loadings.png"
    )

    test_matrix = matrix[test_indices] if len(test_indices) else None
    curve = _reconstruction_curve(train_matrix, test_matrix)
    write_json(run_dir / "reconstruction_curve.json", curve)
    plot_reconstruction_curve(
        curve["component_counts"],
        curve["pca_train_mse"],
        curve["svd_train_mse"],
        run_dir / "reconstruction_curve.png",
        pca_test_errors=curve["pca_test_mse"],
        svd_test_errors=curve["svd_test_mse"],
    )

    if quality_sweep is not None:
        cluster_counts = quality_sweep["cluster_counts"]
        representations = quality_sweep["representations"]
        plot_metric_sweep(
            cluster_counts,
            {
                "Original": [
                    entry["adjusted_rand_index"] for entry in representations["original"]
                ],
                "PCA": [entry["adjusted_rand_index"] for entry in representations["pca"]],
                "SVD": [entry["adjusted_rand_index"] for entry in representations["svd"]],
            },
            run_dir / "quality_clustering_sweep.png",
            ylabel="Adjusted Rand Index",
            title="Quality clustering stability across k",
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
        else evaluate_kmeans(
            sample,
            n_clusters=min(6, count - 1),
            random_state=random_state,
        ).labels
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
    plot_embedding_2d(
        embedding, labels, run_dir / "tsne_2d.png", title="Exact NumPy t-SNE"
    )
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
