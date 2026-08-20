"""Plotting utilities for exploratory analysis and dimensionality-reduction results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _finish(fig: plt.Figure, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_target_distribution(values: np.ndarray, path: str | Path, *, title: str) -> Path:
    """Plot a target-value count distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=np.asarray(values), ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    return _finish(fig, path)


def plot_feature_distributions(
    features: pd.DataFrame,
    path: str | Path,
    *,
    max_features: int = 12,
) -> Path:
    """Plot compact histograms for the first numerical features."""
    columns = list(features.columns[:max_features])
    if not columns:
        raise ValueError("No features available for distribution plot")
    cols = min(3, len(columns))
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), columns, strict=False):
        sns.histplot(features[name].dropna(), kde=True, ax=ax)
        ax.set_title(name)
    for ax in axes.ravel()[len(columns) :]:
        ax.axis("off")
    fig.suptitle("Feature distributions", y=1.01)
    return _finish(fig, path)


def plot_correlation_heatmap(features: pd.DataFrame, path: str | Path) -> Path:
    """Plot feature correlations after numeric coercion."""
    numeric = features.apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0.0, ax=ax)
    ax.set_title("Feature correlation")
    return _finish(fig, path)


def _curves_overlap(curves: Mapping[str, np.ndarray]) -> bool:
    """Return whether PCA and SVD curves coincide within numerical precision."""
    if "PCA" not in curves or "SVD" not in curves:
        return False
    pca_values = np.asarray(curves["PCA"], dtype=float)
    svd_values = np.asarray(curves["SVD"], dtype=float)
    return pca_values.shape == svd_values.shape and np.allclose(
        pca_values, svd_values, rtol=1e-10, atol=1e-12
    )


def plot_cumulative_variance(
    curves: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    threshold: float | None = None,
) -> Path:
    """Plot cumulative explained variance for one or more reducers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    overlap = _curves_overlap(curves)
    styles = (("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"))
    for index, (label, values) in enumerate(curves.items()):
        cumulative = np.asarray(values, dtype=float)
        linestyle, marker = styles[index % len(styles)]
        if overlap and label == "SVD":
            linestyle = "None"
        ax.plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,
            markersize=6,
            label=label,
        )
    if threshold is not None:
        ax.axhline(
            threshold,
            linestyle=":",
            linewidth=1.8,
            color="0.35",
            label=f"target {threshold:.0%}",
        )
    if overlap:
        ax.text(
            0.98,
            0.04,
            "PCA and SVD curves overlap within numerical precision",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    return _finish(fig, path)


def _embedding_label(title: str, label_name: str | None) -> str:
    if label_name is not None:
        return label_name
    return "Cluster" if "cluster" in title.lower() else "Target"


def plot_embedding_2d(
    reduced: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    *,
    title: str,
    label_name: str | None = None,
) -> Path:
    """Plot the first two reduced dimensions colored by target or cluster labels."""
    matrix = np.asarray(reduced, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("At least two reduced dimensions are required")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(matrix[:, 0], matrix[:, 1], c=np.asarray(labels), s=18, alpha=0.75)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label=_embedding_label(title, label_name))
    return _finish(fig, path)


def plot_embedding_3d(
    reduced: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    *,
    title: str,
    label_name: str | None = None,
) -> Path:
    """Plot the first three reduced dimensions colored by target or cluster labels."""
    matrix = np.asarray(reduced, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError("At least three reduced dimensions are required")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        matrix[:, 0], matrix[:, 1], matrix[:, 2], c=np.asarray(labels), s=16, alpha=0.7
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title(title)
    fig.colorbar(
        scatter,
        ax=ax,
        label=_embedding_label(title, label_name),
        shrink=0.7,
    )
    return _finish(fig, path)


def plot_component_loadings(
    components: np.ndarray,
    feature_names: Sequence[str],
    path: str | Path,
    *,
    max_components: int = 5,
) -> Path:
    """Plot a heatmap of signed component loadings."""
    matrix = np.asarray(components, dtype=float)[:max_components]
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("Components and feature names are inconsistent")
    labels = [f"C{index}" for index in range(1, len(matrix) + 1)]
    fig, ax = plt.subplots(figsize=(max(9, len(feature_names) * 0.8), max(4, len(matrix) * 0.8)))
    sns.heatmap(
        matrix,
        xticklabels=list(feature_names),
        yticklabels=labels,
        cmap="vlag",
        center=0.0,
        annot=False,
        ax=ax,
    )
    ax.set_title("Component loadings")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Component")
    return _finish(fig, path)


def plot_reconstruction_curve(
    component_counts: Sequence[int],
    pca_errors: Sequence[float],
    svd_errors: Sequence[float],
    path: str | Path,
) -> Path:
    """Compare reconstruction MSE as retained dimensionality increases."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pca_values = np.asarray(pca_errors, dtype=float)
    svd_values = np.asarray(svd_errors, dtype=float)
    overlap = pca_values.shape == svd_values.shape and np.allclose(
        pca_values, svd_values, rtol=1e-10, atol=1e-12
    )
    ax.plot(
        component_counts,
        pca_errors,
        linestyle="-",
        marker="o",
        linewidth=2.0,
        markersize=6,
        label="PCA",
    )
    ax.plot(
        component_counts,
        svd_errors,
        linestyle="None" if overlap else "--",
        marker="s",
        linewidth=2.0,
        markersize=6,
        label="SVD",
    )
    if overlap:
        ax.text(
            0.98,
            0.92,
            "PCA and SVD curves overlap within numerical precision",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Reconstruction error vs dimensionality")
    ax.grid(alpha=0.25)
    ax.legend()
    return _finish(fig, path)
