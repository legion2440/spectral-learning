"""Command-line entry point for the spectral-learning toolkit."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.image_compression import run_image_compression
from experiments.signal_denoising import run_signal_denoising
from workflows import (
    compare_workflow,
    nonlinear_workflow,
    train_workflow,
    transform_workflow,
    umap_workflow,
)


def _target(value: str) -> str | None:
    return None if value.lower() in {"none", "null", "-"} else value


def _ranks(value: str) -> list[int]:
    try:
        ranks = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from exc
    if not ranks or any(rank < 1 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must contain positive integers")
    return ranks


def _add_reduction_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input CSV dataset")
    parser.add_argument("--target", default="quality", type=_target, help="Target column or 'none'")
    parser.add_argument("--components", type=int, default=None, help="Explicit retained dimension count")
    parser.add_argument("--variance-threshold", type=float, default=0.95, help="Automatic variance target")
    parser.add_argument("--clusters", type=int, default=6, help="K-means cluster count")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction used to fit preprocessing/reducer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="artifacts", help="Artifact root directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PCA/SVD dimensionality reduction with reusable train/inference artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Fit and evaluate one reducer")
    _add_reduction_options(train)
    train.add_argument("--method", choices=("pca", "svd"), required=True)

    compare = subparsers.add_parser("compare", help="Compare PCA and SVD on the same data")
    _add_reduction_options(compare)

    transform = subparsers.add_parser("transform", help="Transform new data with saved artifacts")
    transform.add_argument("--input", required=True)
    transform.add_argument("--model", required=True)
    transform.add_argument("--preprocessor", required=True)
    transform.add_argument("--output", required=True)

    nonlinear = subparsers.add_parser("nonlinear", help="Run optional exact NumPy t-SNE")
    nonlinear.add_argument("--input", required=True)
    nonlinear.add_argument("--target", default="quality", type=_target)
    nonlinear.add_argument("--output", default="artifacts")
    nonlinear.add_argument("--max-samples", type=int, default=1000)
    nonlinear.add_argument("--perplexity", type=float, default=30.0)
    nonlinear.add_argument("--iterations", type=int, default=750)
    nonlinear.add_argument("--seed", type=int, default=42)

    umap = subparsers.add_parser("umap", help="Run optional UMAP visualization")
    umap.add_argument("--input", required=True)
    umap.add_argument("--target", default="quality", type=_target)
    umap.add_argument("--output", default="artifacts")
    umap.add_argument("--max-samples", type=int, default=5000)
    umap.add_argument("--neighbors", type=int, default=15)
    umap.add_argument("--min-dist", type=float, default=0.1)
    umap.add_argument("--seed", type=int, default=42)

    image = subparsers.add_parser("bonus-image", help="Run SVD image compression")
    image.add_argument("--input", required=True)
    image.add_argument("--ranks", type=_ranks, default=[5, 20, 50, 100])
    image.add_argument("--output", default="artifacts/image_compression")

    signal = subparsers.add_parser("bonus-signal", help="Run SVD signal denoising")
    signal.add_argument("--output", default="artifacts/signal_denoising")
    signal.add_argument("--rank", type=int, default=4)
    signal.add_argument("--samples", type=int, default=1000)
    signal.add_argument("--noise-std", type=float, default=0.45)
    signal.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        result = train_workflow(
            args.input,
            method=args.method,
            target=args.target,
            output_root=args.output,
            n_components=args.components,
            variance_threshold=args.variance_threshold,
            n_clusters=args.clusters,
            train_fraction=args.train_fraction,
            random_state=args.seed,
        )
    elif args.command == "compare":
        result = compare_workflow(
            args.input,
            target=args.target,
            output_root=args.output,
            n_components=args.components,
            variance_threshold=args.variance_threshold,
            n_clusters=args.clusters,
            train_fraction=args.train_fraction,
            random_state=args.seed,
        )
    elif args.command == "transform":
        result = transform_workflow(
            args.input,
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            output_path=args.output,
        )
    elif args.command == "nonlinear":
        result = nonlinear_workflow(
            args.input,
            target=args.target,
            output_root=args.output,
            max_samples=args.max_samples,
            perplexity=args.perplexity,
            iterations=args.iterations,
            random_state=args.seed,
        )
    elif args.command == "umap":
        result = umap_workflow(
            args.input,
            target=args.target,
            output_root=args.output,
            max_samples=args.max_samples,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            random_state=args.seed,
        )
    elif args.command == "bonus-image":
        result = run_image_compression(args.input, args.ranks, args.output)
    elif args.command == "bonus-signal":
        result = run_signal_denoising(
            args.output,
            rank=args.rank,
            samples=args.samples,
            noise_std=args.noise_std,
            random_state=args.seed,
        )
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")
    print(Path(result))


if __name__ == "__main__":
    main()
