"""SVD image-compression experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from utils.artifacts import write_json


def compress_image_svd(image: np.ndarray, rank: int) -> tuple[np.ndarray, dict[str, float]]:
    """Compress each image channel with a rank-k SVD reconstruction."""
    array = np.asarray(image, dtype=float)
    if array.ndim not in (2, 3):
        raise ValueError("Image must be grayscale or channel-last RGB/RGBA")
    height, width = array.shape[:2]
    max_rank = min(height, width)
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank must be between 1 and {max_rank}")

    channels = 1 if array.ndim == 2 else array.shape[2]
    planes = [array] if array.ndim == 2 else [array[:, :, index] for index in range(channels)]
    reconstructed_planes: list[np.ndarray] = []
    for plane in planes:
        u, singular_values, vt = np.linalg.svd(plane, full_matrices=False)
        reconstructed_planes.append(
            (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]
        )

    reconstructed = (
        reconstructed_planes[0]
        if array.ndim == 2
        else np.stack(reconstructed_planes, axis=2)
    )
    data_min = float(np.nanmin(array))
    data_max = float(np.nanmax(array))
    reconstructed = np.clip(reconstructed, data_min, data_max)
    mse = float(np.mean((array - reconstructed) ** 2))
    data_range = data_max - data_min
    psnr = float("inf") if mse == 0 else float(20.0 * np.log10(data_range / np.sqrt(mse)))
    original_values = height * width * channels
    compressed_values = rank * (height + width + 1) * channels
    metrics = {
        "rank": float(rank),
        "mse": mse,
        "psnr_db": psnr,
        "storage_ratio_estimate": float(original_values / compressed_values),
    }
    return reconstructed, metrics


def run_image_compression(
    input_path: str | Path,
    ranks: Sequence[int],
    output_dir: str | Path,
) -> Path:
    """Run multiple SVD ranks and save reconstructed images plus a metric summary."""
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    image = mpimg.imread(source)
    results: list[dict[str, float]] = []
    for rank in ranks:
        reconstructed, metrics = compress_image_svd(image, int(rank))
        target = output / f"rank_{int(rank)}.png"
        plt.imsave(target, reconstructed, cmap="gray" if reconstructed.ndim == 2 else None)
        results.append(metrics)

    write_json(output / "metrics.json", {"input": str(source), "results": results})
    return output
