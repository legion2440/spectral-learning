"""Low-rank SVD denoising through a Hankel trajectory matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.artifacts import write_json


def generate_signal(
    *,
    samples: int = 1000,
    noise_std: float = 0.45,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a deterministic two-frequency signal and additive Gaussian noise."""
    if samples < 50:
        raise ValueError("samples must be at least 50")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    t = np.linspace(0.0, 4.0, samples, endpoint=False)
    clean = np.sin(2 * np.pi * 2.0 * t) + 0.45 * np.sin(2 * np.pi * 7.0 * t + 0.3)
    rng = np.random.default_rng(random_state)
    noisy = clean + rng.normal(0.0, noise_std, size=samples)
    return t, clean, noisy


def denoise_signal_svd(signal: np.ndarray, *, rank: int = 2, window: int | None = None) -> np.ndarray:
    """Denoise a 1D signal using low-rank approximation of its Hankel matrix."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite 1D vector with at least four samples")
    window_size = min(len(values) // 2, 200) if window is None else int(window)
    if not 2 <= window_size < len(values):
        raise ValueError("window must be between 2 and len(signal)-1")
    columns = len(values) - window_size + 1
    max_rank = min(window_size, columns)
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank must be between 1 and {max_rank}")

    trajectory = np.column_stack(
        [values[index : index + window_size] for index in range(columns)]
    )
    u, singular_values, vt = np.linalg.svd(trajectory, full_matrices=False)
    low_rank = (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]

    reconstructed = np.zeros(len(values), dtype=float)
    counts = np.zeros(len(values), dtype=float)
    for column in range(columns):
        reconstructed[column : column + window_size] += low_rank[:, column]
        counts[column : column + window_size] += 1.0
    return reconstructed / counts


def _snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    noise = reference - estimate
    numerator = float(np.sum(reference**2))
    denominator = float(np.sum(noise**2))
    return float("inf") if denominator == 0 else float(10.0 * np.log10(numerator / denominator))


def run_signal_denoising(
    output_dir: str | Path,
    *,
    rank: int = 4,
    samples: int = 1000,
    noise_std: float = 0.45,
    random_state: int = 42,
) -> Path:
    """Generate, denoise, evaluate and plot the synthetic signal experiment."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    t, clean, noisy = generate_signal(
        samples=samples, noise_std=noise_std, random_state=random_state
    )
    denoised = denoise_signal_svd(noisy, rank=rank)
    metrics = {
        "rank": rank,
        "samples": samples,
        "noise_std": noise_std,
        "noisy_mse": float(np.mean((clean - noisy) ** 2)),
        "denoised_mse": float(np.mean((clean - denoised) ** 2)),
        "noisy_snr_db": _snr(clean, noisy),
        "denoised_snr_db": _snr(clean, denoised),
    }
    write_json(output / "metrics.json", metrics)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, noisy, alpha=0.35, label="Noisy")
    ax.plot(t, clean, linewidth=1.5, label="Clean")
    ax.plot(t, denoised, linewidth=1.5, label="SVD denoised")
    ax.set_title("Low-rank SVD signal denoising")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "signal_denoising.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output
