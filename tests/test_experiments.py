import numpy as np

from experiments.image_compression import compress_image_svd
from experiments.signal_denoising import denoise_signal_svd, generate_signal


def test_image_reconstruction_shape() -> None:
    rng = np.random.default_rng(5)
    image = rng.random((20, 16, 3))
    reconstructed, metrics = compress_image_svd(image, rank=5)
    assert reconstructed.shape == image.shape
    assert metrics["mse"] >= 0.0


def test_signal_denoising_shape_and_finiteness() -> None:
    _, _, noisy = generate_signal(samples=120, noise_std=0.2, random_state=5)
    denoised = denoise_signal_svd(noisy, rank=2, window=30)
    assert denoised.shape == noisy.shape
    assert np.isfinite(denoised).all()
