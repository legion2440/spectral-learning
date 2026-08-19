"""A compact exact t-SNE implementation for optional nonlinear exploration."""

from __future__ import annotations

import numpy as np

from utils.matrix_operations import as_float_matrix


class TSNEFromScratch:
    """Exact O(n²) t-SNE using NumPy only.

    This implementation is intentionally aimed at small exploratory samples. It is not
    used by the PCA/SVD production transform path and does not provide out-of-sample
    transformation.
    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 750,
        random_state: int = 42,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if perplexity <= 0:
            raise ValueError("perplexity must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iter < 250:
            raise ValueError("n_iter must be at least 250")
        self.n_components = n_components
        self.perplexity = float(perplexity)
        self.learning_rate = float(learning_rate)
        self.n_iter = int(n_iter)
        self.random_state = int(random_state)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Compute an exploratory t-SNE embedding for a finite matrix."""
        matrix = as_float_matrix(X)
        n_samples = matrix.shape[0]
        if self.perplexity >= n_samples:
            raise ValueError("perplexity must be smaller than the sample count")

        probabilities = self._joint_probabilities(matrix)
        rng = np.random.default_rng(self.random_state)
        embedding = rng.normal(0.0, 1e-4, size=(n_samples, self.n_components))
        velocity = np.zeros_like(embedding)

        exaggeration_steps = min(250, self.n_iter // 3)
        for iteration in range(self.n_iter):
            differences = embedding[:, None, :] - embedding[None, :, :]
            distances = np.sum(differences**2, axis=2)
            student = 1.0 / (1.0 + distances)
            np.fill_diagonal(student, 0.0)
            q = student / np.clip(student.sum(), 1e-12, None)
            q = np.clip(q, 1e-12, None)

            p = probabilities * 12.0 if iteration < exaggeration_steps else probabilities
            weights = (p - q) * student
            gradient = 4.0 * np.sum(weights[:, :, None] * differences, axis=1)

            momentum = 0.5 if iteration < exaggeration_steps else 0.8
            velocity = momentum * velocity - self.learning_rate * gradient
            embedding += velocity
            embedding -= embedding.mean(axis=0, keepdims=True)

        self.embedding_ = embedding
        self.kl_divergence_ = self._kl_divergence(probabilities, embedding)
        return embedding.copy()

    def _joint_probabilities(self, matrix: np.ndarray) -> np.ndarray:
        squared_norms = np.sum(matrix**2, axis=1)
        distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * matrix @ matrix.T
        distances = np.maximum(distances, 0.0)

        n_samples = len(matrix)
        conditional = np.zeros((n_samples, n_samples), dtype=float)
        target_entropy = np.log(self.perplexity)

        for index in range(n_samples):
            mask = np.ones(n_samples, dtype=bool)
            mask[index] = False
            row = distances[index, mask]
            beta = 1.0
            beta_min = -np.inf
            beta_max = np.inf

            for _ in range(60):
                weights = np.exp(-row * beta)
                total = float(weights.sum())
                if total <= 1e-300:
                    weights = np.full_like(row, 1.0 / len(row))
                    entropy = np.log(len(row))
                else:
                    probabilities = weights / total
                    entropy = np.log(total) + beta * float(np.sum(row * weights)) / total
                    weights = probabilities

                difference = entropy - target_entropy
                if abs(difference) < 1e-5:
                    break
                if difference > 0:
                    beta_min = beta
                    beta = beta * 2.0 if np.isinf(beta_max) else (beta + beta_max) / 2.0
                else:
                    beta_max = beta
                    beta = beta / 2.0 if np.isinf(beta_min) else (beta + beta_min) / 2.0

            conditional[index, mask] = weights

        joint = (conditional + conditional.T) / (2.0 * n_samples)
        joint = np.clip(joint, 1e-12, None)
        joint /= joint.sum()
        return joint

    @staticmethod
    def _kl_divergence(probabilities: np.ndarray, embedding: np.ndarray) -> float:
        differences = embedding[:, None, :] - embedding[None, :, :]
        distances = np.sum(differences**2, axis=2)
        student = 1.0 / (1.0 + distances)
        np.fill_diagonal(student, 0.0)
        q = np.clip(student / np.clip(student.sum(), 1e-12, None), 1e-12, None)
        p = np.clip(probabilities, 1e-12, None)
        return float(np.sum(p * np.log(p / q)))
