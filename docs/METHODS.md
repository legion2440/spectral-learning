# Methods

## 1. Problem definition

Dimensionality reduction replaces a feature vector in a high-dimensional space with a lower-dimensional representation that preserves as much useful structure as possible. In this repository the retained structure is primarily measured through variance, reconstruction quality and cluster separability.

The production path works on a numerical matrix `X ∈ R^(n×d)`. Preprocessing is fitted on a training subset and stores feature medians, means and standard deviations. Future rows are transformed with exactly those stored values.

## 2. Standardization

For feature `j`, missing values are first replaced with the training median. The standardized value is then

```text
z_ij = (x_ij - μ_j) / σ_j
```

where `μ_j` and `σ_j` are calculated from training rows only. A zero-variance feature receives an operational scale of `1` so transformation remains finite; it is also reported as a constant feature.

Separating fit and transform prevents data leakage. The feature schema is persisted so a future CSV cannot silently reorder or omit required model inputs.

## 3. PCA from covariance eigendecomposition

Given the preprocessed matrix, PCA explicitly performs mean centering:

```text
Xc = X - mean(X)
```

The sample covariance matrix is

```text
C = Xcᵀ Xc / (n - 1)
```

The implementation then calls only the allowed NumPy eigendecomposition primitive:

```text
λ, V = numpy.linalg.eig(C)
```

Project code performs the rest of the PCA algorithm:

1. discard negligible complex numerical residue only when it is within tolerance;
2. clip tiny negative eigenvalues caused by floating-point rounding;
3. sort eigenvalues in descending order;
4. reorder eigenvectors with the same permutation;
5. calculate explained-variance ratios;
6. select top-k directions;
7. project centered samples:

```text
Z = Xc Vk
```

The inverse approximation is

```text
X_hat = Z Vkᵀ + mean(X)
```

No `sklearn.decomposition.PCA` object is involved.

## 4. SVD reduction

The SVD path centers the matrix in the same way and obtains

```text
Xc = U Σ Vᵀ
```

through `numpy.linalg.svd`.

The project explicitly chooses the first `k` right-singular vectors and projects samples into that latent space:

```text
Z = Xc Vk
```

The low-rank reconstruction is

```text
X_hat = Z Vkᵀ + mean(X)
```

For a centered matrix, the covariance matrix can be written as

```text
C = V (Σ² / (n - 1)) Vᵀ
```

so PCA eigenvectors and SVD right-singular vectors span the same principal directions. The corresponding variance values are

```text
variance_i = σ_i² / (n - 1)
```

With the same centered input, the two routes are therefore expected to produce the same cumulative explained-variance and optimal low-rank reconstruction curves up to floating-point precision. The comparison plots intentionally keep both series visible with different line styles and annotate numerical overlap instead of treating it as a missing result.

Individual component vectors may still appear with opposite signs. Eigenvectors and singular vectors are defined only up to multiplication by `-1`, so such sign flips represent the same axis and do not change the subspace, explained variance or reconstruction.

## 5. Explained variance and choosing k

For PCA, each eigenvalue represents variance along one principal direction. For SVD, `σ_i²/(n-1)` has the same interpretation after centering.

The per-component ratio is

```text
r_i = variance_i / sum(variance)
```

and cumulative retained variance is

```text
R_k = sum(r_1 ... r_k)
```

The CLI supports two policies:

- explicit `--components k`;
- automatic `--variance-threshold t`.

Automatic selection chooses the smallest `k` with `R_k >= t`. This makes dimensionality selection traceable instead of picking a two-dimensional representation only because it is easy to plot.

A high variance threshold can legitimately retain most of the original features. That outcome should be reported as evidence that the dataset does not support aggressive linear compression at the requested information-retention level, rather than forcing a smaller `k` for presentation purposes.

## 6. Reconstruction metrics

Two reconstruction metrics are saved.

Mean squared error:

```text
MSE = mean((X - X_hat)²)
```

Relative Frobenius error:

```text
||X - X_hat||_F / ||X||_F
```

The comparison workflow also recomputes PCA and SVD across the valid component range and plots reconstruction MSE against dimensionality. Reconstruction error should generally decrease as more directions are retained.

## 7. Clustering evaluation

K-means is evaluated on:

1. the original standardized feature matrix;
2. PCA coordinates;
3. SVD coordinates.

Internal metrics do not require known labels:

- **Inertia** — within-cluster squared distance used by K-means itself;
- **Silhouette Score** — compares within-cluster cohesion against the nearest alternative cluster;
- **Calinski-Harabasz Score** — ratio of between-cluster to within-cluster dispersion;
- **Davies-Bouldin Score** — similarity of each cluster to its most similar neighbor; lower is better.

When a reference target such as Wine Quality `quality` is available, two permutation-invariant external scores are added:

- **Adjusted Rand Index (ARI)**;
- **Normalized Mutual Information (NMI)**.

Raw cluster IDs are arbitrary, so simple equality-based classification accuracy would be misleading without an explicit cluster-to-class matching procedure.

## 8. Component interpretation

Each PCA or SVD component is a vector of signed feature loadings. The repository sorts features by absolute loading magnitude and stores the strongest signed values.

A large positive loading means the component increases with that standardized feature; a large negative loading means it moves in the opposite direction. Component sign itself is arbitrary: multiplying an eigenvector or singular vector by `-1` represents the same subspace. Interpretation should therefore focus on relative feature relationships rather than the absolute orientation of the sign.

## 9. Train/inference separation and overfitting

PCA and SVD are unsupervised decompositions, so supervised overfitting and regularization do not map directly onto them. The relevant controls are instead:

- fit imputation/scaling on training rows only;
- fit components on training rows only;
- use retained variance and reconstruction error to avoid unnecessary dimensions;
- inspect cluster quality after reduction;
- persist the selected transform and reuse it for future rows rather than refitting on every batch;
- keep a deterministic random seed for train-subset and clustering comparisons.

An excessively small `k` causes information loss. An unnecessarily large `k` retains noise and weakens the point of dimensionality reduction. The variance threshold and reconstruction curve expose this trade-off explicitly.

## 10. Exact t-SNE experiment

The optional t-SNE implementation uses NumPy only. It computes pairwise squared distances, finds Gaussian neighborhood bandwidths by binary search to match the requested perplexity, symmetrizes high-dimensional probabilities, and optimizes a low-dimensional Student-t distribution with gradient descent.

This path is intentionally limited to a bounded sample because exact t-SNE requires O(n²) memory/work. It is for visualization, not for the persisted out-of-sample transformation contract used by PCA/SVD.
