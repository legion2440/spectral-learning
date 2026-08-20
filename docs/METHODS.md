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

`TabularPreprocessor` uses the population standard deviation (`ddof=0`), while PCA later forms the sample covariance with denominator `n-1`. Therefore a standardized non-constant feature has sample variance `n/(n-1)`, which approaches 1 for large `n`. This is why the standardized covariance behaves like a correlation matrix without being numerically identical to the textbook `ddof=1` construction.

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
7. canonicalize each component sign for deterministic presentation;
8. project centered samples:

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

so PCA eigenvectors and SVD right-singular vectors span the same principal subspace. The corresponding variance values are

```text
variance_i = σ_i² / (n - 1)
```

With the same centered input, the two routes are therefore expected to produce the same cumulative explained-variance and optimal low-rank reconstruction curves up to floating-point precision.

### Deterministic sign convention

Individual eigenvectors and singular vectors are mathematically defined only up to multiplication by `-1`. To avoid arbitrary mirrored embeddings and opposite loading colors, both implementations apply the same presentation convention: the loading with largest absolute magnitude in each component is oriented to be non-negative.

This convention does not change the represented subspace, explained variance, distances or reconstruction. It only chooses one of the two equivalent signs. It also does not solve the separate degeneracy case: when eigenvalues are equal or nearly equal, valid algorithms may rotate the basis inside the same principal subspace. For that reason production correctness is defined by the subspace/projection, not by requiring arbitrary component vectors to match for every possible dataset.

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

Automatic selection chooses the smallest `k` with `R_k >= t`. The default 95% threshold is a **reconstruction/information-retention objective**. It is deliberately not tuned to make a two-dimensional plot look cleaner or to maximize a downstream clustering score.

A high variance threshold can legitimately retain most of the original features. On standardized data, weakly correlated features tend to spread variance across many directions, so cumulative explained variance may grow slowly. That outcome is evidence of limited linear compressibility at the requested information-retention level, not a reason to lower the threshold after seeing the result.

A Kaiser-style `λ > 1` rule can be discussed as a diagnostic intuition for correlation-like matrices: a component above that scale explains roughly more variance than one standardized source feature. Because this project uses `ddof=0` for scaling and `n-1` for sample covariance, the exact one-feature sample-variance reference is `n/(n-1)`, not exactly `1`. The project therefore does not use Kaiser as its component selector.

For visualization or a dedicated clustering experiment, an explicit `k=2` or `k=3` answers a different question: how much useful structure remains in a deliberately compact representation. Reporting that alongside the 95% reconstruction choice is not cherry-picking as long as the objectives remain clearly separated.

## 6. Reconstruction and held-out generalization

Two basic reconstruction metrics are saved.

Mean squared error:

```text
MSE = mean((X - X_hat)²)
```

Relative Frobenius error:

```text
||X - X_hat||_F / ||X||_F
```

The selected PCA/SVD model is fitted only on the configured training rows. `metrics.json` then reports reconstruction separately for train and held-out test rows, together with `test_minus_train_mse` and a test/train MSE ratio when a held-out split exists.

The reconstruction curve is also built from a **single full train-fitted orthonormal basis per method**. For `k=1..p`, the workflow takes nested prefixes of that same basis and evaluates them on both train and test rows. It does not refit a different model for every point on the curve.

For a fixed orthonormal basis, adding another retained direction cannot increase squared projection residual, so both train and test reconstruction curves should be monotone non-increasing apart from negligible floating-point noise. Consequently this is not a supervised learning curve with an expected U-shaped minimum. The useful overfitting diagnostic is the **train/test gap**: a small gap means the train-fitted subspace generalizes similarly to unseen rows; a widening gap indicates that the estimated directions are becoming specific to the fitted sample.

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

### Quality k-sweep

The default CLI still exposes one requested `--clusters` value for the main clustering comparison. In addition, the comparison workflow performs a diagnostic sweep over K-means `k=2..10` (bounded by sample count) on the original, PCA and SVD representations. The sweep is **not** used to select a post-hoc best `k`; it tests whether conclusions drawn from the default clustering count are stable across a reasonable range.

Wine `quality` is ordinal, but ARI and NMI treat its values as nominal categories. They therefore do not distinguish a near miss such as grouping qualities 5 and 6 from a much wider ordinal mismatch such as grouping 3 and 9. Low ARI/NMI against `quality` should be interpreted as weak recovery of the exact category partition, not as a complete statement about ordinal predictive structure.

### Wine-type separability

When the combined UCI dataset contains `wine_type`, the workflow adds a second, distinct reference task. It reports cleaned red/white class counts and evaluates K-means with the natural class count on:

- the original standardized feature space;
- an explicitly train-fitted PCA 2D representation;
- an explicitly train-fitted SVD 2D representation.

The useful claim is comparative: whether strong red/white separability is preserved after an intentional high-dimensional-to-2D reduction. A high wine-type ARI primarily reflects real structure in the data; comparing original versus 2D indicates how much of that structure the reduction preserves.

## 8. Component interpretation

Each PCA or SVD component is a vector of signed feature loadings. The repository sorts features by absolute loading magnitude and stores the strongest signed values.

A large positive loading means the component increases with that standardized feature; a large negative loading means it moves in the opposite direction. The underlying mathematical sign remains arbitrary, but the deterministic sign convention makes equivalent PCA/SVD directions easier to compare visually and across persisted runs.

The comparison metadata also records paired PCA/SVD component cosine similarities. Values near `1` indicate that the canonicalized component vectors align directly. A lower value does not automatically imply a wrong subspace in a degenerate spectrum; projection/reconstruction equivalence remains the stronger invariant.

## 9. Train/inference separation and overfitting

PCA and SVD are unsupervised decompositions, so supervised overfitting and regularization do not map directly onto them. The relevant controls are instead:

- fit imputation/scaling on training rows only;
- fit components on training rows only;
- evaluate reconstruction on held-out rows without refitting;
- inspect the train/test reconstruction gap across nested component prefixes;
- use retained variance and reconstruction error to avoid unnecessary dimensions;
- inspect cluster quality after reduction and across a bounded clustering sweep;
- persist the selected transform and reuse it for future rows rather than refitting on every batch;
- keep a deterministic random seed for train-subset and clustering comparisons.

An excessively small `k` causes information loss. An unnecessarily large `k` weakens the point of dimensionality reduction and can retain sample-specific directions. The variance threshold, held-out reconstruction curve and generalization gap expose these trade-offs without importing a supervised notion of a validation-loss minimum where it does not apply.

## 10. Exact t-SNE experiment

The optional t-SNE implementation uses NumPy only. It computes pairwise squared distances, finds Gaussian neighborhood bandwidths by binary search to match the requested perplexity, symmetrizes high-dimensional probabilities, and optimizes a low-dimensional Student-t distribution with gradient descent.

This path is intentionally limited to a bounded sample because exact t-SNE requires O(n²) memory/work. It is for visualization, not for the persisted out-of-sample transformation contract used by PCA/SVD.
