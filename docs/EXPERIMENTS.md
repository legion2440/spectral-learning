# Experiments and interpretation

This document describes what the generated artifacts mean. It intentionally does not contain fixed metric values: the repository writes measurements from the local dataset and environment into each run directory.

## 1. PCA vs SVD comparison

Run:

```bash
python main.py compare --input data/raw/wine_quality.csv --variance-threshold 0.95
```

Inspect these artifacts together:

- `metrics.json` — selected dimensions, retained variance, train/test reconstruction, clustering and component-alignment diagnostics;
- `variance_comparison.png` — cumulative explained variance;
- `reconstruction_curve.png` — train and held-out reconstruction MSE as dimensions are added;
- `reconstruction_curve.json` — numeric values behind that curve;
- `quality_clustering_sweep.png` and `quality_clustering_sweep.json` — clustering stability across `k=2..10` when a target exists;
- `wine_type_separability.json` — original-vs-2D red/white separability when `wine_type` exists;
- `pca_2d.png`, `svd_2d.png` — two-dimensional views;
- `pca_3d.png`, `svd_3d.png` — three-dimensional views when enough dimensions are retained;
- `pca_loadings.png`, `svd_loadings.png` — signed feature contribution heatmaps;
- `component_interpretation.json` — strongest loadings in machine-readable form.

For centered data, PCA and SVD should identify the same principal subspace up to floating-point effects and possible rotations inside degenerate eigenspaces. The code canonicalizes the ordinary sign ambiguity so directly corresponding directions are easier to compare. The comparison remains useful because the two code paths expose the eigenvalue and singular-value formulations separately.

## 2. Cluster views and reference labels

The workflow evaluates K-means before and after dimensionality reduction. Compare internal scores rather than assuming reduction must always improve clustering.

When `quality` exists, it is used as the color/reference target for the main embedding and as the reference labels for ARI/NMI. The PCA/SVD fit never receives `quality` as a feature.

The comparison also sweeps K-means cluster count from 2 through 10, bounded by sample count, for the original/PCA/SVD representations. This is a robustness check, not a model-selection loop: the workflow reports the complete range instead of selecting whichever `k` makes ARI look best.

Wine `quality` is ordinal, while ARI/NMI treat labels as nominal categories. Therefore these scores measure recovery of the exact quality partition and do not give partial credit for grouping adjacent quality levels.

If the combined Wine Quality dataset contains `wine_type`, the comparison adds a separate red/white task. It reports class counts after cleaning and evaluates the original standardized representation against explicit train-fitted PCA 2D and SVD 2D representations. The intended interpretation is whether red/white separability is preserved after deliberate compression to two dimensions, not that a high wine-type ARI by itself proves PCA/SVD quality.

The existing per-type cluster plots remain useful robustness views: they show structure inside red and white subsets separately.

## 3. Choosing dimensionality

The default policy retains the smallest number of components reaching 95% cumulative variance. Useful comparison points are 80%, 90%, 95% and 99%.

A lower threshold gives a smaller representation and usually larger reconstruction error. A higher threshold preserves more information but reduces the compression benefit. If 95% keeps most of the original dimensions, report that as limited linear compressibility at the chosen information-retention level; do not lower the threshold after observing the result merely to obtain a more dramatic reduction.

The 95% rule and an explicit `k=2`/`k=3` experiment answer different questions. The former is a reconstruction/information-retention objective. The latter is appropriate for visualization or for asking whether useful downstream structure survives in a deliberately compact space. Keeping those objectives separate avoids post-hoc threshold tuning.

## 4. Held-out reconstruction

With the default `--train-fraction 0.8`, preprocessing and spectral directions are fitted on the deterministic training subset only. The selected reducer is then evaluated separately on training and held-out rows.

`metrics.json` records:

- overall reconstruction MSE and relative Frobenius error;
- train reconstruction metrics;
- test reconstruction metrics;
- `test_minus_train_mse`;
- test/train MSE ratio.

`reconstruction_curve.json` and `reconstruction_curve.png` use one full train-fitted basis per method and evaluate nested component prefixes on both train and test. The workflow does not fit a new PCA/SVD model for each plotted `k`.

For a fixed orthonormal basis, train and test reconstruction error should both decrease as more directions are retained. There is no expected supervised-style U-shaped validation curve. The useful diagnostic is the distance between train and test curves: a small gap indicates that the train-estimated subspace reconstructs unseen rows similarly; a widening gap indicates sample-specific directions.

When `--train-fraction 1.0` is used, there is no held-out set and test diagnostics are recorded as `null`/omitted from the plot.

## 5. Nonlinear visualization

### Exact NumPy t-SNE

Run:

```bash
python main.py nonlinear \
  --input data/raw/wine_quality.csv \
  --max-samples 1000 \
  --perplexity 30 \
  --iterations 750
```

The exact NumPy t-SNE implementation is deliberately separated from the persisted reducers:

- it is stochastic but reproducible under the configured seed;
- pairwise work scales quadratically;
- distances in the final 2D plot are local-neighborhood oriented and should not be read like PCA axes;
- there is no supported `transform(new_rows)` contract.

Use it to inspect nonlinear neighborhoods, not to replace the production PCA/SVD transform pipeline.

### UMAP

UMAP is an optional external dependency so the core PCA/SVD environment remains lightweight. Install it with:

```bash
python -m pip install -r requirements-extra.txt
```

Then run:

```bash
python main.py umap \
  --input data/raw/wine_quality.csv \
  --max-samples 5000 \
  --neighbors 15 \
  --min-dist 0.1
```

The UMAP workflow writes a two-dimensional embedding, its parameters and the deterministic sample configuration. Like t-SNE, this experiment is used for nonlinear visualization and comparison rather than as the persisted production reducer in this repository.

## 6. Image compression

Run:

```bash
python main.py bonus-image --input path/to/image.png --ranks 5,20,50,100
```

For each image channel, SVD factorizes the pixel matrix and retains only the requested rank. Increasing rank generally improves reconstruction while storing more singular directions.

Generated `metrics.json` contains:

- rank;
- MSE;
- PSNR;
- an estimated ratio between raw pixel values and low-rank matrix parameters.

The estimate intentionally ignores image-codec headers, quantization and entropy coding. It demonstrates matrix compression, not a replacement for PNG/JPEG/WebP codecs.

## 7. Signal denoising

Run:

```bash
python main.py bonus-signal --rank 4 --samples 1000 --noise-std 0.45
```

The experiment creates a deterministic two-frequency signal plus Gaussian noise. A sliding window turns the 1D sequence into a Hankel trajectory matrix. SVD keeps a low-rank approximation and diagonal averaging maps the matrix back into a 1D series.

Each real sinusoidal component contributes a two-dimensional Hankel subspace (sine and cosine phase directions), so the default two-frequency synthetic signal uses rank 4. Lower rank can remove genuine signal structure; higher rank progressively admits noise directions.

The output compares:

- noisy MSE vs the clean reference;
- denoised MSE vs the clean reference;
- noisy SNR;
- denoised SNR.

Rank controls the bias/noise trade-off. Too little rank removes real signal structure; too much rank begins to reconstruct noise.

## 8. Reproducibility checklist

For a comparable local rerun, keep fixed:

- input dataset revision;
- Python/dependency versions;
- `--seed`;
- `--train-fraction`;
- `--variance-threshold` or `--components`;
- requested main K-means cluster count;
- the documented diagnostic sweep bounds;
- t-SNE sample size/perplexity/iterations for nonlinear experiments;
- UMAP sample size/neighbors/min-dist for nonlinear experiments.

The generated JSON files are the source of truth for the actual local measurements.
