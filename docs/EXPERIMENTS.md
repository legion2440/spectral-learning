# Experiments and interpretation

This document describes what the generated artifacts mean. It intentionally does not contain fixed metric values: the repository writes measurements from the local dataset and environment into each run directory.

## 1. PCA vs SVD comparison

Run:

```bash
python main.py compare --input data/raw/wine_quality.csv --variance-threshold 0.95
```

Inspect these artifacts together:

- `metrics.json` — selected dimensions, retained variance, reconstruction and clustering scores;
- `variance_comparison.png` — cumulative explained variance;
- `reconstruction_curve.png` — reconstruction MSE as dimensions are added;
- `pca_2d.png`, `svd_2d.png` — two-dimensional views;
- `pca_3d.png`, `svd_3d.png` — three-dimensional views when enough dimensions are retained;
- `pca_loadings.png`, `svd_loadings.png` — signed feature contribution heatmaps;
- `component_interpretation.json` — strongest loadings in machine-readable form.

For centered data, PCA and SVD should identify the same principal subspace up to sign/numerical differences. The comparison is still useful because the two code paths expose the eigenvalue and singular-value formulations separately.

## 2. Cluster views

The workflow evaluates K-means before and after dimensionality reduction. Compare internal scores rather than assuming reduction must always improve clustering.

If the combined Wine Quality dataset contains `wine_type`, the comparison also writes separate red-wine and white-wine cluster plots. These views are robustness examples: they show whether the same reduced space remains structurally useful across meaningful subsets rather than only on the aggregate dataset.

When `quality` exists it is used as the color/reference target for the main embedding and as the reference labels for ARI/NMI. The PCA/SVD fit never receives `quality` as a feature.

## 3. Choosing dimensionality

The default policy retains the smallest number of components reaching 95% cumulative variance. Useful comparison points are 80%, 90%, 95% and 99%.

A lower threshold gives a smaller representation and usually larger reconstruction error. A higher threshold preserves more information but reduces the compression benefit. The reconstruction curve helps determine whether later components provide material improvement.

## 4. Nonlinear visualization

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

## 5. Image compression

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

## 6. Signal denoising

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

## 7. Reproducibility checklist

For a comparable local rerun, keep fixed:

- input dataset revision;
- Python/dependency versions;
- `--seed`;
- `--train-fraction`;
- `--variance-threshold` or `--components`;
- K-means cluster count;
- t-SNE sample size/perplexity/iterations for nonlinear experiments;
- UMAP sample size/neighbors/min-dist for nonlinear experiments.

The generated JSON files are the source of truth for the actual local measurements.
