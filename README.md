# Spectral Learning

A production-oriented dimensionality-reduction toolkit built around **PCA and SVD implemented from first principles with NumPy building blocks**. The project covers tabular preprocessing, variance-based component selection, held-out reconstruction diagnostics, clustering evaluation, reusable train/inference artifacts, interpretable visualizations, nonlinear t-SNE/UMAP experiments, and spectral applications for images and signals.

The default dataset is the UCI Wine Quality collection, but the CLI accepts other numerical CSV datasets with the same workflow.

· [Русская версия](README_RU.md)

## 📋 TOC

- [🚀 Quick start](#-quick-start)
- [📝 About](#-about)
- [✨ Features](#-features)
- [🔄 Architecture](#-architecture)
- [🧠 PCA and SVD](#-pca-and-svd)
- [🍷 Dataset and preprocessing](#-dataset-and-preprocessing)
- [⚙️ CLI workflows](#️-cli-workflows)
- [📊 Evaluation and interpretation](#-evaluation-and-interpretation)
- [📦 Reusable artifacts](#-reusable-artifacts)
- [🧪 Optional experiments](#-optional-experiments)
- [✅ Tests](#-tests)
- [📁 Project structure](#-project-structure)
- [⚠️ Engineering notes](#️-engineering-notes)
- [🧑‍💻 Author](#-author)

## 🚀 Quick start

### Requirements

- Python `3.11+`
- internet access only for the optional dataset download step

### Clone and install

```bash
git clone https://01.tomorrow-school.ai/git/nyestaye/spectral-learning.git
cd spectral-learning

python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux / macOS / WSL:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Download the Wine Quality dataset

```bash
python scripts/download_wine_quality.py
```

The script downloads the official UCI archive and creates:

```text
data/raw/wine_quality.csv
```

### Compare PCA and SVD

```bash
python main.py compare \
  --input data/raw/wine_quality.csv \
  --variance-threshold 0.95
```

Each run creates a timestamped directory under `artifacts/runs/` with metrics, plots, model parameters and preprocessing state.

## 📝 About

The repository is structured as a reusable ML component rather than a single analysis notebook. PCA and SVD own their fitting and transformation state, preprocessing is fitted once and persisted separately, and the same artifacts can later transform new compatible CSV data without recalculating training statistics.

The core decomposition logic stays explicit:

- PCA performs mean centering, covariance construction, `numpy.linalg.eig`, eigenvalue sorting, component selection and projection in project code;
- SVD performs centering, `numpy.linalg.svd`, singular-direction selection, projection and reconstruction in project code;
- `sklearn.decomposition.PCA`, `TruncatedSVD` and SciPy SVD helpers are not used;
- scikit-learn is limited to K-means and evaluation metrics in the main PCA/SVD workflow.

The default path fits preprocessing and dimensionality reduction on a deterministic training subset, then transforms the complete dataset. Held-out rows are used for reconstruction generalization diagnostics without refitting. This demonstrates the same separation required when new production data arrives later.

## ✨ Features

### Spectral reduction

- PCA from explicit covariance/eigendecomposition steps;
- SVD reduction around NumPy's matrix decomposition primitive;
- explicit `fit`, `transform`, `fit_transform` and `inverse_transform` APIs;
- manual component count or automatic variance threshold;
- deterministic component-sign convention for stable loadings/embeddings;
- explained variance and cumulative explained variance;
- reconstruction MSE and relative Frobenius error;
- feature loadings for component interpretation.

### Data pipeline

- delimiter auto-detection for CSV-like inputs;
- duplicate source-header validation before pandas can rename them;
- clean handling of empty/malformed CSV parser failures;
- target validation;
- duplicate-row removal for training datasets;
- numeric feature discovery;
- NaN/Inf normalization;
- train-fitted median imputation;
- train-fitted standardization;
- constant-feature handling;
- strict inference schema validation.

### Evaluation

- K-means on the original standardized representation;
- K-means on PCA and SVD representations;
- inertia;
- Silhouette Score;
- Calinski-Harabasz Score;
- Davies-Bouldin Score;
- Adjusted Rand Index when reference labels are available;
- Normalized Mutual Information when reference labels are available;
- diagnostic `k=2..10` clustering sweep for target-partition stability;
- red/white `wine_type` separability comparison from original space to explicit PCA/SVD 2D;
- train/test reconstruction metrics and generalization gap;
- reconstruction error across nested retained dimensions using one train-fitted basis per method.

### Reproducibility

- deterministic random seed;
- persisted preprocessing statistics;
- persisted PCA/SVD parameters in compressed NumPy artifacts without pickle;
- JSON metrics and interpretation reports;
- timestamped run directories;
- explicit train fraction and component-selection configuration.

## 🔄 Architecture

```text
                    +----------------------+
                    |      input CSV       |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | load / validate data |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | train-only fit       |
                    | impute + standardize |
                    +----------+-----------+
                               |
                     +---------+---------+
                     |                   |
                     v                   v
             +---------------+   +---------------+
             | PCA from      |   | SVD reduction |
             | covariance +  |   | via NumPy SVD |
             | eig           |   |               |
             +-------+-------+   +-------+-------+
                     |                   |
                     +---------+---------+
                               |
             +-----------------+------------------+
             |                 |                  |
             v                 v                  v
       +-----------+     +-----------+      +-----------+
       | clustering|     | plots and |      | inverse   |
       | metrics   |     | loadings  |      | transform |
       +-----+-----+     +-----+-----+      +-----+-----+
             |                 |                  |
             +-----------------+------------------+
                               |
                               v
                    +----------------------+
                    | persisted run bundle |
                    | NPZ + JSON + CSV     |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | future CSV transform |
                    +----------------------+
```

## 🧠 PCA and SVD

### PCA

`PCAFromScratch` implements the full algorithmic flow:

1. subtract the feature mean;
2. calculate the covariance matrix;
3. obtain eigenvalues and eigenvectors with `numpy.linalg.eig`;
4. sort eigenpairs by descending eigenvalue;
5. convert eigenvalues into explained-variance ratios;
6. choose the retained component count;
7. apply a deterministic sign convention to the selected eigenvectors;
8. project centered samples onto the selected eigenvectors;
9. reconstruct with `inverse_transform` when required.

### SVD

`SVDFromScratch` performs:

1. mean centering;
2. `U, Σ, Vᵀ = numpy.linalg.svd(X)`;
3. conversion of singular values into variance contributions;
4. top-k right-singular-vector selection;
5. the same deterministic sign convention used by PCA;
6. projection onto the selected latent directions;
7. low-rank reconstruction.

For centered data, PCA eigenvectors and SVD right-singular vectors describe the same principal subspace up to floating-point effects and possible rotations inside degenerate eigenspaces. Ordinary sign ambiguity is normalized for stable presentation. Detailed mathematics and metric definitions are in [docs/METHODS.md](docs/METHODS.md).

## 🍷 Dataset and preprocessing

The default source is the UCI Wine Quality dataset. The downloader combines red and white wine tables and adds a non-feature `wine_type` metadata column.

The numerical source fields include acidity measures, sugar, chlorides, sulfur dioxide, density, pH, sulphates and alcohol. `quality` is used as the default reference target for external cluster comparison; it is not fed into PCA or SVD.

Preprocessing is deliberately persisted as a separate artifact. `TabularPreprocessor` stores:

- exact feature order;
- per-feature training median;
- training mean;
- training standard deviation;
- detected constant features.

At inference time extra columns are ignored, but every required training feature must exist. Required feature order is restored automatically before transformation.

Use a custom dataset by supplying another CSV path and target:

```bash
python main.py compare \
  --input data/raw/custom.csv \
  --target class \
  --variance-threshold 0.95
```

For a genuinely unlabeled dataset:

```bash
python main.py compare \
  --input data/raw/custom.csv \
  --target none
```

`--target none` means **no column is reserved as a target**. If it is used on a labeled numeric dataset such as Wine Quality, the former numeric target column (`quality`) becomes an ordinary feature. Use this mode only when that is the intended schema.

## ⚙️ CLI workflows

### Train one reducer

Automatic component count by retained variance:

```bash
python main.py train \
  --input data/raw/wine_quality.csv \
  --method pca \
  --variance-threshold 0.95
```

Explicit dimension count:

```bash
python main.py train \
  --input data/raw/wine_quality.csv \
  --method svd \
  --components 5
```

### Compare PCA and SVD

```bash
python main.py compare \
  --input data/raw/wine_quality.csv \
  --variance-threshold 0.95 \
  --clusters 6 \
  --train-fraction 0.8 \
  --seed 42
```

The comparison uses one preprocessing state and one deterministic train subset for both reducers so the metrics are directly comparable. The requested `--clusters` value remains the main clustering configuration; an additional diagnostic sweep reports `k=2..10` without selecting a post-hoc winner.

### Transform new data

After a training run:

```bash
python main.py transform \
  --input data/raw/new_wines.csv \
  --model artifacts/runs/<run>/pca_model.npz \
  --preprocessor artifacts/runs/<run>/preprocessor.json \
  --output data/processed/new_wines_reduced.csv
```

No fit operation occurs in this command. It reuses the stored train-time medians, means, scales and spectral components.

## 📊 Evaluation and interpretation

A normal comparison run saves:

- cumulative explained-variance comparison;
- PCA 2D and 3D embeddings when enough dimensions are retained;
- SVD 2D and 3D embeddings when enough dimensions are retained;
- PCA and SVD component-loading heatmaps;
- train/test reconstruction error vs component count plus numeric JSON;
- target clustering sweep (`k=2..10`) plus ARI plot when a target exists;
- red/white `wine_type` class counts and original-vs-2D separability metrics when available;
- target distribution;
- feature distributions;
- feature correlation heatmap;
- red-wine and white-wine cluster views when `wine_type` exists;
- JSON component interpretation and paired PCA/SVD component cosine similarities.

The selected dimensionality is justified either by an explicit `--components` value or by the smallest number of dimensions meeting `--variance-threshold`. The default 95% threshold is an information-retention/reconstruction objective. A separate explicit 2D visualization or separability experiment answers a different downstream question and does not replace that threshold.

If 95% variance requires most of the original dimensions, that is reported as limited linear compressibility rather than hidden by lowering the threshold after the result is known.

For unsupervised dimensionality reduction, conventional supervised overfitting is not the primary failure mode. The repository instead controls the relevant risks: train/inference leakage, unstable component selection, excessive dimensionality and information loss. Fit statistics come only from the configured training subset unless `--train-fraction 1.0` is explicitly requested. Held-out reconstruction and the train/test MSE gap provide direct evidence of how the train-fitted subspace behaves on unseen rows.

Wine `quality` is ordinal, while ARI/NMI treat target values as nominal categories. The target clustering sweep therefore measures recovery of the exact quality partition across cluster counts; it is not an ordinal prediction score.

No numeric result tables are hard-coded into the repository. Measurements are generated from the local run and stored in JSON artifacts.

## 📦 Reusable artifacts

Typical training output:

```text
artifacts/runs/20260819T184500Z_pca/
├── pca_model.npz
├── preprocessor.json
├── metrics.json
├── component_interpretation.json
├── reduced.csv
├── cumulative_variance.png
├── embedding_2d.png
├── embedding_3d.png
├── component_loadings.png
├── feature_distributions.png
├── feature_correlation.png
└── target_distribution.png
```

A comparison run additionally stores both model artifacts, PCA/SVD comparison plots, `reconstruction_curve.json`, `quality_clustering_sweep.json`, `wine_type_separability.json` when applicable, and per-subset cluster views.

The NPZ model format stores numeric arrays plus JSON metadata and is loaded with `allow_pickle=False`.

## 🧪 Optional experiments

### Exact NumPy t-SNE

A compact exact t-SNE implementation is included for nonlinear visualization without introducing a prebuilt dimensionality-reduction class:

```bash
python main.py nonlinear \
  --input data/raw/wine_quality.csv \
  --max-samples 1000 \
  --perplexity 30 \
  --iterations 750
```

It is intentionally bounded to exploratory samples because exact t-SNE requires O(n²) pairwise work and does not provide the reusable out-of-sample transform contract of PCA/SVD.

### UMAP

UMAP is kept as an optional dependency so the core environment remains lightweight:

```bash
python -m pip install -r requirements-extra.txt
python main.py umap \
  --input data/raw/wine_quality.csv \
  --max-samples 5000 \
  --neighbors 15 \
  --min-dist 0.1
```

The command saves a deterministic two-dimensional nonlinear embedding and its configuration. It is used as an exploratory comparison rather than as the persisted production reducer.

### SVD image compression

```bash
python main.py bonus-image \
  --input path/to/image.png \
  --ranks 5,20,50,100
```

Each rank produces a reconstructed image and reports MSE, PSNR and an estimated storage ratio.

### SVD signal denoising

```bash
python main.py bonus-signal \
  --rank 4 \
  --samples 1000 \
  --noise-std 0.45
```

The experiment embeds a synthetic noisy two-frequency signal into a Hankel trajectory matrix, keeps a low-rank SVD approximation and reconstructs the one-dimensional signal by diagonal averaging. The default rank is 4 because two real sinusoidal components require two phase directions each in the Hankel subspace.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for interpretation guidance.

## ✅ Tests

Install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run:

```bash
pytest -q
```

The suite covers:

- PCA dimensions, variance ordering, reconstruction and persistence;
- deterministic component-sign canonicalization and PCA/SVD alignment on a separated spectrum;
- SVD dimensions, explained variance, reconstruction and persistence;
- preprocessing imputation, constant features, schema checks and persistence;
- duplicate CSV headers and empty CSV error handling;
- nested train/test reconstruction curves and PCA/SVD reconstruction equivalence;
- end-to-end reduction plus K-means;
- exact t-SNE output sanity;
- image compression and signal-denoising primitives.

## 📁 Project structure

```text
spectral-learning/
├── data/
│   └── README.md
├── docs/
│   ├── EXPERIMENTS.md
│   └── METHODS.md
├── experiments/
│   ├── __init__.py
│   ├── image_compression.py
│   ├── signal_denoising.py
│   └── umap_embedding.py
├── models/
│   ├── __init__.py
│   ├── pca_model.py
│   ├── svd_model.py
│   └── tsne_model.py
├── scripts/
│   ├── __init__.py
│   └── download_wine_quality.py
├── tests/
│   ├── test_experiments.py
│   ├── test_pca.py
│   ├── test_pipeline.py
│   ├── test_preprocessing.py
│   ├── test_svd.py
│   └── test_tsne.py
├── utils/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── clustering.py
│   ├── data_loader.py
│   ├── interpretation.py
│   ├── matrix_operations.py
│   ├── metrics.py
│   ├── preprocessing.py
│   └── visualization.py
├── .gitignore
├── main.py
├── README.md
├── README_RU.md
├── requirements-dev.txt
├── requirements-extra.txt
├── requirements.txt
└── workflows.py
```

## ⚠️ Engineering notes

- PCA/SVD fit state is never recalculated during `transform`.
- Inference requires the complete training feature schema and rejects incompatible inputs.
- Missing and infinite numeric values are normalized before median imputation.
- Constant features receive a safe scale of `1.0` and are reported in metadata.
- Explicit `n_components` takes precedence over the variance threshold.
- PCA/SVD component signs are canonicalized for deterministic presentation; degenerate eigenspaces can still admit equivalent rotated bases.
- Generated datasets, models, metrics and plots are ignored by Git; source code and documentation remain clean.
- Exact t-SNE and UMAP are exploratory only; they are deliberately separate from persisted PCA/SVD inference.
- Runtime numbers depend on the local dataset, package versions and hardware and are therefore generated rather than embedded in documentation.

## 🧑‍💻 Author

- Nazar Yestayev (@nyestaye)
