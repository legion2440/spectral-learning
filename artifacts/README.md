# Artifacts

Generated experiment output is stored here locally and ignored by Git.

Typical layout:

```text
artifacts/
└── runs/
    └── <UTC timestamp>_<workflow>/
        ├── metrics.json
        ├── preprocessor.json
        ├── *_model.npz
        ├── component_interpretation.json
        ├── reduced.csv
        └── *.png
```

Model NPZ files contain numeric parameters plus JSON metadata and are loaded with NumPy `allow_pickle=False`.
