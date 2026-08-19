# Data

The default workflow uses the UCI Wine Quality data. Raw CSV files are intentionally not committed.

Create the combined dataset with:

```bash
python scripts/download_wine_quality.py
```

The command writes `data/raw/wine_quality.csv` and adds a `wine_type` metadata column while keeping the original numerical measurements and `quality` target.

Custom numerical CSV datasets are also supported through the CLI `--input` option.
