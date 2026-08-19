"""Download and combine the UCI Wine Quality red and white datasets."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
OUTPUT_PATH = Path("data/raw/wine_quality.csv")


def download_dataset(output: Path = OUTPUT_PATH) -> Path:
    """Fetch the official archive and write one combined CSV with wine type metadata."""
    try:
        with urlopen(DATASET_URL, timeout=30) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to download Wine Quality dataset: {exc}") from exc

    try:
        with ZipFile(BytesIO(payload)) as archive:
            with archive.open("winequality-red.csv") as handle:
                red = pd.read_csv(handle, sep=";")
            with archive.open("winequality-white.csv") as handle:
                white = pd.read_csv(handle, sep=";")
    except (BadZipFile, KeyError, OSError) as exc:
        raise RuntimeError(f"Downloaded Wine Quality archive is invalid: {exc}") from exc

    red["wine_type"] = "red"
    white["wine_type"] = "white"
    combined = pd.concat([red, white], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    path = download_dataset()
    print(path)
