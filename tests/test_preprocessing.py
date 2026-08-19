import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import TabularPreprocessor


def test_preprocessor_imputes_and_standardizes() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [5.0, 5.0, 5.0, 5.0]})
    model = TabularPreprocessor()
    transformed = model.fit_transform(frame, ["a", "b"])
    assert np.isfinite(transformed).all()
    assert "b" in model.constant_features_
    assert np.allclose(transformed[:, 1], 0.0)


def test_preprocessor_rejects_missing_inference_feature() -> None:
    model = TabularPreprocessor().fit(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), ["a", "b"])
    with pytest.raises(ValueError, match="Missing required"):
        model.transform(pd.DataFrame({"a": [5]}))


def test_preprocessor_save_load_roundtrip(tmp_path) -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 6.0, 8.0]})
    model = TabularPreprocessor().fit(frame, ["a", "b"])
    path = model.save(tmp_path / "preprocessor.json")
    loaded = TabularPreprocessor.load(path)
    assert np.allclose(model.transform(frame), loaded.transform(frame))
