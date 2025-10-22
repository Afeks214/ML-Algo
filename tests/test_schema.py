from __future__ import annotations

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype

from ml_algo.schema import (
    SchemaValidationError,
    validate_heikin_ashi,
    validate_inference_input,
    validate_labels,
    validate_ohlcv,
)


def test_validate_ohlcv_enforces_columns() -> None:
    df = pd.DataFrame(
        {
            "ts": ["2024-01-01 09:30:00"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.3],
            "volume": [1000],
        }
    )
    validated = validate_ohlcv(df)
    assert "ts" in validated
    assert is_datetime64_any_dtype(validated["ts"])


def test_validate_heikin_ashi_blocks_nulls() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01 09:30:00"], utc=True),
            "ha_open": [1.0],
            "ha_high": [None],
            "ha_low": [0.9],
            "ha_close": [1.05],
        }
    )
    with pytest.raises(SchemaValidationError):
        validate_heikin_ashi(df)


def test_validate_labels_requires_timezone() -> None:
    df = pd.DataFrame({"ts": ["2024-01-01"], "y": [1]})
    with pytest.raises(SchemaValidationError):
        validate_labels(df)


def test_validate_inference_accepts_feature_vectors() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01 09:30:00"], utc=True),
            "feature_vector": [[0.1, 0.2, 0.3]],
        }
    )
    validated = validate_inference_input(df)
    assert tuple(validated.columns) == ("ts", "feature_vector")
