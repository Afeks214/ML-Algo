from __future__ import annotations

import pandas as pd
import pytest

from ml_algo.heikin_ashi import assert_invariants, transform


@pytest.fixture(scope="module")
def sample_ohlcv() -> pd.DataFrame:
    df = pd.read_csv("data/raw/@NQ - 5 min - ETH.csv", nrows=500)
    df = df.rename(
        columns={
            "Timestamp": "ts",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["ts"] = pd.to_datetime(df["ts"], dayfirst=True, format="mixed").dt.tz_localize("America/New_York")
    df["ts_utc"] = df["ts"].dt.tz_convert("UTC")
    return df[["ts_utc", "open", "high", "low", "close", "volume"]]


def test_transform_shapes(sample_ohlcv: pd.DataFrame) -> None:
    ha = transform(sample_ohlcv)
    assert len(ha) == len(sample_ohlcv)
    assert {"ha_open", "ha_high", "ha_low", "ha_close"}.issubset(ha.columns)


def test_assert_invariants(sample_ohlcv: pd.DataFrame) -> None:
    ha = transform(sample_ohlcv)
    assert_invariants(ha)
