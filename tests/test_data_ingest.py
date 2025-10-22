from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_algo.data_ingest import GapPolicy, apply_gap_policy, load, validate_schema

DATA_PATH = Path("data") / "raw" / "@NQ - 5 min - ETH.csv"


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, nrows=1000)
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
    return df


def test_validate_schema(sample_df: pd.DataFrame) -> None:
    validated = validate_schema(sample_df)
    assert validated["ts"].dtype.kind in {"M"}
    assert validated[["open", "high", "low", "close", "volume"]].isnull().sum().sum() == 0


def test_load_roundtrip(monkeypatch: pytest.MonkeyPatch, sample_df: pd.DataFrame) -> None:
    tmp_path = Path("data/raw_test.csv")
    sample_df.to_csv(tmp_path, index=False)
    try:
        result = load([tmp_path], timezone="America/New_York", bar_sizes=["5min"])
        assert "ts_utc" in result.columns
        assert result["ts_utc"].dt.tz is not None
    finally:
        tmp_path.unlink(missing_ok=True)


def test_gap_policy_filters_large_gaps(sample_df: pd.DataFrame) -> None:
    df = validate_schema(sample_df)
    df = df.iloc[:10].copy()
    df.loc[df.index[-1], "ts"] = df.loc[df.index[-1], "ts"] + pd.Timedelta(hours=10)
    df = df.rename(columns={"ts": "ts"})
    df["ts"] = pd.to_datetime(df["ts"])
    df["ts"] = df["ts"].dt.tz_localize("America/New_York")
    df["ts_utc"] = df["ts"].dt.tz_convert("UTC")
    filtered = apply_gap_policy(df, GapPolicy(max_gap_minutes=60))
    assert len(filtered) == 9  # last row dropped
