from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_algo.data_ingest import GapPolicy
from ml_algo.pipeline import ingest_and_transform


def test_ingest_and_transform_phase02(tmp_path: Path) -> None:
    df = pd.read_csv("data/raw/@NQ - 5 min - ETH.csv", nrows=2000)
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    test_path = tmp_path / "sample.csv"
    df.to_csv(test_path, index=False)
    ha = ingest_and_transform(
        sources=[test_path],
        timezone="America/New_York",
        bar_sizes=["5min"],
        gap_policy=GapPolicy(max_gap_minutes=30),
    )
    assert {"ha_open", "ha_close", "ts_utc"}.issubset(ha.columns)
