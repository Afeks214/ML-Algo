from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ml_algo.data_ingest import GapPolicy, apply_gap_policy, load
from ml_algo.heikin_ashi import transform as ha_transform


def ingest_and_transform(
    sources: Iterable[str | Path],
    timezone: str,
    bar_sizes: Iterable[str],
    gap_policy: GapPolicy,
) -> pd.DataFrame:
    """Phase 1–2 pipeline stub: ingest OHLCV and produce Heikin-Ashi candles."""
    ohlcv = load(sources, timezone=timezone, bar_sizes=bar_sizes)
    ohlcv = apply_gap_policy(ohlcv, gap_policy)
    ha = ha_transform(ohlcv)
    return ha

