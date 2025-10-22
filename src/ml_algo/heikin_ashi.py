from __future__ import annotations

import pandas as pd


def transform(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Convert OHLCV bars to Heikin-Ashi candles (SPEC §4.1)."""
    required = ["ts_utc", "open", "high", "low", "close"]
    missing = [col for col in required if col not in ohlcv.columns]
    if missing:
        raise ValueError(f"Missing columns for HA transform: {missing}")
    df = ohlcv.sort_values("ts_utc").reset_index(drop=True).copy()
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for idx in range(1, len(df)):
        ha_open.iloc[idx] = (ha_open.iloc[idx - 1] + ha_close.iloc[idx - 1]) / 2
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
    out = pd.DataFrame(
        {
            "ts_utc": df["ts_utc"],
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
            "ha_close": ha_close,
            "volume": df["volume"],
        }
    )
    assert_invariants(out)
    return out


def assert_invariants(ha_df: pd.DataFrame) -> None:
    """Validate HA recurrence identities and numeric sanity."""
    if ha_df[["ha_open", "ha_high", "ha_low", "ha_close"]].isnull().any().any():
        raise ValueError("HA transform produced NaNs")
    if not (ha_df["ha_high"] >= ha_df[["ha_open", "ha_close"]].max(axis=1)).all():
        raise AssertionError("ha_high invariant violated")
    if not (ha_df["ha_low"] <= ha_df[["ha_open", "ha_close"]].min(axis=1)).all():
        raise AssertionError("ha_low invariant violated")
