from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pytz

from ml_algo.schema import SchemaValidationError, validate_ohlcv

COLUMN_ALIASES = {
    "Timestamp": "ts",
    "Date": "ts",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GapPolicy:
    max_gap_minutes: int
    fill_volume: bool = False
    forward_fill: bool = False


def load(sources: Iterable[str | Path], timezone: str, bar_sizes: Iterable[str]) -> pd.DataFrame:
    """
    Load OHLCV CSV sources, enforce schema, and annotate metadata.

    Args:
        sources: iterable of file paths.
        timezone: timezone string for naive timestamps in files.
        bar_sizes: list of expected bar resolutions (currently informational).
    """
    frames: List[pd.DataFrame] = []
    for src in sources:
        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")
        df = pd.read_csv(path)
        df = df.rename(columns={src_col: dst_col for src_col, dst_col in COLUMN_ALIASES.items() if src_col in df.columns})
        logger.debug("Loaded %s rows from %s", len(df), path)
        frames.append(df)
    if not frames:
        raise ValueError("No sources provided")
    combined = pd.concat(frames, ignore_index=True)
    combined["bar_size"] = list(bar_sizes)[0] if bar_sizes else "unknown"
    try:
        combined = validate_schema(combined)
    except SchemaValidationError as exc:
        raise ValueError(str(exc)) from exc
    combined = to_utc(combined, timezone)
    return combined.sort_values("ts_utc").reset_index(drop=True)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and numeric types are coerced."""
    return validate_ohlcv(df)


def to_utc(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Convert timestamp column to UTC and add `ts_utc` column."""
    df = df.copy()
    tz = pytz.timezone(timezone)
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(tz)
    df["ts_utc"] = df["ts"].dt.tz_convert(pytz.UTC)
    return df


def apply_gap_policy(df: pd.DataFrame, policy: GapPolicy) -> pd.DataFrame:
    """Drop or flag gaps larger than allowed; enforce no forward-filling by default."""
    df = df.sort_values("ts_utc").reset_index(drop=True).copy()
    df["delta_minutes"] = df["ts_utc"].diff().dt.total_seconds().div(60.0).fillna(0.0)
    mask = df["delta_minutes"] <= policy.max_gap_minutes
    filtered = df.loc[mask].drop(columns=["delta_minutes"])
    removed = len(df) - len(filtered)
    if removed:
        logger.warning("Removed %s rows exceeding gap policy", removed)
    if not policy.fill_volume:
        filtered = filtered.dropna(subset=["volume"])
    return filtered.reset_index(drop=True)
