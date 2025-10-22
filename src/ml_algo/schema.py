from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class SchemaDefinition:
    """
    Declarative schema description enforcing column presence and dtypes.

    Cross-reference: SPEC.md §9.2 Data Contracts.
    """

    name: str
    columns: Mapping[str, str]
    timezone_field: str | None = None
    allow_nulls: Iterable[str] = ()
    datetime_kwargs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


OHLCV_SCHEMA = SchemaDefinition(
    name="Raw OHLCV",
    columns={
        "ts": "datetime64[ns]",
        "open": "float",
        "high": "float",
        "low": "float",
        "close": "float",
        "volume": "float",
    },
    timezone_field=None,
    allow_nulls=("volume",),
    datetime_kwargs={"ts": {"dayfirst": True, "format": "mixed"}},
)

HEIKIN_ASHI_SCHEMA = SchemaDefinition(
    name="HeikinAshi",
    columns={
        "ts": "datetime64[ns, tz]",
        "ha_open": "float",
        "ha_high": "float",
        "ha_low": "float",
        "ha_close": "float",
    },
    timezone_field="ts",
)

LABELS_SCHEMA = SchemaDefinition(
    name="Labels",
    columns={"ts": "datetime64[ns, tz]", "y": "int"},
    timezone_field="ts",
)

INFERENCE_INPUT_SCHEMA = SchemaDefinition(
    name="InferenceInput",
    columns={
        "ts": "datetime64[ns, tz]",
        "feature_vector": "object",
    },
    timezone_field="ts",
)


class SchemaValidationError(ValueError):
    pass


def _ensure_columns(df: pd.DataFrame, schema: SchemaDefinition) -> None:
    missing = [col for col in schema.columns if col not in df.columns]
    if missing:
        raise SchemaValidationError(f"{schema.name}: missing columns {missing}")


def _ensure_timezone(df: pd.DataFrame, schema: SchemaDefinition) -> None:
    if schema.timezone_field is None:
        return
    field = schema.timezone_field
    tz = df[field].dt.tz if field in df.columns else None
    if tz is None:
        raise SchemaValidationError(f"{schema.name}: column '{field}' must be timezone-aware.")


def _coerce_types(df: pd.DataFrame, schema: SchemaDefinition) -> pd.DataFrame:
    coerced = df.copy()
    for col, dtype in schema.columns.items():
        if dtype == "datetime64[ns, tz]":
            kwargs = dict(schema.datetime_kwargs.get(col, {}))
            kwargs.setdefault("errors", "raise")
            coerced[col] = pd.to_datetime(coerced[col], **kwargs)
        elif dtype == "datetime64[ns]":
            kwargs = dict(schema.datetime_kwargs.get(col, {}))
            kwargs.setdefault("errors", "raise")
            coerced[col] = pd.to_datetime(coerced[col], **kwargs)
        elif dtype == "float":
            coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
        elif dtype == "int":
            coerced[col] = pd.to_numeric(coerced[col], errors="coerce", downcast="integer")
        else:
            # fallback: no coercion
            coerced[col] = coerced[col]
    return coerced


def _ensure_null_policy(df: pd.DataFrame, schema: SchemaDefinition) -> None:
    disallowed = set(schema.columns.keys()) - set(schema.allow_nulls)
    if df[list(disallowed)].isnull().any().any():
        raise SchemaValidationError(f"{schema.name}: nulls detected in columns {sorted(disallowed)}")


def validate_dataframe(df: pd.DataFrame, schema: SchemaDefinition) -> pd.DataFrame:
    """
    Validate dataframe against schema definition.

    Returns a coerced copy to avoid mutating caller state.
    """
    _ensure_columns(df, schema)
    coerced = _coerce_types(df, schema)
    _ensure_timezone(coerced, schema)
    _ensure_null_policy(coerced, schema)
    return coerced


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, OHLCV_SCHEMA)


def validate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, HEIKIN_ASHI_SCHEMA)


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, LABELS_SCHEMA)


def validate_inference_input(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, INFERENCE_INPUT_SCHEMA)
