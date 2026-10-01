"""Column type helpers that behave the same on pandas 2.x and 3.x.

pandas 3 stores text in a dedicated ``str`` dtype instead of ``object``, so
``select_dtypes(include="object")`` silently misses text columns there.
These helpers hide that difference from the rest of the code base.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import pandas as pd

TEXT_DTYPES: list[Any] = ["object", "string", "category"]


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Numeric (int/float) columns. Booleans are *not* included."""
    skip = set(exclude)
    return [c for c in df.select_dtypes(include="number").columns if c not in skip]


def categorical_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Text / categorical columns (object, string or category dtype)."""
    skip = set(exclude)
    return [c for c in df.select_dtypes(include=TEXT_DTYPES).columns if c not in skip]


def boolean_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    skip = set(exclude)
    return [c for c in df.select_dtypes(include="bool").columns if c not in skip]


def datetime_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    skip = set(exclude)
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) and c not in skip]


def is_text(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ) and not pd.api.types.is_datetime64_any_dtype(series)


def is_binary(series: pd.Series) -> bool:
    """True when a column has at most two distinct non-null values."""
    return bool(series.nunique(dropna=True) <= 2)


def numeric_parse_rate(series: pd.Series) -> float:
    """Share of non-null values that parse as numbers (0.0 for an empty series)."""
    values = series.dropna()
    if values.empty:
        return 0.0
    parsed = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
    return float(parsed.notna().mean())


def to_datetime(series: pd.Series) -> pd.Series:
    """Parse a column to datetime, coercing unparseable values to ``NaT``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce", format="mixed")


def looks_like_datetime(series: pd.Series, threshold: float = 0.8, sample_size: int = 200) -> bool:
    """Heuristic: does a text column hold dates?

    Purely numeric strings ("1", "2024") are rejected because the date parser
    happily turns them into timestamps, which would misclassify ID and count
    columns as dates.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not is_text(series):
        return False
    sample = series.dropna().astype(str).head(sample_size)
    if sample.empty or numeric_parse_rate(sample) > 0.5:
        return False
    # Real dates contain a separator or a month name; this rejects free text
    # that dateutil can occasionally coerce (e.g. "may" or "march").
    if (
        sample.str.contains(r"[-/:.]|\d\s+[A-Za-z]{3}|[A-Za-z]{3}\s+\d", regex=True).mean()
        < threshold
    ):
        return False
    return bool(to_datetime(sample).notna().mean() >= threshold)
