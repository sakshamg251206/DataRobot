"""Dataset profiling: what kind of data each column holds and how healthy it is."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from autods.core.dtypes import (
    categorical_columns,
    is_text,
    looks_like_datetime,
    numeric_columns,
    skewness,
)

ID_UNIQUE_RATIO = 0.95  # unique ratio above this -> probably an identifier
LOW_CARDINALITY = 10  # this many distinct values or fewer -> categorical-like
SKEW_THRESHOLD = 1.0  # |skew| above this -> consider a log transform


class ColumnKind(str, Enum):
    EMPTY = "Empty"
    DATETIME = "Date / time"
    ID = "Identifier"
    BINARY = "Binary"
    CATEGORICAL = "Categorical"
    HIGH_CARD_TEXT = "Free text / high-cardinality"
    DISCRETE = "Discrete numeric"
    CONTINUOUS = "Continuous numeric"


def classify_column(series: pd.Series) -> ColumnKind:
    """Assign a column to a :class:`ColumnKind` using simple, explainable rules."""
    values = series.dropna()
    if values.empty:
        return ColumnKind.EMPTY

    n_unique = values.nunique()
    unique_ratio = n_unique / max(len(series), 1)

    if looks_like_datetime(series):
        return ColumnKind.DATETIME
    if n_unique <= 2:
        return ColumnKind.BINARY

    if is_text(series):
        if n_unique <= LOW_CARDINALITY:
            return ColumnKind.CATEGORICAL
        if unique_ratio >= ID_UNIQUE_RATIO:
            # Short unique codes look like IDs; long unique strings look like free text.
            avg_len = values.astype(str).str.len().mean()
            return ColumnKind.ID if avg_len <= 20 else ColumnKind.HIGH_CARD_TEXT
        return ColumnKind.CATEGORICAL if unique_ratio < 0.5 else ColumnKind.HIGH_CARD_TEXT

    if pd.api.types.is_integer_dtype(series):
        if unique_ratio >= ID_UNIQUE_RATIO and len(series) >= 20 and _is_sequential(values):
            return ColumnKind.ID
        return ColumnKind.DISCRETE
    if pd.api.types.is_numeric_dtype(series):
        return ColumnKind.CONTINUOUS
    return ColumnKind.CATEGORICAL


def _is_sequential(values: pd.Series) -> bool:
    """Integer IDs are typically consecutive (1, 2, 3, …) once sorted."""
    diffs = np.diff(np.sort(values.to_numpy()))
    return bool(len(diffs) and np.median(diffs) == 1)


def _safe_skew(series: pd.Series) -> float | None:
    values = series.dropna()
    if len(values) < 3 or values.nunique() < 2:
        return None
    return skewness(values)


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, kind, cardinality, missingness, skew and sample values."""
    rows = []
    for col in df.columns:
        s = df[col]
        skew = _safe_skew(s) if pd.api.types.is_numeric_dtype(s) else None
        sample = ", ".join(str(v) for v in s.dropna().unique()[:3])
        rows.append(
            {
                "Column": col,
                "Type": str(s.dtype),
                "Kind": classify_column(s).value,
                "Unique": int(s.nunique(dropna=True)),
                "Missing %": round(float(s.isna().mean() * 100), 1),
                "Skewness": None if skew is None else round(skew, 2),
                "Examples": sample,
            }
        )
    return pd.DataFrame(rows)


def overview(df: pd.DataFrame) -> dict[str, float | int]:
    """Headline numbers for the dataset."""
    n_cells = max(df.size, 1)
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "numeric": len(numeric_columns(df)),
        "categorical": len(categorical_columns(df)),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": float(df.isna().sum().sum() / n_cells * 100),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
    }


def _severity(pct: float) -> str:
    if pct > 50:
        return "Critical (>50%)"
    if pct > 20:
        return "High (20–50%)"
    if pct > 5:
        return "Moderate (5–20%)"
    if pct > 0:
        return "Low (<5%)"
    return "None"


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values per column, worst first. Only columns with gaps are included."""
    counts = df.isna().sum()
    report = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing": counts.to_numpy(),
            "Missing %": (counts / max(len(df), 1) * 100).round(2).to_numpy(),
        }
    )
    report = report[report["Missing"] > 0].sort_values("Missing %", ascending=False)
    report["Severity"] = report["Missing %"].map(_severity)
    return report.reset_index(drop=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """``describe()`` for numeric columns plus skewness and kurtosis."""
    num = df[numeric_columns(df)]
    if num.empty:
        return pd.DataFrame()
    desc = num.describe().T.rename(
        columns={"std": "std dev", "25%": "Q1", "50%": "median", "75%": "Q3"}
    )
    desc["skewness"] = num.skew()
    desc["kurtosis"] = num.kurt()
    return desc.round(3)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Cardinality and dominant value for each text column."""
    rows = []
    for col in categorical_columns(df):
        s = df[col]
        counts = s.value_counts()
        top_freq = int(counts.iloc[0]) if not counts.empty else 0
        rows.append(
            {
                "Column": col,
                "Unique": int(s.nunique()),
                "Most common": str(counts.index[0]) if not counts.empty else "—",
                "Count": top_freq,
                "Share %": round(top_freq / max(len(s), 1) * 100, 1),
                "Missing %": round(float(s.isna().mean() * 100), 1),
            }
        )
    return pd.DataFrame(rows)


def skewed_columns(df: pd.DataFrame, threshold: float = SKEW_THRESHOLD) -> pd.Series:
    """Skewness of numeric columns with |skew| above ``threshold``, largest first."""
    skews = {c: _safe_skew(df[c]) for c in numeric_columns(df)}
    series = pd.Series({c: v for c, v in skews.items() if v is not None}, dtype=float)
    series = series[series.abs() >= threshold]
    return series.reindex(series.abs().sort_values(ascending=False).index)


def readiness_flags(profile: pd.DataFrame) -> dict[str, list[str]]:
    """Columns that need attention before machine learning, grouped by reason."""

    def cols(kind: ColumnKind) -> list[str]:
        return profile.loc[profile["Kind"] == kind.value, "Column"].tolist()

    return {
        "Identifier columns (drop before modelling)": cols(ColumnKind.ID),
        "Date columns (split into year / month / day features)": cols(ColumnKind.DATETIME),
        "Free-text or very high-cardinality columns (drop or summarise)": cols(
            ColumnKind.HIGH_CARD_TEXT
        ),
        "Empty columns (drop)": cols(ColumnKind.EMPTY),
    }


def top_correlations(
    df: pd.DataFrame, columns: list[str] | None = None, n: int = 10
) -> pd.DataFrame:
    """Strongest pairwise Pearson correlations between numeric columns."""
    cols = columns or numeric_columns(df)
    if len(cols) < 2:
        return pd.DataFrame(columns=["Feature A", "Feature B", "Correlation"])
    corr = df[cols].corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs: pd.Series = pd.Series(corr.where(mask).stack()).dropna()
    if pairs.empty:
        return pd.DataFrame(columns=["Feature A", "Feature B", "Correlation"])
    pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(n)
    out = pairs.reset_index()
    out.columns = ["Feature A", "Feature B", "Correlation"]
    out["Correlation"] = out["Correlation"].round(3)
    return out
