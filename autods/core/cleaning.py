"""Data cleaning steps.

Each step is a pure function ``(df, log, ...) -> df`` that appends a plain-English
description of what it did to ``log``. :func:`clean_dataset` runs them in a
sensible order. Nothing here mutates its input.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from autods.core.dtypes import (
    categorical_columns,
    is_binary,
    looks_like_datetime,
    numeric_columns,
    to_datetime,
)

IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
NUMERIC_STRING_MIN_PARSE_RATE = 0.95
DATETIME_MIN_PARSE_RATE = 0.8
_NUMERIC_JUNK = r"[\$£€¥%,\s]"
_MISSING_TOKENS = {"", "-", "na", "n/a", "nan", "null", "none", "?"}


class ImputeStrategy(str, Enum):
    MEDIAN = "Median"
    MEAN = "Mean"
    KNN = "KNN (5 nearest rows)"
    FILL = "Forward / backward fill (time-ordered data)"


class OutlierMethod(str, Enum):
    IQR = "IQR (1.5 x interquartile range)"
    ZSCORE = "Z-score (3 standard deviations)"


class OutlierAction(str, Enum):
    CAP = "Cap to the boundary"
    REMOVE = "Remove the row"
    KEEP = "Leave as is"


@dataclass(frozen=True)
class CleaningOptions:
    impute: ImputeStrategy = ImputeStrategy.MEDIAN
    outlier_method: OutlierMethod = OutlierMethod.IQR
    outlier_action: OutlierAction = OutlierAction.CAP


@dataclass
class CleaningReport:
    rows_in: int
    cols_in: int
    rows_out: int = 0
    cols_out: int = 0
    remaining_missing: int = 0
    log: list[str] = field(default_factory=list)


def _list(cols: Iterable[str]) -> str:
    return ", ".join(f"`{c}`" for c in cols)


# ── Steps ──────────────────────────────────────────────────────────────────────
def drop_useless_columns(
    df: pd.DataFrame, log: list[str], protect: Iterable[str] = ()
) -> pd.DataFrame:
    """Drop empty, constant and index-artifact columns (never the protected ones)."""
    keep = set(protect)
    empty = [c for c in df.columns if c not in keep and df[c].isna().all()]
    constant = [
        c
        for c in df.columns
        if c not in keep and c not in empty and df[c].nunique(dropna=True) <= 1
    ]
    artifacts = [
        c
        for c in df.columns
        if c not in keep and c not in empty + constant and str(c).lower().startswith("unnamed")
    ]

    if empty:
        log.append(f"Dropped {len(empty)} empty column(s): {_list(empty)}.")
    if constant:
        log.append(f"Dropped {len(constant)} constant column(s): {_list(constant)}.")
    if artifacts:
        log.append(f"Dropped index-artifact column(s): {_list(artifacts)}.")
    if not (empty or constant or artifacts):
        log.append("No empty or constant columns found.")
    return df.drop(columns=empty + constant + artifacts)


def fix_numeric_strings(
    df: pd.DataFrame, log: list[str], protect: Iterable[str] = ()
) -> pd.DataFrame:
    """Convert text columns that really hold numbers ("$1,200", "45%") to numbers.

    A column is converted only if at least 95% of its non-missing values parse,
    so genuine text columns are never destroyed.
    """
    out = df.copy()
    keep = set(protect)
    converted = []
    for col in categorical_columns(out, exclude=keep):
        s = out[col]
        if s.dropna().empty:
            continue
        text = s.astype("string").str.strip()
        text = text.mask(text.str.lower().isin(_MISSING_TOKENS))
        stripped = text.str.replace(_NUMERIC_JUNK, "", regex=True)
        if not stripped.str.contains(r"\d", regex=True).fillna(False).any():
            continue
        parsed = pd.to_numeric(stripped, errors="coerce")
        present = text.notna()
        if present.sum() and parsed[present].notna().mean() >= NUMERIC_STRING_MIN_PARSE_RATE:
            out[col] = parsed.astype("float64")
            converted.append(col)
    if converted:
        log.append(f"Converted text to numbers in: {_list(converted)}.")
    return out


def parse_datetime_columns(
    df: pd.DataFrame, log: list[str], protect: Iterable[str] = ()
) -> pd.DataFrame:
    """Convert text columns that hold dates into real datetime columns."""
    out = df.copy()
    converted = []
    for col in categorical_columns(out, exclude=protect):
        if not looks_like_datetime(out[col]):
            continue
        parsed = to_datetime(out[col])
        present = out[col].notna()
        if present.sum() and parsed[present].notna().mean() >= DATETIME_MIN_PARSE_RATE:
            out[col] = parsed
            converted.append(col)
    if converted:
        log.append(f"Parsed dates in: {_list(converted)}.")
    return out


def remove_duplicates(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    n_before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    removed = n_before - len(out)
    if removed:
        log.append(f"Removed {removed:,} duplicate row(s) ({removed / n_before:.1%} of the data).")
    else:
        log.append("No duplicate rows found.")
    return out


def drop_missing_target(df: pd.DataFrame, target: str, log: list[str]) -> pd.DataFrame:
    """Rows without a target value cannot be used for training and must not be imputed."""
    missing = df[target].isna()
    if not missing.any():
        return df
    log.append(f"Dropped {int(missing.sum()):,} row(s) with no value for target `{target}`.")
    return df.loc[~missing].reset_index(drop=True)


def fill_missing_values(
    df: pd.DataFrame,
    log: list[str],
    strategy: ImputeStrategy = ImputeStrategy.MEDIAN,
    protect: Iterable[str] = (),
) -> pd.DataFrame:
    """Impute numeric gaps with ``strategy`` and text gaps with the most common value."""
    out = df.copy()
    keep = set(protect)
    num_cols = numeric_columns(out, exclude=keep)
    num_missing = [c for c in num_cols if out[c].isna().any()]

    if num_missing:
        if strategy is ImputeStrategy.KNN:
            imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
            out[num_cols] = imputer.fit_transform(out[num_cols])
        elif strategy is ImputeStrategy.FILL:
            out[num_missing] = out[num_missing].ffill().bfill()
        elif strategy is ImputeStrategy.MEAN:
            out[num_missing] = out[num_missing].fillna(out[num_missing].mean())
        else:
            out[num_missing] = out[num_missing].fillna(out[num_missing].median())
        log.append(f"Filled numeric gaps using {strategy.value.lower()}: {_list(num_missing)}.")

    cat_missing = [c for c in categorical_columns(out, exclude=keep) if out[c].isna().any()]
    for col in cat_missing:
        if strategy is ImputeStrategy.FILL:
            out[col] = out[col].ffill().bfill()
        else:
            mode = out[col].mode(dropna=True)
            if not mode.empty:
                out[col] = out[col].fillna(mode.iloc[0])
    if cat_missing:
        how = "neighbouring rows" if strategy is ImputeStrategy.FILL else "the most common value"
        log.append(f"Filled text gaps with {how}: {_list(cat_missing)}.")

    if not num_missing and not cat_missing:
        log.append("No missing values to fill.")
    return out


def outlier_bounds(series: pd.Series, method: OutlierMethod) -> tuple[float, float] | None:
    """Lower/upper fences for a numeric series, or ``None`` if it has no spread."""
    values = series.dropna()
    if values.empty:
        return None
    if method is OutlierMethod.IQR:
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            return None
        return float(q1 - IQR_MULTIPLIER * iqr), float(q3 + IQR_MULTIPLIER * iqr)
    mean, std = values.mean(), values.std()
    if not std or np.isnan(std):
        return None
    return float(mean - ZSCORE_THRESHOLD * std), float(mean + ZSCORE_THRESHOLD * std)


def handle_outliers(
    df: pd.DataFrame,
    log: list[str],
    method: OutlierMethod = OutlierMethod.IQR,
    action: OutlierAction = OutlierAction.CAP,
    protect: Iterable[str] = (),
) -> pd.DataFrame:
    """Cap or remove extreme numeric values. Binary columns are never touched.

    In remove mode every column's outlier mask is combined first and applied
    once, so the result does not depend on column order.
    """
    if action is OutlierAction.KEEP:
        log.append("Outliers left unchanged.")
        return df

    out = df.copy()
    keep_rows = pd.Series(True, index=out.index)
    counts: dict[str, int] = {}

    for col in numeric_columns(out, exclude=protect):
        if is_binary(out[col]):
            continue
        bounds = outlier_bounds(out[col], method)
        if bounds is None:
            continue
        lower, upper = bounds
        is_outlier = (out[col] < lower) | (out[col] > upper)
        n = int(is_outlier.sum())
        if not n:
            continue
        counts[col] = n
        if action is OutlierAction.CAP:
            out[col] = out[col].clip(lower=lower, upper=upper)
        else:
            keep_rows &= ~is_outlier

    if not counts:
        log.append("No outliers detected.")
        return out

    detail = ", ".join(f"`{c}` ({n})" for c, n in counts.items())
    if action is OutlierAction.CAP:
        log.append(f"Capped outliers to their boundaries in: {detail}.")
        return out

    n_before = len(out)
    out = out.loc[keep_rows].reset_index(drop=True)
    removed = n_before - len(out)
    log.append(
        f"Removed {removed:,} row(s) containing outliers ({removed / n_before:.1%} of the data). "
        f"Outliers per column: {detail}."
    )
    return out


# ── Orchestrator ───────────────────────────────────────────────────────────────
def clean_dataset(
    df: pd.DataFrame,
    options: CleaningOptions | None = None,
    target: str | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run every cleaning step in order and return the cleaned data plus a report.

    Order: drop useless columns → numbers stored as text → dates → duplicates →
    (rows missing the target) → impute → outliers. If ``target`` is given it is
    never dropped, imputed or outlier-treated.
    """
    options = options or CleaningOptions()
    protect = [target] if target else []
    report = CleaningReport(rows_in=len(df), cols_in=df.shape[1])
    log = report.log

    out = drop_useless_columns(df, log, protect=protect)
    out = fix_numeric_strings(out, log, protect=protect)
    out = parse_datetime_columns(out, log, protect=protect)
    out = remove_duplicates(out, log)
    if target:
        out = drop_missing_target(out, target, log)
    out = fill_missing_values(out, log, strategy=options.impute, protect=protect)
    out = handle_outliers(
        out, log, method=options.outlier_method, action=options.outlier_action, protect=protect
    )

    report.rows_out, report.cols_out = out.shape
    report.remaining_missing = int(out.isna().sum().sum())
    return out, report
