"""Smart Auto Prep: one click from raw data to a fully numeric, model-ready table.

The pipeline makes the same decisions an experienced analyst would make by
default, logs each one, and reports a before/after *readiness score* so the
effect is visible. All steps are target-aware: the target column is never
dropped, imputed, capped, encoded or used to choose features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from autods.core.cleaning import (
    OutlierAction,
    OutlierMethod,
    drop_missing_target,
    drop_useless_columns,
    fix_numeric_strings,
    handle_outliers,
    parse_datetime_columns,
    remove_duplicates,
)
from autods.core.dtypes import categorical_columns, numeric_columns
from autods.core.features import encode_booleans, extract_datetime_features
from autods.core.modeling import plan_features

ONE_HOT_MAX_UNIQUE = 10
COLLINEARITY_THRESHOLD = 0.95


@dataclass
class SmartResult:
    df: pd.DataFrame
    target: str
    score_before: float
    score_after: float
    log: list[str] = field(default_factory=list)


def readiness_score(df: pd.DataFrame, target: str | None = None) -> float:
    """0–100 score of how ready a table is for machine learning.

    Starts at 100 and subtracts (weights in brackets):
    missing cells (30), duplicate rows (10), non-numeric features (30),
    constant features (15) and features more than half empty (15).
    """
    if df.empty:
        return 0.0
    features = [c for c in df.columns if c != target]
    score = 100.0
    score -= df.isna().sum().sum() / max(df.size, 1) * 30
    score -= df.duplicated().sum() / max(len(df), 1) * 10
    if features:
        n = len(features)
        non_numeric = n - len(numeric_columns(df[features]))
        constant = sum(df[c].nunique(dropna=True) <= 1 for c in features)
        mostly_empty = sum(df[c].isna().mean() > 0.5 for c in features)
        score -= non_numeric / n * 30
        score -= constant / n * 15
        score -= mostly_empty / n * 15
    return round(float(np.clip(score, 0, 100)), 1)


def skew_aware_impute(df: pd.DataFrame, target: str, log: list[str]) -> pd.DataFrame:
    """Median for skewed numeric columns (|skew| > 1), mean otherwise; mode for text."""
    out = df.copy()
    median_cols, mean_cols = [], []
    for col in numeric_columns(out, exclude=[target]):
        values = out[col]
        if not values.isna().any() or values.dropna().empty:
            continue
        skew = values.skew() if values.nunique() > 2 else 0.0
        if abs(skew) > 1:
            out[col] = values.fillna(values.median())
            median_cols.append(col)
        else:
            out[col] = values.fillna(values.mean())
            mean_cols.append(col)
    if median_cols:
        log.append(f"Filled gaps in skewed columns with the median: {', '.join(median_cols)}.")
    if mean_cols:
        log.append(f"Filled gaps in symmetric columns with the mean: {', '.join(mean_cols)}.")

    text_cols = [c for c in categorical_columns(out, exclude=[target]) if out[c].isna().any()]
    for col in text_cols:
        mode = out[col].mode(dropna=True)
        if not mode.empty:
            out[col] = out[col].fillna(mode.iloc[0])
    if text_cols:
        log.append(
            f"Filled gaps in text columns with the most common value: {', '.join(text_cols)}."
        )
    return out


def encode_for_model(df: pd.DataFrame, target: str, log: list[str]) -> pd.DataFrame:
    """One-hot encode low-cardinality text, ordinal-encode the rest. Target untouched."""
    out = encode_booleans(df, log)
    cols = categorical_columns(out, exclude=[target])
    one_hot = [c for c in cols if out[c].nunique() <= ONE_HOT_MAX_UNIQUE]
    ordinal = [c for c in cols if c not in one_hot]
    for col in ordinal:
        out[col] = pd.factorize(out[col], sort=True)[0]
    if ordinal:
        log.append(f"Ordinal-encoded columns with many categories: {', '.join(ordinal)}.")
    if one_hot:
        n_before = out.shape[1]
        out = pd.get_dummies(out, columns=one_hot, drop_first=True, dtype=int)
        log.append(
            f"One-hot encoded {', '.join(one_hot)} "
            f"({out.shape[1] - n_before + len(one_hot)} new 0/1 columns)."
        )
    return out


def drop_collinear(df: pd.DataFrame, target: str, log: list[str]) -> pd.DataFrame:
    """Drop the later column of every feature pair with |correlation| above the threshold."""
    cols = numeric_columns(df, exclude=[target])
    if len(cols) < 2:
        return df
    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [c for c in upper.columns if (upper[c] > COLLINEARITY_THRESHOLD).any()]
    if to_drop:
        log.append(
            f"Dropped near-duplicate features (|r| > {COLLINEARITY_THRESHOLD}): {', '.join(to_drop)}."
        )
        return df.drop(columns=to_drop)
    return df


def smart_prepare(df: pd.DataFrame, target: str) -> SmartResult:
    """Run the full automatic preparation pipeline for predicting ``target``."""
    if target not in df.columns:
        raise ValueError(f"Target column `{target}` not found.")
    if df[target].isna().all():
        raise ValueError(f"Target column `{target}` has no values.")

    log: list[str] = []
    before = readiness_score(df, target)
    protect = [target]

    out = drop_useless_columns(df, log, protect=protect)
    out = fix_numeric_strings(out, log, protect=protect)
    out = parse_datetime_columns(out, log, protect=protect)
    out = remove_duplicates(out, log)
    out = drop_missing_target(out, target, log)

    plan = plan_features(out, target)
    unusable = [c for c in plan.excluded if c in out.columns]
    if unusable:
        reasons = "; ".join(f"{c} ({plan.excluded[c]})" for c in unusable)
        log.append(f"Dropped columns that can't help a model: {reasons}.")
        out = out.drop(columns=unusable)

    out = skew_aware_impute(out, target, log)
    out = handle_outliers(out, log, OutlierMethod.IQR, OutlierAction.CAP, protect=protect)
    out = extract_datetime_features(out, log, protect=protect)
    out = encode_for_model(out, target, log)
    out = drop_collinear(out, target, log)

    leftover = int(out.drop(columns=[target]).isna().sum().sum())
    if leftover:
        feature_cols = [c for c in out.columns if c != target]
        out[feature_cols] = out[feature_cols].fillna(out[feature_cols].median())
        log.append(f"Filled {leftover} remaining gap(s) (from partial dates) with column medians.")

    after = readiness_score(out, target)
    log.append(f"Readiness score: {before:.1f} → {after:.1f}.")
    return SmartResult(df=out, target=target, score_before=before, score_after=after, log=log)
