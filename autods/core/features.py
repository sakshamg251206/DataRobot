"""Feature engineering: turning cleaned columns into model-friendly numbers.

These transformations produce a downloadable, fully numeric dataset. Model
training in :mod:`autods.core.modeling` re-learns its own preprocessing inside
a scikit-learn pipeline so evaluation stays leakage-free either way.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from autods.core.dtypes import (
    boolean_columns,
    categorical_columns,
    datetime_columns,
    is_binary,
    numeric_columns,
)

ONE_HOT_MAX_CATEGORIES = 50
RATIO_MAX_FEATURES = 5
POLY_MAX_INPUTS = 4
DATE_PARTS = ("year", "month", "day", "dayofweek", "quarter")


class Encoding(str, Enum):
    ONE_HOT = "One-hot (one 0/1 column per category)"
    ORDINAL = "Ordinal (one integer per category)"


class Scaling(str, Enum):
    NONE = "None"
    STANDARD = "Standardise (mean 0, std 1)"
    MINMAX = "Min-max (0 to 1)"


@dataclass(frozen=True)
class FeatureOptions:
    encoding: Encoding = Encoding.ONE_HOT
    scaling: Scaling = Scaling.NONE
    ratios: bool = False
    polynomial: bool = False


@dataclass
class FeatureResult:
    df: pd.DataFrame
    log: list[str] = field(default_factory=list)
    category_maps: dict[str, dict[int, str]] = field(default_factory=dict)
    scaler: StandardScaler | MinMaxScaler | None = None


def _list(cols: Iterable[str]) -> str:
    return ", ".join(f"`{c}`" for c in cols)


# ── Individual steps ───────────────────────────────────────────────────────────
def encode_booleans(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    cols = boolean_columns(df)
    if not cols:
        return df
    out = df.copy()
    out[cols] = out[cols].astype(int)
    log.append(f"Converted true/false columns to 0/1: {_list(cols)}.")
    return out


def extract_datetime_features(
    df: pd.DataFrame, log: list[str], protect: Iterable[str] = ()
) -> pd.DataFrame:
    """Replace each date column with year / month / day / weekday / quarter columns."""
    cols = datetime_columns(df, exclude=protect)
    if not cols:
        return df
    out = df.copy()
    for col in cols:
        for part in DATE_PARTS:
            out[f"{col}_{part}"] = getattr(out[col].dt, part).astype("Float64").astype(float)
    out = out.drop(columns=cols)
    log.append(f"Split dates into {', '.join(DATE_PARTS)} features: {_list(cols)}.")
    return out


def add_ratio_features(
    df: pd.DataFrame, log: list[str], protect: Iterable[str] = ()
) -> pd.DataFrame:
    """Add up to ``RATIO_MAX_FEATURES`` ``a / b`` columns from continuous numeric pairs.

    Pairs whose denominator is (near) zero for more than 30% of rows are skipped.
    """
    cols = [c for c in numeric_columns(df, exclude=protect) if not is_binary(df[c])]
    out = df.copy()
    created = []
    for numerator, denominator in combinations(cols, 2):
        denom = out[denominator]
        if (denom.abs() < 1e-9).mean() > 0.3:
            continue
        name = f"{numerator}_per_{denominator}"
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = out[numerator] / denom.where(denom.abs() >= 1e-9)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        out[name] = ratio.fillna(ratio.median() if ratio.notna().any() else 0.0)
        created.append(name)
        if len(created) >= RATIO_MAX_FEATURES:
            break
    if created:
        log.append(f"Added ratio features: {_list(created)}.")
    else:
        log.append("No ratio features added (need two continuous columns with non-zero values).")
    return out


def add_polynomial_features(
    df: pd.DataFrame, log: list[str], protect: Iterable[str] = ()
) -> pd.DataFrame:
    """Add squared and pairwise-product terms for the most variable numeric columns."""
    cols = [c for c in numeric_columns(df, exclude=protect) if not is_binary(df[c])]
    if not cols:
        log.append("No polynomial features added (no continuous columns).")
        return df
    top = df[cols].var().sort_values(ascending=False).index[:POLY_MAX_INPUTS].tolist()
    inputs = df[top].fillna(df[top].median())
    poly = PolynomialFeatures(degree=2, include_bias=False)
    values = poly.fit_transform(inputs)
    names = [
        n.replace(" ", "_x_").replace("^2", "_squared") for n in poly.get_feature_names_out(top)
    ]
    new = pd.DataFrame(values, columns=names, index=df.index).drop(columns=top)
    log.append(f"Added {new.shape[1]} polynomial features from {_list(top)}.")
    return pd.concat([df, new], axis=1)


def encode_categorical(
    df: pd.DataFrame,
    log: list[str],
    encoding: Encoding = Encoding.ONE_HOT,
    protect: Iterable[str] = (),
) -> tuple[pd.DataFrame, dict[str, dict[int, str]]]:
    """Encode text columns as numbers.

    One-hot is the safe default for unordered categories; columns with more than
    ``ONE_HOT_MAX_CATEGORIES`` values fall back to ordinal codes to avoid
    creating thousands of columns. Returns the code → label maps for ordinal columns.
    """
    cols = categorical_columns(df, exclude=protect)
    if not cols:
        log.append("No text columns to encode.")
        return df, {}

    out = df.copy()
    maps: dict[str, dict[int, str]] = {}
    one_hot = (
        []
        if encoding is Encoding.ORDINAL
        else [c for c in cols if out[c].nunique() <= ONE_HOT_MAX_CATEGORIES]
    )
    ordinal = [c for c in cols if c not in one_hot]

    for col in ordinal:
        codes, uniques = pd.factorize(out[col], sort=True)
        out[col] = codes  # missing values become -1
        maps[col] = {i: str(v) for i, v in enumerate(uniques)}
    if ordinal:
        log.append(f"Ordinal-encoded: {_list(ordinal)} (missing values -> -1).")

    if one_hot:
        n_before = out.shape[1]
        out = pd.get_dummies(out, columns=one_hot, drop_first=True, dtype=int)
        log.append(
            f"One-hot encoded {_list(one_hot)} into {out.shape[1] - n_before + len(one_hot)} columns."
        )
    if encoding is Encoding.ONE_HOT and ordinal:
        log.append(
            f"Columns with more than {ONE_HOT_MAX_CATEGORIES} categories were ordinal-encoded "
            "instead of one-hot encoded to keep the column count manageable."
        )
    return out, maps


def scale_numeric(
    df: pd.DataFrame,
    log: list[str],
    scaling: Scaling = Scaling.NONE,
    protect: Iterable[str] = (),
) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler | None]:
    """Rescale continuous numeric columns. Binary and constant columns are left alone."""
    if scaling is Scaling.NONE:
        return df, None
    cols = [
        c
        for c in numeric_columns(df, exclude=protect)
        if not is_binary(df[c]) and df[c].std(skipna=True) > 0
    ]
    if not cols:
        log.append("No continuous columns to scale.")
        return df, None
    scaler: StandardScaler | MinMaxScaler = (
        StandardScaler() if scaling is Scaling.STANDARD else MinMaxScaler()
    )
    out = df.copy()
    out[cols] = scaler.fit_transform(out[cols])
    log.append(f"Scaled {len(cols)} column(s) with {scaling.value.lower()}.")
    return out, scaler


# ── Orchestrator ───────────────────────────────────────────────────────────────
def engineer_features(
    df: pd.DataFrame, options: FeatureOptions | None = None, target: str | None = None
) -> FeatureResult:
    """Booleans → dates → ratios/polynomials → encoding → scaling.

    New features are generated from unscaled values; the target column (if
    given) is never transformed or used as an input.
    """
    options = options or FeatureOptions()
    protect = [target] if target else []
    result = FeatureResult(df=df)
    log = result.log

    out = encode_booleans(df, log)
    out = extract_datetime_features(out, log, protect=protect)
    if options.ratios:
        out = add_ratio_features(out, log, protect=protect)
    if options.polynomial:
        out = add_polynomial_features(out, log, protect=protect)
    out, result.category_maps = encode_categorical(
        out, log, encoding=options.encoding, protect=protect
    )
    out, result.scaler = scale_numeric(out, log, scaling=options.scaling, protect=protect)
    result.df = out
    return result
