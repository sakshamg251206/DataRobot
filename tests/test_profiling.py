import numpy as np
import pandas as pd

from autods.core.profiling import (
    ColumnKind,
    classify_column,
    column_profile,
    missing_report,
    overview,
    readiness_flags,
    skewed_columns,
    top_correlations,
)


def test_classify_column_kinds():
    n = 50
    assert classify_column(pd.Series(range(1, n + 1))) is ColumnKind.ID
    assert classify_column(pd.Series(["a", "b"] * 25)) is ColumnKind.BINARY
    assert classify_column(pd.Series(["a", "b", "c"] * 10)) is ColumnKind.CATEGORICAL
    assert (
        classify_column(pd.Series(pd.date_range("2024", periods=n).astype(str)))
        is ColumnKind.DATETIME
    )
    assert classify_column(pd.Series(np.linspace(0, 1, n))) is ColumnKind.CONTINUOUS
    assert classify_column(pd.Series([None, None], dtype=object)) is ColumnKind.EMPTY
    long_text = pd.Series([f"this is a longer free text comment number {i}" for i in range(n)])
    assert classify_column(long_text) is ColumnKind.HIGH_CARD_TEXT


def test_numeric_strings_are_not_dates():
    assert classify_column(pd.Series(["1", "2", "3", "4"])) is not ColumnKind.DATETIME


def test_profile_and_missing_report(messy_df):
    profile = column_profile(messy_df)
    assert set(profile["Column"]) == set(messy_df.columns)
    missing = missing_report(messy_df)
    assert missing.iloc[0]["Column"] == "empty"
    assert missing.iloc[0]["Severity"].startswith("Critical")
    assert "customer_id" not in set(missing["Column"])


def test_overview_counts(messy_df):
    stats = overview(messy_df)
    assert stats["rows"] == 7
    assert stats["duplicate_rows"] == 1


def test_readiness_flags_identifies_dates():
    df = pd.DataFrame({"d": pd.date_range("2024", periods=30).astype(str), "v": range(30)})
    flags = readiness_flags(column_profile(df))
    assert any("d" in cols for key, cols in flags.items() if key.startswith("Date"))


def test_skewed_columns_and_correlations():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "y": 2 * x + 0.01, "skewed": rng.exponential(size=200) ** 3})
    assert "skewed" in skewed_columns(df).index
    corr = top_correlations(df)
    assert {corr.iloc[0]["Feature A"], corr.iloc[0]["Feature B"]} == {"x", "y"}
    assert corr.iloc[0]["Correlation"] == 1.0
