import numpy as np
import pandas as pd

from autods.core.cleaning import (
    CleaningOptions,
    ImputeStrategy,
    OutlierAction,
    OutlierMethod,
    clean_dataset,
    fix_numeric_strings,
    handle_outliers,
    parse_datetime_columns,
)


def test_clean_dataset_end_to_end(messy_df):
    cleaned, report = clean_dataset(messy_df)
    assert "empty" not in cleaned.columns
    assert "constant" not in cleaned.columns
    assert pd.api.types.is_numeric_dtype(cleaned["revenue"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["signup_date"])
    assert report.rows_out == 6  # one duplicate removed
    assert cleaned.drop(columns=["signup_date"]).isna().sum().sum() == 0
    assert report.log
    assert not messy_df.empty and "empty" in messy_df.columns  # input untouched


def test_numeric_strings_keep_real_text():
    df = pd.DataFrame({"money": ["$1,000", "2,500", "-", "45%"], "code": ["A1", "B2", "C3", "D4"]})
    out = fix_numeric_strings(df, [])
    assert out["money"].tolist()[:2] == [1000.0, 2500.0]
    assert np.isnan(out["money"].iloc[2])
    assert out["code"].tolist() == ["A1", "B2", "C3", "D4"]


def test_mostly_text_column_is_not_converted():
    df = pd.DataFrame({"mixed": ["1", "2", "three", "four", "5"]})
    assert fix_numeric_strings(df, [])["mixed"].tolist() == ["1", "2", "three", "four", "5"]


def test_datetime_parsing_skips_free_text():
    df = pd.DataFrame({"when": ["2024-01-01", "2024-01-02"], "note": ["hello", "world"]})
    out = parse_datetime_columns(df, [])
    assert pd.api.types.is_datetime64_any_dtype(out["when"])
    assert not pd.api.types.is_datetime64_any_dtype(out["note"])


def test_outlier_cap_and_remove():
    df = pd.DataFrame({"v": [1, 2, 3, 2, 1, 2, 3, 100], "flag": [0, 1] * 4})
    capped = handle_outliers(df, [], OutlierMethod.IQR, OutlierAction.CAP)
    assert capped["v"].max() < 100
    assert len(capped) == len(df)
    removed = handle_outliers(df, [], OutlierMethod.IQR, OutlierAction.REMOVE)
    assert len(removed) == len(df) - 1
    assert capped["flag"].tolist() == df["flag"].tolist()


def test_target_is_protected():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0, 5.0], "target": [1.0, 2.0, np.nan, 4.0, 500.0]})
    cleaned, report = clean_dataset(df, target="target")
    assert len(cleaned) == 4  # row with missing target dropped, not imputed
    assert cleaned["target"].max() == 500.0  # target outliers untouched
    assert any("target" in line for line in report.log)


def test_knn_imputation_runs():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0], "b": [2.0, 4.0, 6.0, np.nan, 10.0, 12.0]}
    )
    cleaned, _ = clean_dataset(df, CleaningOptions(impute=ImputeStrategy.KNN))
    assert cleaned.isna().sum().sum() == 0


def test_day_first_dates_are_parsed_consistently():
    df = pd.DataFrame({"d": ["01/01/2022", "02/01/2022", "13/01/2022", "31/01/2022"]})
    out = parse_datetime_columns(df, [])
    assert out["d"].dt.strftime("%Y-%m-%d").tolist() == [
        "2022-01-01",
        "2022-01-02",
        "2022-01-13",
        "2022-01-31",
    ]


def test_iso_and_month_first_dates_are_unchanged():
    iso = parse_datetime_columns(pd.DataFrame({"d": ["2024-01-05", "2024-02-06"]}), [])
    assert iso["d"].dt.month.tolist() == [1, 2]
    us = parse_datetime_columns(pd.DataFrame({"d": ["01/02/2022", "12/31/2022"]}), [])
    assert us["d"].dt.strftime("%m-%d").tolist() == ["01-02", "12-31"]
