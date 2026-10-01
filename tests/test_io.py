import io

import pandas as pd
import pytest

from autods.core.io import (
    DataLoadError,
    find_quality_issues,
    load_dataset,
    read_table,
    standardize_column_names,
)


def test_reads_semicolon_delimited_csv_with_bom():
    raw = "﻿a;b\n1;2\n3;4\n".encode()
    df = read_table(raw, "data.csv")
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_reads_windows_encoded_csv():
    raw = "name,city\nJosé,Málaga\n".encode("cp1252")
    df = read_table(raw, "data.csv")
    assert df.loc[0, "city"] == "Málaga"


def test_reads_xlsx():
    buffer = io.BytesIO()
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buffer, index=False)
    df = read_table(buffer.getvalue(), "book.xlsx")
    assert df.shape == (2, 2)


@pytest.mark.parametrize(
    ("raw", "name", "message"),
    [
        (b"a,b\n1,2", "data.json", "Unsupported"),
        (b"", "data.csv", "empty"),
    ],
)
def test_read_errors_are_friendly(raw, name, message):
    with pytest.raises(DataLoadError, match=message):
        read_table(raw, name)


def test_standardize_column_names_is_unique_and_snake_case():
    df = pd.DataFrame([[1, 2, 3, 4]], columns=[" Total Sales ($)", "total sales", "%%", "Age"])
    out, renamed = standardize_column_names(df)
    assert list(out.columns) == ["total_sales", "total_sales_1", "column_3", "age"]
    assert renamed["Age"] == "age"


def test_load_dataset_drops_index_artifacts_and_blank_rows():
    raw = b"Unnamed: 0,Value,Group\n0,1,a\n1,2,b\n,,\n2,3,a\n3,4,b\n4,5,a\n"
    result = load_dataset(raw, "x.csv")
    assert list(result.df.columns) == ["value", "group"]
    assert len(result.df) == 5
    assert result.dropped_columns == ["Unnamed: 0"]


def test_load_dataset_truncates_to_max_rows():
    raw = ("a,b\n" + "".join(f"{i},{i}\n" for i in range(50))).encode()
    result = load_dataset(raw, "x.csv", max_rows=10)
    assert len(result.df) == 10
    assert "truncated" in result.warnings[0]


def test_quality_issues_flag_common_problems(messy_df):
    issues = " ".join(find_quality_issues(messy_df))
    assert "empty" in issues.lower()
    assert "single value" in issues
    assert "duplicate" in issues
