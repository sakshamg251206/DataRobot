import pandas as pd
import pytest

from autods.core.smart import readiness_score, smart_prepare


def test_smart_prepare_makes_numeric_table(messy_df, classification_df):
    df = classification_df.copy()
    df.loc[::17, "color"] = None
    df["row_id"] = range(1, len(df) + 1)
    df["x1_copy"] = df["x1"] * 2
    result = smart_prepare(df, "label")
    features = result.df.drop(columns=["label"])
    assert all(pd.api.types.is_numeric_dtype(features[c]) for c in features.columns)
    assert features.isna().sum().sum() == 0
    assert "row_id" not in result.df.columns
    assert not ({"x1", "x1_copy"} <= set(result.df.columns))  # collinear pair reduced
    assert result.df["label"].tolist()[:5] == df["label"].tolist()[:5]
    assert result.score_after >= result.score_before


def test_smart_prepare_validates_target(messy_df):
    with pytest.raises(ValueError, match="not found"):
        smart_prepare(messy_df, "nope")
    with pytest.raises(ValueError, match="no values"):
        smart_prepare(messy_df, "empty")


def test_readiness_score_bounds(messy_df):
    assert 0 <= readiness_score(messy_df) <= 100
    assert readiness_score(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})) == 100.0
    assert readiness_score(pd.DataFrame()) == 0.0
