import pandas as pd

from autods.core.features import (
    Encoding,
    FeatureOptions,
    Scaling,
    encode_categorical,
    engineer_features,
)


def test_engineer_features_produces_numeric_output(classification_df):
    result = engineer_features(
        classification_df,
        FeatureOptions(scaling=Scaling.STANDARD, ratios=True, polynomial=True),
        target="label",
    )
    features = result.df.drop(columns=["label"])
    assert all(pd.api.types.is_numeric_dtype(features[c]) for c in features.columns)
    assert "when" not in result.df.columns and "when_year" in result.df.columns
    assert result.df["label"].tolist() == classification_df["label"].tolist()
    assert result.scaler is not None
    assert any("ratio" in line for line in result.log)


def test_ordinal_encoding_records_mapping():
    df = pd.DataFrame({"size": ["s", "m", "l", "m", None]})
    out, maps = encode_categorical(df, [], Encoding.ORDINAL)
    assert out["size"].tolist() == [2, 1, 0, 1, -1]
    assert maps["size"] == {0: "l", 1: "m", 2: "s"}


def test_high_cardinality_falls_back_to_ordinal():
    df = pd.DataFrame({"id": [f"user{i}" for i in range(80)]})
    log: list[str] = []
    out, maps = encode_categorical(df, log, Encoding.ONE_HOT)
    assert out.shape[1] == 1
    assert "id" in maps
    assert any("ordinal" in line.lower() for line in log)
