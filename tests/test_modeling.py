import io

import joblib
import numpy as np
import pandas as pd
import pytest

from autods.core.modeling import (
    ModelingError,
    TaskType,
    confusion,
    detect_task_type,
    export_model,
    feature_importance,
    learning_curve_data,
    plan_features,
    predict_with_bundle,
    shap_explanation,
    train_models,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["a", "b", "c"], TaskType.CLASSIFICATION),
        ([0, 1, 0, 1], TaskType.CLASSIFICATION),
        ([1, 2, 3, 4, 5] * 4, TaskType.CLASSIFICATION),
        ([1.0, 2.0, 3.0, 4.0, 5.0] * 4, TaskType.CLASSIFICATION),
        (list(np.linspace(0, 1, 50)), TaskType.REGRESSION),
        (list(range(100)), TaskType.REGRESSION),
    ],
)
def test_detect_task_type(values, expected):
    assert detect_task_type(pd.Series(values)) is expected


def test_plan_excludes_identifiers_and_free_text(classification_df):
    df = classification_df.copy()
    df["row_id"] = range(1, len(df) + 1)
    df["comment"] = [f"free text comment number {i} with words" for i in range(len(df))]
    plan = plan_features(df, "label")
    assert "row_id" in plan.excluded
    assert "comment" in plan.excluded
    assert set(plan.numeric) == {"x1", "x2"}
    assert plan.categorical == ["color"]
    assert plan.datetime == ["when"]


def test_classification_pipeline_end_to_end(classification_df):
    result = train_models(classification_df, "label", random_state=0)
    assert result.task is TaskType.CLASSIFICATION
    assert result.class_labels == ["no", "yes"]
    assert result.leaderboard.loc[0, "F1"] > 0.7
    assert len(result.X_test) == 60
    matrix = confusion(result)
    assert matrix.to_numpy().sum() == len(result.X_test)
    assert not feature_importance(result.best_pipeline).empty


def test_regression_pipeline_end_to_end(regression_df):
    result = train_models(regression_df, "y", random_state=0)
    assert result.task is TaskType.REGRESSION
    assert result.leaderboard["R²"].max() > 0.9
    linear = result.leaderboard.set_index("Model").loc["Linear Regression"]
    assert linear["RMSE"] < 1.5


def test_preprocessing_is_fitted_on_training_rows_only(regression_df):
    result = train_models(regression_df, "y", model_names=["Linear Regression"], random_state=0)
    scaler = (
        result.best_pipeline.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    )
    assert np.allclose(scaler.mean_, result.X_train[["x1", "x2"]].mean().to_numpy())


def test_unseen_category_at_prediction_time(classification_df):
    result = train_models(classification_df, "label", model_names=["Random Forest"])
    new = result.X_test.head(3).copy()
    new["color"] = "purple"
    assert len(result.best_pipeline.predict(new)) == 3


def test_export_bundle_roundtrip(classification_df):
    result = train_models(classification_df, "label", model_names=["Logistic Regression"])
    bundle = joblib.load(io.BytesIO(export_model(result)))
    assert bundle["target"] == "label"
    preds = predict_with_bundle(bundle, classification_df.head(5))
    assert set(preds) <= {"yes", "no"}
    with pytest.raises(ModelingError, match="missing columns"):
        predict_with_bundle(bundle, classification_df[["x1"]])


def test_shap_and_learning_curve(classification_df):
    result = train_models(classification_df, "label", model_names=["Random Forest"])
    explanation = shap_explanation(result, max_rows=20)
    assert explanation.values.shape == (20, len(explanation.feature_names))
    assert explanation.class_label == "yes"
    curve = learning_curve_data(result)
    assert list(curve.columns[:2]) == ["Training rows", "Train score"]


def test_shap_rejects_linear_models(regression_df):
    result = train_models(regression_df, "y", model_names=["Linear Regression"])
    with pytest.raises(ModelingError, match="tree models"):
        shap_explanation(result)


@pytest.mark.parametrize(
    ("frame", "target", "message"),
    [
        (pd.DataFrame({"x": range(5), "y": range(5)}), "y", "at least"),
        (pd.DataFrame({"x": range(40), "y": ["a"] * 40}), "y", "only one class"),
        (
            pd.DataFrame({"x": np.arange(100) % 7, "y": [f"c{i}" for i in range(100)]}),
            "y",
            "too many",
        ),
    ],
)
def test_friendly_errors(frame, target, message):
    with pytest.raises(ModelingError, match=message):
        train_models(frame, target)


def test_regression_requires_numeric_target(classification_df):
    with pytest.raises(ModelingError, match="numeric target"):
        train_models(classification_df, "label", task=TaskType.REGRESSION)
