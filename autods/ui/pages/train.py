"""Train & compare: fit several models, score them on unseen rows and explain the winner."""

from collections.abc import Callable
from functools import partial
from typing import TypeVar

import pandas as pd
import streamlit as st

from autods.ai.prompts import feature_importance_prompt
from autods.config import get_settings
from autods.core.modeling import (
    ModelingError,
    TaskType,
    TrainingResult,
    available_models,
    confusion,
    detect_task_type,
    export_model,
    feature_importance,
    learning_curve_data,
    plan_features,
    shap_explanation,
    train_models,
)
from autods.ui import charts, state
from autods.ui.components import ai_insight, column_list, dataset_picker, page_header, require_data

T = TypeVar("T")


def cached(key: str, compute: Callable[[], T]) -> T:
    """Compute once per trained result; slow artefacts survive widget reruns."""
    cache = state.training_cache()
    if key not in cache:
        cache[key] = compute()
    return cache[key]


page_header(
    "Train & compare models",
    "Choose what to predict. The app holds back a test set, trains several models on the rest "
    "and reports how well each one predicts rows it has never seen.",
    eyebrow="Step 4 · Model",
)
require_data()

source_name, df = dataset_picker("train_version")
settings = get_settings()

# ── Setup ──────────────────────────────────────────────────────────────────────
with st.container(border=True):
    columns = list(df.columns)
    version = state.get_version(source_name)
    remembered = version.target if version and version.target in columns else None
    target = st.selectbox(
        "Target column (what to predict)",
        columns,
        index=columns.index(remembered) if remembered else len(columns) - 1,
        key=f"target_{source_name}",
    )
    guessed = detect_task_type(df[target]) if df[target].notna().any() else TaskType.REGRESSION
    left, mid, right = st.columns(3)
    task = (
        left.segmented_control(
            "Problem type",
            list(TaskType),
            default=guessed,
            format_func=lambda t: t.value,
            key=f"task_{source_name}_{target}",
            help="Classification predicts a category (yes/no, a label). Regression predicts a "
            "number. The app guesses from the target's values; change it if the guess is wrong.",
        )
        or guessed
    )
    test_pct = mid.slider(
        "Test set size",
        10,
        40,
        20,
        step=5,
        format="%d%%",
        help="Share of rows held back for the final score.",
    )
    model_options = list(available_models(task))
    chosen = right.multiselect("Models", model_options, default=model_options)

    plan = plan_features(df, target)
    st.caption(
        f"**{len(plan.columns)} input columns** "
        f"({len(plan.numeric)} numeric, {len(plan.categorical)} categorical, {len(plan.datetime)} date)."
    )
    if plan.excluded:
        st.caption(
            "Left out automatically: "
            + "; ".join(f"`{c}` — {why}" for c, why in plan.excluded.items())
        )
    run = st.button(
        "Train models", type="primary", icon=":material/model_training:", disabled=not chosen
    )

if run:
    with st.status("Training models…", expanded=True) as status:
        st.write(f"Holding back {test_pct}% of rows as a test set.")
        st.write(
            f"Fitting {len(chosen)} model(s) with 5-fold cross-validation on the training rows."
        )
        try:
            trained = train_models(
                df,
                target,
                task=task,
                test_size=test_pct / 100,
                model_names=chosen,
                random_state=settings.random_state,
            )
        except ModelingError as exc:
            status.update(label="Training failed", state="error")
            st.error(str(exc), icon=":material/error:")
            st.stop()
        state.set_training(trained, source_name)
        status.update(
            label=f"Trained {len(trained.pipelines)} model(s)", state="complete", expanded=False
        )

result = state.training()
if result is None:
    st.stop()
trained_result: TrainingResult = result  # narrowed for use in callbacks below
if state.training_source() != source_name or result.target != target:
    st.info(
        f"Showing the last results: target **{result.target}** on the "
        f"**{state.training_source()}** data. Press *Train models* to update.",
        icon=":material/history:",
    )

# ── Results ────────────────────────────────────────────────────────────────────
metric = result.primary_metric
best = result.leaderboard.iloc[0]
st.divider()
st.subheader(f"Best model: {result.best_model}", anchor=False)
cols = st.columns(4)
for col, name in zip(
    cols, [c for c in result.leaderboard.columns if c != "Model"][:4], strict=False
):
    col.metric(f"Test {name}", f"{best[name]:.3f}")
explain = (
    "F1 balances precision (how many predicted positives were right) and recall (how many "
    "actual positives were found); 1.0 is perfect."
    if result.task is TaskType.CLASSIFICATION
    else "R² is the share of variation in the target the model explains (1.0 is perfect, 0 is "
    "no better than always predicting the average). RMSE and MAE are typical errors in the "
    "target's own units."
)
st.caption(explain)
if result.task is TaskType.CLASSIFICATION and result.class_labels and len(result.class_labels) == 2:
    st.caption(
        f"Precision, recall and F1 treat **{result.class_labels[1]}** as the positive class."
    )
for note in result.notes:
    st.caption(f":material/info: {note}")

board_tab, diag_tab, why_tab, export_tab = st.tabs(
    ["Leaderboard", "Prediction quality", "What drives predictions", "Export"]
)

with board_tab:
    st.dataframe(result.leaderboard, hide_index=True)
    st.plotly_chart(charts.metric_bar(result.leaderboard, metric))
    st.caption(
        f"Test scores come from {len(result.X_test):,} held-out rows. CV columns are the "
        "mean ± std of 5-fold cross-validation on the training rows; a big gap between the two "
        "suggests an unstable model or a lucky/unlucky split."
    )

with diag_tab:
    model_name = st.selectbox("Model", list(result.pipelines), key="diag_model")
    if result.task is TaskType.CLASSIFICATION:
        st.markdown("**Confusion matrix**: rows are the true class, columns the predicted class")
        st.plotly_chart(charts.confusion_heatmap(confusion(result, model_name)))
    else:
        st.markdown(
            "**Actual vs predicted** on the test set: points on the dashed line are perfect"
        )
        st.plotly_chart(charts.actual_vs_predicted(result.y_test, result.predictions[model_name]))
    if st.toggle(
        "Show learning curve", help="Takes a few seconds: refits the model on growing subsets."
    ):
        try:
            with st.spinner("Computing learning curve…"):
                curve = cached(
                    f"curve:{model_name}", partial(learning_curve_data, trained_result, model_name)
                )
            st.plotly_chart(charts.learning_curve(curve))
            st.caption(
                "If the validation score keeps rising with more rows, more data would help. "
                "A large, persistent gap to the train score means overfitting."
            )
        except ModelingError as exc:
            st.info(str(exc))

with why_tab:
    model_name = st.selectbox("Model", list(result.pipelines), key="why_model")
    importance = feature_importance(result.pipelines[model_name])
    if importance.empty:
        st.info("This model type does not expose feature importance.")
    else:
        label = (
            "Importance"
            if "Forest" in model_name or "XGB" in model_name
            else "|Coefficient| (scaled inputs)"
        )
        st.markdown(f"**Top features** · {label}")
        st.plotly_chart(charts.importance_bar(importance))
        ai_insight(
            feature_importance_prompt(result.target, result.task.value, importance),
            key=f"ai_importance_{model_name}",
        )

    st.markdown("**Explain a single prediction (SHAP)**")
    try:
        with st.spinner("Computing SHAP values…"):
            explanation = cached(
                f"shap:{model_name}", partial(shap_explanation, trained_result, model_name)
            )
    except ModelingError as exc:
        st.caption(str(exc))
    else:
        global_imp = (
            pd.DataFrame(
                {
                    "Feature": explanation.feature_names,
                    "Mean |SHAP|": abs(explanation.values).mean(axis=0),
                }
            )
            .sort_values("Mean |SHAP|", ascending=False)
            .head(15)
        )
        row = st.slider("Test row", 0, len(explanation.data) - 1, 0, key=f"shap_row_{model_name}")
        contrib = (
            pd.DataFrame(
                {"Feature": explanation.feature_names, "Contribution": explanation.values[row]}
            )
            .assign(size=lambda d: d["Contribution"].abs())
            .sort_values("size", ascending=False)
            .head(12)
            .drop(columns="size")
        )
        prediction = result.predictions[model_name][row]
        if result.class_labels:
            prediction = result.class_labels[int(prediction)]
        subject = (
            f"probability of **{explanation.class_label}**"
            if explanation.class_label
            else "prediction"
        )
        st.caption(
            f"Predicted: **{prediction}**. Green bars pushed the {subject} up, red bars pushed it "
            f"down, starting from the model's average output of {explanation.base_value:.3g}."
        )
        left, right = st.columns(2, gap="large")
        left.plotly_chart(charts.contribution_bar(contrib), key="shap_local")
        right.plotly_chart(charts.importance_bar(global_imp, x="Mean |SHAP|"), key="shap_global")
        right.caption(f"Overall influence across {len(explanation.data)} test rows.")

with export_tab:
    model_name = st.selectbox("Model to export", list(result.pipelines), key="export_model")
    st.download_button(
        f"Download {model_name} (.joblib)",
        data=cached(f"export:{model_name}", partial(export_model, trained_result, model_name)),
        file_name=f"{result.target}_{model_name.lower().replace(' ', '_')}.joblib",
        mime="application/octet-stream",
        icon=":material/download:",
        type="primary",
    )
    st.markdown(
        "The file contains the **whole pipeline** (gap filling, encoding, scaling and the model), "
        f"so it predicts directly from raw data with these columns: {column_list(result.plan.columns)}."
    )
    st.code(
        "import joblib, pandas as pd\n"
        "from autods.core.modeling import predict_with_bundle\n\n"
        f"bundle = joblib.load('{result.target}_model.joblib')\n"
        "new_rows = pd.read_csv('new_data.csv')\n"
        "predictions = predict_with_bundle(bundle, new_rows)",
        language="python",
    )
    st.caption(
        "Only load model files you created yourself or trust: joblib/pickle files can run code "
        "when loaded. Use the same library versions to load it."
    )
