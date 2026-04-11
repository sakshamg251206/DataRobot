import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import io
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE          = 42
CV_FOLDS              = 5
CLASSIFICATION_NUNIQUE_LIMIT = 20   # numeric cols with fewer unique vals → classification


# ── Task detection ─────────────────────────────────────────────────────────────
def detect_task_type(series: pd.Series) -> str:
    """
    Determine whether to treat prediction as Classification or Regression.

    Rules (in order):
    1. Non-numeric dtype → Classification
    2. Binary (2 unique values) → Classification
    3. Integer dtype with ≤ CLASSIFICATION_NUNIQUE_LIMIT unique values → Classification
    4. Float dtype → Regression
    5. Integer with many unique values → Regression
    """
    if not pd.api.types.is_numeric_dtype(series):
        return "Classification"

    n_unique = series.nunique(dropna=True)

    if n_unique == 2:
        return "Classification"

    if pd.api.types.is_integer_dtype(series) and n_unique <= CLASSIFICATION_NUNIQUE_LIMIT:
        return "Classification"

    if pd.api.types.is_float_dtype(series):
        return "Regression"

    if n_unique <= CLASSIFICATION_NUNIQUE_LIMIT:
        return "Classification"

    return "Regression"


# ── Data preparation ───────────────────────────────────────────────────────────
def _prepare_X_y(
    df:         pd.DataFrame,
    target_col: str,
    task_type:  str,
) -> tuple[pd.DataFrame, pd.Series, LabelEncoder | None]:
    """
    Split df into X (features) and y (target).
    Encode target if Classification and non-numeric.
    Returns (X, y, label_encoder_or_None).
    """
    df = df.dropna(subset=[target_col]).copy()
    X  = df.drop(columns=[target_col])
    y  = df[target_col].copy()

    le = None
    if task_type == "Classification" and (
        y.dtype == object or str(y.dtype) == "category"
    ):
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=target_col)

    return X, y, le


def _get_train_test(
    df:         pd.DataFrame,
    X:          pd.DataFrame,
    y:          pd.Series,
    target_col: str,
    test_size:  float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Use the held-out test set from Smart Auto Mode if available.
    Otherwise perform a fresh split — and warn the user that leakage
    may have occurred if the pipeline was run manually.
    """
    test_data = st.session_state.get("test_data", None)

    if test_data is not None and target_col in test_data.columns:
        st.success(
            "✅ Using held-out test set from Smart Auto Mode. "
            "No leakage — test data was never seen during preprocessing."
        )
        # Align columns between train X and test
        test_features = [c for c in test_data.columns if c != target_col]
        train_features = X.columns.tolist()

        # Keep only common features, fill missing test dummies with 0
        test_X = test_data[test_features].reindex(columns=train_features, fill_value=0)
        test_y = test_data[target_col].copy()

        # Encode test target if needed
        if test_y.dtype == object or str(test_y.dtype) == "category":
            le_t = LabelEncoder()
            le_t.fit(y.astype(str))
            test_y = pd.Series(
                le_t.transform(test_y.astype(str)), index=test_y.index
            )

        return X, test_X, y, test_y

    else:
        st.warning(
            "⚠️ No Smart Auto Mode test set found. Performing a fresh train/test split. "
            "If you ran manual cleaning/feature engineering, preprocessing was fitted on "
            "all data — consider using Smart Auto Mode to avoid leakage."
        )
        return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


# ── Model definitions ──────────────────────────────────────────────────────────
def _get_models(task_type: str) -> dict:
    if task_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100, random_state=RANDOM_STATE,
                eval_metric="logloss", verbosity=0,
            )
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(
                n_estimators=100, random_state=RANDOM_STATE, verbosity=0
            )

    return models


# ── Metrics ────────────────────────────────────────────────────────────────────
def _classification_metrics(
    model, X_train, X_test, y_train, y_test, model_name: str
) -> dict:
    """Full classification scorecard: accuracy, F1, precision, recall + confusion matrix."""
    y_pred = model.predict(X_test)

    avg    = "binary" if len(np.unique(y_test)) == 2 else "weighted"
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average=avg, zero_division=0)
    prec   = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec    = recall_score(y_test, y_pred, average=avg, zero_division=0)

    # Cross-validation on train set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="f1_weighted")

    return {
        "Model":      model_name,
        "Accuracy":   round(acc,  4),
        "F1":         round(f1,   4),
        "Precision":  round(prec, 4),
        "Recall":     round(rec,  4),
        f"CV F1 ({CV_FOLDS}-fold)": round(cv_scores.mean(), 4),
        "CV Std":     round(cv_scores.std(),  4),
        "_y_pred":    y_pred,
    }


def _regression_metrics(
    model, X_train, X_test, y_train, y_test, model_name: str
) -> dict:
    """Full regression scorecard: R², RMSE, MAE + CV R²."""
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="r2")

    return {
        "Model":               model_name,
        "R²":                  round(r2,   4),
        "RMSE":                round(rmse, 4),
        "MAE":                 round(mae,  4),
        f"CV R² ({CV_FOLDS}-fold)": round(cv_scores.mean(), 4),
        "CV Std":              round(cv_scores.std(),  4),
        "_y_pred":             y_pred,
    }


# ── Model comparison ───────────────────────────────────────────────────────────
def run_model_comparison(
    task_type: str,
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series,    y_test:  pd.Series,
) -> tuple[object, str, list[dict]]:
    """
    Train all models, compute full metrics, display leaderboard.
    Returns (best_model, best_model_name, all_results).
    """
    models  = _get_models(task_type)
    results = []
    fitted  = {}

    progress = st.progress(0, text="Training models...")
    n        = len(models)

    for i, (name, model) in enumerate(models.items()):
        with st.spinner(f"Training {name}..."):
            model.fit(X_train, y_train)
            fitted[name] = model

            if task_type == "Classification":
                row = _classification_metrics(model, X_train, X_test, y_train, y_test, name)
            else:
                row = _regression_metrics(model, X_train, X_test, y_train, y_test, name)

            results.append(row)
        progress.progress((i + 1) / n, text=f"Trained {name}")

    progress.empty()

    # Sort by primary metric
    sort_key = "F1" if task_type == "Classification" else "R²"
    results_sorted = sorted(results, key=lambda r: r[sort_key], reverse=True)

    # Display table (hide internal _y_pred column)
    display_cols = [k for k in results_sorted[0].keys() if not k.startswith("_")]
    results_df   = pd.DataFrame(results_sorted)[display_cols]
    st.dataframe(results_df, use_container_width=True)

    # Bar chart
    fig = px.bar(
        results_df, x=sort_key, y="Model", orientation="h",
        color=sort_key, color_continuous_scale="Blues",
        title=f"Model comparison — {sort_key}",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    best_name  = results_sorted[0]["Model"]
    best_model = fitted[best_name]

    st.success(
        f"🏆 Best model: **{best_name}** "
        f"— {sort_key}: **{results_sorted[0][sort_key]:.4f}** "
        f"(CV {results_sorted[0][f'CV {sort_key} ({CV_FOLDS}-fold)']:.4f} ± "
        f"{results_sorted[0]['CV Std']:.4f})"
    )

    # Confusion matrix for classification
    if task_type == "Classification":
        best_result = next(r for r in results if r["Model"] == best_name)
        y_pred      = best_result["_y_pred"]
        cm          = confusion_matrix(y_test, y_pred)
        fig_cm, ax  = plt.subplots(figsize=(6, 4))
        disp        = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion matrix — {best_name}")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # Actual vs predicted for regression
    if task_type == "Regression":
        best_result = next(r for r in results if r["Model"] == best_name)
        y_pred      = best_result["_y_pred"]
        fig_ap = px.scatter(
            x=y_test, y=y_pred,
            labels={"x": "Actual", "y": "Predicted"},
            title=f"Actual vs Predicted — {best_name}",
            opacity=0.6,
        )
        fig_ap.add_shape(
            type="line",
            x0=float(y_test.min()), y0=float(y_test.min()),
            x1=float(y_test.max()), y1=float(y_test.max()),
            line=dict(color="red", dash="dash"),
        )
        st.plotly_chart(fig_ap, use_container_width=True)

    return best_model, best_name, results


# ── Feature importance ─────────────────────────────────────────────────────────
def explain_features(
    model, feature_names: list, target_col: str, task_type: str
):
    st.subheader("Feature Importance")

    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        coef = model.coef_
        # Multiclass: coef_ is (n_classes, n_features) → mean across classes
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)

    if importances is None:
        st.info("Feature importance not available for this model type.")
        return

    feat_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    fig = px.bar(
        feat_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        title="Top 15 most important features",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # AI explanation
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.info("Enter your Google Gemini API key in the sidebar for an AI explanation of these features.")
        return

    if not LANGCHAIN_AVAILABLE:
        st.info("Install `langchain-google-genai` for AI feature explanations.")
        return

    with st.spinner("Generating AI explanation..."):
        try:
            llm    = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            prompt = (
                f"You are an expert Data Scientist. I trained a {task_type} model to predict "
                f"'{target_col}'. The top features by importance are: "
                f"{', '.join(feat_df['Feature'].tolist())}. "
                "In 2-3 concise paragraphs, explain to a business user why these features "
                "likely matter for this prediction. Do not mention code or model internals."
            )
            response = llm.invoke(prompt)
            st.markdown(f"**🤖 AI Insights:**\n\n{response.content}")
        except Exception as e:
            st.error(f"AI explanation failed: {e}")


# ── SHAP explainability ────────────────────────────────────────────────────────
def explain_shap(model, X_train: pd.DataFrame, X_test: pd.DataFrame):
    st.subheader("SHAP Explainability")
    st.info(
        "SHAP shows exactly how much each feature pushed a single prediction "
        "above or below the model's baseline."
    )

    if not hasattr(model, "feature_importances_"):
        st.warning("SHAP waterfall plots are only supported for tree-based models (Random Forest, XGBoost).")
        return

    # Let user pick which instance to explain
    instance_idx = st.number_input(
        "Explain prediction for row index (test set):",
        min_value=0, max_value=len(X_test) - 1, value=0, step=1,
    )

    with st.spinner("Computing SHAP values..."):
        try:
            explainer   = shap.TreeExplainer(model)
            instance    = X_test.iloc[[instance_idx]]
            shap_values = explainer.shap_values(instance)

            # Handle binary classification (list of 2 arrays) and multiclass
            if isinstance(shap_values, list):
                # Use positive class for binary; mean across classes for multiclass
                sv = shap_values[1] if len(shap_values) == 2 else np.mean(
                    np.abs(shap_values), axis=0
                )
            else:
                sv = shap_values

            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1] if len(base_val) == 2 else float(np.mean(base_val))

            explanation = shap.Explanation(
                values=sv[0],
                base_values=float(base_val),
                data=instance.iloc[0].values,
                feature_names=X_test.columns.tolist(),
            )

            fig, ax = plt.subplots(figsize=(9, 4))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
            plt.close("all")

        except Exception as e:
            st.error(f"SHAP rendering failed: {e}")


# ── Learning curve ─────────────────────────────────────────────────────────────
def plot_learning_curve(model, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Learning curve computed on TRAIN set only via cross-validation.
    Never uses test data.
    """
    st.subheader("Learning Curves")
    st.info(
        "A widening gap between train and CV score = overfitting. "
        "Both scores low = underfitting."
    )

    with st.spinner("Computing learning curves (this may take a moment)..."):
        train_sizes, train_scores, cv_scores = learning_curve(
            model, X_train, y_train,
            cv=CV_FOLDS, n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 6),
            error_score="raise",
        )

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores,  axis=1)
    cv_mean    = np.mean(cv_scores,    axis=1)
    cv_std     = np.std(cv_scores,     axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        name="Train score", mode="lines+markers",
        line=dict(color="#1d9e75"),
        error_y=dict(type="data", array=train_std, visible=True),
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=cv_mean,
        name=f"CV score ({CV_FOLDS}-fold)", mode="lines+markers",
        line=dict(color="#e24b4a"),
        error_y=dict(type="data", array=cv_std, visible=True),
    ))
    fig.update_layout(
        title="Learning curve",
        xaxis_title="Training examples",
        yaxis_title="Score",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Model export ───────────────────────────────────────────────────────────────
def export_model(model, model_name: str):
    """Let the user download the fitted model as a pickle file."""
    st.subheader("Export Model")
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)

    st.download_button(
        label=f"📥 Download {model_name} (.pkl)",
        data=buffer,
        file_name=f"{model_name.lower().replace(' ', '_')}_model.pkl",
        mime="application/octet-stream",
    )
    st.caption(
        "Load with: `import pickle; model = pickle.load(open('model.pkl', 'rb'))`  \n"
        "Apply to new data: `model.predict(X_new)` — ensure X_new uses the same "
        "features and scaling as the training data."
    )


# ── Main entry point ───────────────────────────────────────────────────────────
def prepare_modeling(df: pd.DataFrame):
    st.subheader("Machine Learning")

    target_col = st.selectbox("Select target column", df.columns.tolist())
    if not target_col:
        return

    # Validate all features are numeric
    feature_cols     = [c for c in df.columns if c != target_col]
    non_numeric_cols = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric_cols:
        st.error(
            f"Non-numeric feature columns detected: `{'`, `'.join(non_numeric_cols)}`. "
            "Please encode them in Feature Engineering or run Smart Auto Mode first."
        )
        return

    task_type = detect_task_type(df[target_col])
    st.info(f"Detected task type: **{task_type}**")

    if task_type == "Classification":
        n_classes = df[target_col].nunique()
        st.write(f"Classes: **{n_classes}** unique values → {df[target_col].unique().tolist()[:10]}")

    # Prepare X, y
    X, y, le = _prepare_X_y(df, target_col, task_type)

    # Train/test split (uses Smart Mode held-out set if available)
    test_size = st.slider(
        "Test set size (%) — only used if no Smart Auto Mode split exists",
        min_value=10, max_value=40, value=20, step=5
    ) / 100.0

    X_train, X_test, y_train, y_test = _get_train_test(
        df, X, y, target_col, test_size
    )

    st.write(
        f"Train: **{len(X_train):,} rows** | "
        f"Test: **{len(X_test):,} rows** | "
        f"Features: **{X_train.shape[1]}**"
    )

    if st.button("🚀 Train & Compare Models", type="primary"):
        with st.spinner("Training..."):
            best_model, best_name, all_results = run_model_comparison(
                task_type, X_train, X_test, y_train, y_test
            )

        st.divider()
        explain_features(best_model, X_train.columns.tolist(), target_col, task_type)

        st.divider()
        plot_learning_curve(best_model, X_train, y_train)

        st.divider()
        explain_shap(best_model, X_train, X_test)

        st.divider()
        export_model(best_model, best_name)

        # Save best model to session state for report generator
        st.session_state["best_model"]      = best_model
        st.session_state["best_model_name"] = best_name
        st.session_state["model_results"]   = all_results
        st.session_state["feature_cols"]    = X_train.columns.tolist()
        st.session_state["task_type"]       = task_type