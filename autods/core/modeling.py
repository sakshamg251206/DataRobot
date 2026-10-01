"""Model training, evaluation and explanation.

Every model is a scikit-learn :class:`~sklearn.pipeline.Pipeline` of
``preprocessing -> estimator``. The pipeline is fitted on the training split
only, so imputation statistics, category vocabularies and scaling parameters
never see the test rows. It also means the exported model accepts the same raw
columns the user uploaded, not a pre-encoded matrix.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.model_selection import learning_curve as sk_learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from autods.core.dtypes import (
    boolean_columns,
    categorical_columns,
    datetime_columns,
    numeric_columns,
)
from autods.core.profiling import ColumnKind, classify_column

try:  # Optional dependency
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    XGBOOST_AVAILABLE = False

MIN_TRAINING_ROWS = 20
MAX_CLASSES = 50
CLASSIFICATION_MAX_UNIQUE = 20
ONE_HOT_MAX_CATEGORIES = 20
MAX_TEXT_CARDINALITY = 100
MISSING_TOKEN = "__missing__"


class TaskType(str, Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"


class ModelingError(ValueError):
    """Raised when the data cannot be modelled; the message is shown to the user."""


# ── Task detection ─────────────────────────────────────────────────────────────
def detect_task_type(target: pd.Series) -> TaskType:
    """Guess whether predicting ``target`` is classification or regression.

    Text, boolean and two-valued targets are classification; floats are
    regression; integers with few distinct values (e.g. ratings) are
    classification. The UI lets the user override the guess.
    """
    values = target.dropna()
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return TaskType.CLASSIFICATION
    n_unique = values.nunique()
    if n_unique <= 2:
        return TaskType.CLASSIFICATION
    is_whole = pd.api.types.is_integer_dtype(values) or bool(
        np.all(np.mod(values.to_numpy(dtype=float), 1) == 0)
    )
    if is_whole and n_unique <= CLASSIFICATION_MAX_UNIQUE:
        return TaskType.CLASSIFICATION
    return TaskType.REGRESSION


# ── Custom transformers (module level so fitted pipelines can be pickled) ──────
class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Expand date columns into numeric year / month / day / weekday columns."""

    parts = ("year", "month", "day", "dayofweek")

    def fit(self, X: pd.DataFrame, y: Any = None) -> DatetimeFeatures:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in X.columns:
            parsed = pd.to_datetime(X[col], errors="coerce", format="mixed")
            for part in self.parts:
                out[f"{col}_{part}"] = getattr(parsed.dt, part).astype(float)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        cols = self.feature_names_in_ if input_features is None else input_features
        return np.asarray([f"{c}_{p}" for c in cols for p in self.parts], dtype=object)


class CategoricalToString(BaseEstimator, TransformerMixin):
    """Normalise categorical inputs to plain strings with an explicit missing token.

    Avoids the many ways pandas represents missing text (``None``, ``NaN``,
    ``pd.NA``) and mixed-type columns tripping up the one-hot encoder.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> CategoricalToString:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).astype(object)
        return frame.where(frame.notna(), MISSING_TOKEN).astype(str)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


# ── Feature selection ──────────────────────────────────────────────────────────
@dataclass
class FeaturePlan:
    """Which columns feed the model, and why any were left out."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical + self.datetime


def plan_features(df: pd.DataFrame, target: str) -> FeaturePlan:
    """Decide how each column is used. Identifiers and free text are excluded."""
    plan = FeaturePlan()
    dates, numbers = set(datetime_columns(df)), set(numeric_columns(df))
    texts = set(boolean_columns(df)) | set(categorical_columns(df))
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        kind = classify_column(s)
        if kind is ColumnKind.EMPTY:
            plan.excluded[col] = "empty"
        elif kind is ColumnKind.ID:
            plan.excluded[col] = "looks like an identifier"
        elif col in dates or kind is ColumnKind.DATETIME:
            plan.datetime.append(col)
        elif col in numbers:
            if s.nunique(dropna=True) <= 1:
                plan.excluded[col] = "constant"
            else:
                plan.numeric.append(col)
        elif col in texts:
            n_unique = s.nunique(dropna=True)
            if n_unique <= 1:
                plan.excluded[col] = "constant"
            elif kind is ColumnKind.HIGH_CARD_TEXT or n_unique > MAX_TEXT_CARDINALITY:
                plan.excluded[col] = f"free text / too many categories ({n_unique})"
            else:
                plan.categorical.append(col)
        else:
            plan.excluded[col] = f"unsupported type ({s.dtype})"
    return plan


def build_preprocessor(plan: FeaturePlan) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if plan.numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                plan.numeric,
            )
        )
    if plan.categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("to_str", CategoricalToString()),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                max_categories=ONE_HOT_MAX_CATEGORIES,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                plan.categorical,
            )
        )
    if plan.datetime:
        transformers.append(
            (
                "date",
                Pipeline(
                    [
                        ("expand", DatetimeFeatures()),
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ]
                ),
                plan.datetime,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# ── Models ─────────────────────────────────────────────────────────────────────
def available_models(task: TaskType, random_state: int = 42) -> dict[str, BaseEstimator]:
    """The candidate estimators: a linear baseline, a random forest and (if installed) XGBoost."""
    models: dict[str, BaseEstimator]
    if task is TaskType.CLASSIFICATION:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, random_state=random_state, n_jobs=-1
            ),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200, random_state=random_state, verbosity=0, n_jobs=-1
            )
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, random_state=random_state, n_jobs=-1
            ),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(
                n_estimators=200, random_state=random_state, verbosity=0, n_jobs=-1
            )
    return models


PRIMARY_METRIC = {TaskType.CLASSIFICATION: "F1", TaskType.REGRESSION: "R²"}


@dataclass
class TrainingResult:
    task: TaskType
    target: str
    plan: FeaturePlan
    leaderboard: pd.DataFrame
    pipelines: dict[str, Pipeline]
    best_model: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: dict[str, np.ndarray]
    class_labels: list[str] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def best_pipeline(self) -> Pipeline:
        return self.pipelines[self.best_model]

    @property
    def primary_metric(self) -> str:
        return PRIMARY_METRIC[self.task]


def _prepare_target(
    y: pd.Series, task: TaskType
) -> tuple[pd.Series, list[str] | None, LabelEncoder | None]:
    if task is TaskType.REGRESSION:
        if not pd.api.types.is_numeric_dtype(y):
            raise ModelingError(
                "Regression needs a numeric target. Choose classification instead, "
                "or clean the target column first."
            )
        return y.astype(float), None, None
    encoder = LabelEncoder()
    encoded = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index, name=y.name)
    return encoded, [str(c) for c in encoder.classes_], encoder


def _cv_splitter(task: TaskType, y_train: pd.Series, folds: int, seed: int) -> Any:
    if task is TaskType.CLASSIFICATION:
        smallest_class = int(y_train.value_counts().min())
        n_splits = min(folds, smallest_class)
        if n_splits < 2:
            return None
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_splits = min(folds, len(y_train) // 5)
    if n_splits < 2:
        return None
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _score(
    task: TaskType, y_true: pd.Series, y_pred: np.ndarray, n_classes: int
) -> dict[str, float]:
    if task is TaskType.CLASSIFICATION:
        average = "binary" if n_classes == 2 else "weighted"
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
            "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        }
    return {
        "R²": r2_score(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def train_models(
    df: pd.DataFrame,
    target: str,
    task: TaskType | None = None,
    test_size: float = 0.2,
    model_names: list[str] | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> TrainingResult:
    """Split, fit every candidate pipeline on the training rows and score it on the test rows.

    Raises :class:`ModelingError` with a user-facing explanation when the data
    cannot be modelled (too few rows, no usable features, too many classes, …).
    """
    if target not in df.columns:
        raise ModelingError(f"Target column `{target}` not found.")
    data = df.dropna(subset=[target])
    dropped_rows = len(df) - len(data)
    if len(data) < MIN_TRAINING_ROWS:
        raise ModelingError(
            f"Need at least {MIN_TRAINING_ROWS} rows with a target value to train a model "
            f"(found {len(data)})."
        )

    task = task or detect_task_type(data[target])
    notes: list[str] = []
    if dropped_rows:
        notes.append(f"Ignored {dropped_rows:,} row(s) with no target value.")

    if task is TaskType.CLASSIFICATION:
        n_classes = data[target].nunique()
        if n_classes < 2:
            raise ModelingError("The target has only one class, so there is nothing to predict.")
        if n_classes > MAX_CLASSES:
            raise ModelingError(
                f"The target has {n_classes} distinct values — too many for classification. "
                "If it is a continuous quantity, switch the task to regression."
            )

    plan = plan_features(data, target)
    if not plan.columns:
        raise ModelingError(
            "No usable feature columns remain after excluding identifiers and text."
        )

    X = data[plan.columns]
    y, class_labels, _ = _prepare_target(data[target], task)

    stratify = None
    if task is TaskType.CLASSIFICATION and y.value_counts().min() >= 2:
        stratify = y
    elif task is TaskType.CLASSIFICATION:
        notes.append("Some classes have a single example, so the split is not stratified.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    candidates = available_models(task, random_state)
    if model_names:
        candidates = {k: v for k, v in candidates.items() if k in model_names}
    if not candidates:
        raise ModelingError("Select at least one model to train.")

    splitter = _cv_splitter(task, y_train, cv_folds, random_state)
    cv_metric = "f1_weighted" if task is TaskType.CLASSIFICATION else "r2"
    n_classes = len(class_labels) if class_labels else 0

    rows, pipelines, predictions = [], {}, {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("prep", build_preprocessor(plan)), ("model", clone(estimator))])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                cv = (
                    cross_val_score(
                        clone(pipeline), X_train, y_train, cv=splitter, scoring=cv_metric
                    )
                    if splitter is not None
                    else np.array([np.nan])
                )
        except Exception as exc:  # one failing model should not sink the comparison
            notes.append(f"{name} failed to train: {exc}")
            continue
        row: dict[str, Any] = {"Model": name, **_score(task, y_test, y_pred, n_classes)}
        row[f"CV {PRIMARY_METRIC[task]}"] = float(np.nanmean(cv))
        row["CV ±"] = float(np.nanstd(cv))
        rows.append(row)
        pipelines[name] = pipeline
        predictions[name] = np.asarray(y_pred)

    if not rows:
        raise ModelingError("Every model failed to train. " + " ".join(notes))
    if splitter is None:
        notes.append("Too little data per class/fold for cross-validation; CV scores are blank.")

    primary = PRIMARY_METRIC[task]
    leaderboard = (
        pd.DataFrame(rows).sort_values(primary, ascending=False).reset_index(drop=True).round(4)
    )
    return TrainingResult(
        task=task,
        target=target,
        plan=plan,
        leaderboard=leaderboard,
        pipelines=pipelines,
        best_model=str(leaderboard.loc[0, "Model"]),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        class_labels=class_labels,
        notes=notes,
    )


# ── Diagnostics & explanations ─────────────────────────────────────────────────
def confusion(result: TrainingResult, model: str | None = None) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows = actual, columns = predicted)."""
    model = model or result.best_model
    labels = result.class_labels or []
    matrix = confusion_matrix(result.y_test, result.predictions[model], labels=range(len(labels)))
    return pd.DataFrame(matrix, index=labels, columns=labels)


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    return [str(n) for n in pipeline.named_steps["prep"].get_feature_names_out()]


def feature_importance(pipeline: Pipeline, top: int = 15) -> pd.DataFrame:
    """Model-native importance: tree importances or absolute linear coefficients."""
    model = pipeline.named_steps["model"]
    names = transformed_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    frame = pd.DataFrame({"Feature": names[: len(values)], "Importance": values[: len(names)]})
    return frame.sort_values("Importance", ascending=False).head(top).reset_index(drop=True)


@dataclass
class ShapExplanation:
    values: np.ndarray  # (rows, features)
    base_value: float
    feature_names: list[str]
    data: pd.DataFrame  # transformed feature values for the explained rows
    class_label: str | None = None


def shap_explanation(
    result: TrainingResult, model: str | None = None, max_rows: int = 200
) -> ShapExplanation:
    """SHAP values for tree models on (a sample of) the test set.

    For classification the explanation is for the positive / last class
    (binary) or the predicted-most class (multiclass) to keep it readable.
    Raises ``ModelingError`` for unsupported models or a missing ``shap`` install.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ModelingError("Install the `shap` package to enable SHAP explanations.") from exc

    model = model or result.best_model
    pipeline = result.pipelines[model]
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        raise ModelingError(
            "SHAP explanations are available for tree models (Random Forest, XGBoost)."
        )

    sample = result.X_test.head(max_rows)
    names = transformed_feature_names(pipeline)
    transformed = pd.DataFrame(
        np.asarray(pipeline.named_steps["prep"].transform(sample), dtype=float),
        columns=names,
        index=sample.index,
    )
    explainer = shap.TreeExplainer(estimator)
    raw = explainer.shap_values(transformed)
    base = np.atleast_1d(explainer.expected_value)

    class_label = None
    values = np.asarray(raw)
    if isinstance(raw, list):  # older shap: one array per class
        values = np.stack(raw, axis=-1)
    if values.ndim == 3:  # (rows, features, classes)
        labels = result.class_labels or []
        class_idx = (
            1 if values.shape[2] == 2 else int(np.bincount(result.predictions[model]).argmax())
        )
        values = values[:, :, class_idx]
        base_value = float(base[class_idx]) if len(base) > class_idx else float(base[0])
        class_label = labels[class_idx] if class_idx < len(labels) else None
    else:
        base_value = float(base[-1]) if len(base) else 0.0
        if result.task is TaskType.CLASSIFICATION and result.class_labels:
            class_label = result.class_labels[-1]
    return ShapExplanation(
        values=values,
        base_value=base_value,
        feature_names=names,
        data=transformed,
        class_label=class_label,
    )


def learning_curve_data(result: TrainingResult, model: str | None = None) -> pd.DataFrame:
    """Train vs cross-validated score for growing training-set sizes."""
    model = model or result.best_model
    splitter = _cv_splitter(result.task, result.y_train, 5, 42)
    if splitter is None:
        raise ModelingError("Not enough data per fold to compute a learning curve.")
    scoring = "f1_weighted" if result.task is TaskType.CLASSIFICATION else "r2"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sizes, train_scores, cv_scores = sk_learning_curve(
            clone(result.pipelines[model]),
            result.X_train,
            result.y_train,
            cv=splitter,
            scoring=scoring,
            train_sizes=np.linspace(0.2, 1.0, 5),
            n_jobs=-1,
        )
    return pd.DataFrame(
        {
            "Training rows": sizes,
            "Train score": train_scores.mean(axis=1),
            "Train std": train_scores.std(axis=1),
            "Validation score": cv_scores.mean(axis=1),
            "Validation std": cv_scores.std(axis=1),
        }
    )


# ── Export ─────────────────────────────────────────────────────────────────────
def export_model(result: TrainingResult, model: str | None = None) -> bytes:
    """Serialise a self-describing model bundle with joblib.

    The bundle is a dict with the fitted pipeline plus everything needed to use
    it correctly: input columns, target name, task, class labels and metrics.
    """
    model = model or result.best_model
    metrics = result.leaderboard.set_index("Model").loc[model].to_dict()
    bundle = {
        "pipeline": result.pipelines[model],
        "model_name": model,
        "task": result.task.value,
        "target": result.target,
        "feature_columns": result.plan.columns,
        "class_labels": result.class_labels,
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    return buffer.getvalue()


def predict_with_bundle(bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    """Predict with a bundle created by :func:`export_model` (maps classes back to labels)."""
    missing = [c for c in bundle["feature_columns"] if c not in df.columns]
    if missing:
        raise ModelingError(f"Input is missing columns: {', '.join(missing)}")
    preds = bundle["pipeline"].predict(df[bundle["feature_columns"]])
    labels = bundle.get("class_labels")
    if labels:
        return np.asarray([labels[int(p)] for p in preds])
    return np.asarray(preds)
