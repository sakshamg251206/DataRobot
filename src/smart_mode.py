import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.cleaning import (
    drop_useless_columns,
    fix_numeric_strings,
    detect_datetime_columns,
    remove_duplicates,
    fill_missing_values,
    handle_outliers,
)
from src.feature_engineering import (
    encode_booleans,
    extract_datetime_features,
    auto_feature_engineering,
    encode_categorical,
    scale_numerical,
    get_column_types,
)
from src.modeling import detect_task_type


# ── Constants ──────────────────────────────────────────────────────────────────
COLLINEARITY_THRESHOLD  = 0.90   # drop one of any pair above this correlation
RF_N_ESTIMATORS         = 50     # more trees = more stable importances
RF_ZERO_IMPORTANCE_ONLY = True   # only drop features with exactly 0.0 importance
TEST_SIZE               = 0.20   # 20% held out as test set
RANDOM_STATE            = 42


# ── Readiness score ────────────────────────────────────────────────────────────
def compute_readiness_score(df: pd.DataFrame, target_col: str) -> float:
    """
    Score from 0–100 reflecting true ML readiness:
        - Penalise missing values       (up to -30)
        - Penalise duplicate rows       (up to -10)
        - Penalise non-numeric columns  (up to -30)
          (ML models need all-numeric input)
        - Penalise zero-variance cols   (up to -15)
        - Penalise >50% missing cols    (up to -15)
    """
    score = 100.0

    # Missing values
    missing_ratio = df.isnull().sum().sum() / max(df.shape[0] * df.shape[1], 1)
    score -= missing_ratio * 30

    # Duplicate rows
    dup_ratio = df.duplicated().sum() / max(df.shape[0], 1)
    score -= dup_ratio * 10

    # Non-numeric columns (excluding target)
    feature_cols = [c for c in df.columns if c != target_col]
    non_numeric  = df[feature_cols].select_dtypes(exclude="number").shape[1]
    non_num_ratio = non_numeric / max(len(feature_cols), 1)
    score -= non_num_ratio * 30

    # Zero-variance columns (excluding target)
    zero_var = sum(
        1 for c in feature_cols
        if df[c].nunique(dropna=True) <= 1
    )
    score -= (zero_var / max(len(feature_cols), 1)) * 15

    # Columns >50% missing
    high_missing = sum(
        1 for c in feature_cols
        if df[c].isnull().mean() > 0.50
    )
    score -= (high_missing / max(len(feature_cols), 1)) * 15

    return round(max(0.0, min(100.0, score)), 1)


# ── Train/test split ───────────────────────────────────────────────────────────
def split_data(
    df:         pd.DataFrame,
    target_col: str,
    logs:       list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train (80%) and test (20%) BEFORE any fitting step.
    Returns (train_df, test_df) with target column included in both.
    All encoders/scalers/RF must be fitted on train only.
    """
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logs.append(
        f"Train/test split: {len(train_df):,} train rows / "
        f"{len(test_df):,} test rows (80/20, stratification not applied here — "
        "use modeling.py for stratified splits)."
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ── Step: Smart imputation (skew-aware) ───────────────────────────────────────
def smart_impute(
    df:         pd.DataFrame,
    target_col: str,
    logs:       list,
) -> pd.DataFrame:
    """
    Per-column imputation logic:
    - Numeric: median if |skew| > 1, else mean
    - Categorical: mode
    - Target: drop rows where target is missing (never impute target)
    """
    df = df.copy()

    # Drop rows where target is missing — never impute target
    if df[target_col].isnull().any():
        n_before = len(df)
        df       = df.dropna(subset=[target_col])
        n_dropped = n_before - len(df)
        logs.append(
            f"Dropped {n_dropped:,} rows where target `{target_col}` is missing. "
            "Target values must never be imputed."
        )

    num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != target_col
    ]
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target_col
    ]

    for col in num_cols:
        if not df[col].isnull().any():
            continue
        skew = df[col].skew()
        if abs(skew) > 1:
            df[col] = df[col].fillna(df[col].median())
            logs.append(f"Imputed `{col}` with median (skew={skew:.2f}).")
        else:
            df[col] = df[col].fillna(df[col].mean())
            logs.append(f"Imputed `{col}` with mean (skew={skew:.2f}).")

    for col in cat_cols:
        if not df[col].isnull().any():
            continue
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])
            logs.append(f"Imputed categorical `{col}` with mode.")

    return df


# ── Step: Smart encoding ───────────────────────────────────────────────────────
def smart_encode(
    df:         pd.DataFrame,
    target_col: str,
    logs:       list,
) -> pd.DataFrame:
    """
    Encoding strategy chosen per column:
    - nunique <= 10  → One-Hot Encoding (nominal safe)
    - nunique  > 10  → Label Encoding   (high cardinality)
    - Boolean        → cast to int
    Target column is never encoded here (handled separately in modeling).
    """
    df = df.copy()

    # Booleans → int first
    df = encode_booleans(df, logs)

    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target_col
    ]

    ohe_cols   = [c for c in cat_cols if df[c].nunique() <= 10]
    label_cols = [c for c in cat_cols if df[c].nunique() >  10]

    if ohe_cols:
        n_before = df.shape[1]
        df       = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
        n_new    = df.shape[1] - n_before + len(ohe_cols)
        # Cast bool → int (pandas get_dummies returns bool in newer versions)
        bool_new = df.select_dtypes(include="bool").columns
        if len(bool_new):
            df[bool_new] = df[bool_new].astype(int)
        logs.append(f"One-Hot Encoded (≤10 unique): {ohe_cols} → {n_new} new columns.")

    for col in label_cols:
        le       = LabelEncoder()
        df[col]  = le.fit_transform(df[col].astype(str).fillna("__missing__"))
        logs.append(f"Label Encoded (>10 unique): `{col}`.")

    return df


# ── Step: Drop collinear features ─────────────────────────────────────────────
def drop_collinear(
    df:         pd.DataFrame,
    target_col: str,
    logs:       list,
) -> pd.DataFrame:
    """
    Drop one column from each pair with |correlation| > COLLINEARITY_THRESHOLD.
    Never drops the target column.
    Uses upper triangle of correlation matrix to avoid double-counting.
    """
    df       = df.copy()
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]

    if len(num_cols) < 3:
        return df

    corr    = df[num_cols].corr().abs()
    upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [
        col for col in upper.columns
        if any(upper[col] > COLLINEARITY_THRESHOLD) and col != target_col
    ]

    if to_drop:
        df = df.drop(columns=to_drop)
        logs.append(
            f"Dropped {len(to_drop)} collinear features "
            f"(|r| > {COLLINEARITY_THRESHOLD}): {to_drop}"
        )
    else:
        logs.append(f"No collinear features found above threshold {COLLINEARITY_THRESHOLD}.")

    return df


# ── Step: RF feature importance pruning ───────────────────────────────────────
def rf_feature_prune(
    train_df:   pd.DataFrame,
    target_col: str,
    logs:       list,
) -> list[str]:
    """
    Fit a Random Forest on TRAIN data only to identify zero-importance features.
    Returns list of column names to drop.

    ⚠️  Fitted on train set only — test set is never seen here.
        This prevents leaking test distribution into feature selection.
    """
    task_type  = detect_task_type(train_df[target_col])
    feature_cols = [
        c for c in train_df.select_dtypes(include="number").columns
        if c != target_col
    ]

    if len(feature_cols) < 3:
        logs.append("RF pruning skipped — fewer than 3 numeric features.")
        return []

    X = train_df[feature_cols].copy()
    y = train_df[target_col].copy()

    # Encode target if categorical
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

    # Final NaN guard — RF cannot handle NaN
    if X.isnull().any().any() or y.isnull().any():
        logs.append("RF pruning skipped — NaN values remain after imputation.")
        return []

    try:
        if task_type == "Classification":
            rf = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
            )

        rf.fit(X, y)
        importances    = rf.feature_importances_
        zero_imp_cols  = [feature_cols[i] for i, imp in enumerate(importances) if imp == 0.0]

        if zero_imp_cols:
            logs.append(
                f"RF feature pruning: dropped {len(zero_imp_cols)} zero-importance features: "
                f"{zero_imp_cols}"
            )
        else:
            logs.append("RF feature pruning: all features carry non-zero importance.")

        # Show top features
        importance_df = pd.DataFrame({
            "Feature":    feature_cols,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(10)
        st.write("**Top features by RF importance (train set):**")
        st.dataframe(importance_df, use_container_width=True)

        return zero_imp_cols

    except Exception as e:
        logs.append(f"RF pruning failed: {e}")
        st.warning(f"RF feature pruning failed: {e}")
        return []


# ── Main pipeline ──────────────────────────────────────────────────────────────
def smart_auto_pipeline(
    raw_df:     pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, list, float, float]:
    """
    Full smart pipeline in correct ML order:

        0.  Compute initial readiness score
        1.  Drop useless columns (empty, constant, unnamed)
        2.  Fix numeric strings ($, commas, %)
        3.  Detect and convert datetime columns
        4.  Remove duplicate rows
        5.  Drop target-missing rows / smart impute features
        6.  Handle outliers (IQR cap) — on features only
        7.  Train/test split  ← everything fitted after this point uses train only
        8.  Extract datetime features (year, month, day…)
        9.  Auto feature engineering (ratios + poly) — target excluded
        10. Smart encode categoricals (OHE ≤10, Label >10) — target excluded
        11. Drop collinear features
        12. RF feature importance pruning — fitted on train only
        13. Apply same column drops to test set
        14. Compute final readiness score
        15. Return train_df (ML-ready), logs, scores

    Returns:
        train_df    — cleaned, encoded, pruned training set
        logs        — full audit trail of every step
        init_score  — readiness score before pipeline
        final_score — readiness score after pipeline
    """
    logs       = []
    df         = raw_df.copy()

    # Validate target exists
    if target_col not in df.columns:
        st.error(f"Target column `{target_col}` not found in dataset.")
        return df, ["ERROR: target column not found."], 0.0, 0.0

    init_score = compute_readiness_score(df, target_col)
    logs.append(f"Initial readiness score: {init_score}%")

    # ── Step 1: Drop useless columns ─────────────────────────────────────────
    df = drop_useless_columns(df, logs)

    # Ensure target wasn't dropped
    if target_col not in df.columns:
        st.error(f"Target column `{target_col}` was dropped as useless. Choose a different target.")
        return raw_df, logs, init_score, 0.0

    # ── Step 2: Fix numeric strings ───────────────────────────────────────────
    df = fix_numeric_strings(df, logs)

    # ── Step 3: Detect datetimes ──────────────────────────────────────────────
    df = detect_datetime_columns(df, logs)

    # ── Step 4: Remove duplicates ─────────────────────────────────────────────
    df = remove_duplicates(df, logs)

    # ── Step 5: Impute (skew-aware, target rows dropped not imputed) ──────────
    df = smart_impute(df, target_col, logs)

    # ── Step 6: Cap outliers on features only ─────────────────────────────────
    target_series = df[target_col].copy()
    df_features   = df.drop(columns=[target_col])
    df_features   = handle_outliers(df_features, logs, method="IQR", action="Cap")
    df            = pd.concat([df_features, target_series], axis=1)
    logs.append("Outlier capping applied to feature columns only (target unchanged).")

    # ── Step 7: Train/test split ──────────────────────────────────────────────
    train_df, test_df = split_data(df, target_col, logs)

    # ── Step 8: Extract datetime features ────────────────────────────────────
    train_df = extract_datetime_features(train_df, logs)
    test_df  = extract_datetime_features(test_df,  [])   # same structure, no double-log

    # ── Step 9: Feature engineering (target excluded) ─────────────────────────
    if len(train_df) < 5000:
        train_df = auto_feature_engineering(
            train_df, poly=True, ratios=True, target=target_col, logs=logs
        )
        # Apply same columns to test (only keep cols that exist in both)
        shared_cols = [c for c in train_df.columns if c in test_df.columns or c == target_col]
    else:
        train_df = auto_feature_engineering(
            train_df, poly=False, ratios=True, target=target_col, logs=logs
        )
        logs.append("Polynomial features skipped (dataset > 5,000 rows — memory constraint).")

    # ── Step 10: Smart encode ─────────────────────────────────────────────────
    train_df = smart_encode(train_df, target_col, logs)
    test_df  = smart_encode(test_df,  target_col, [])

    # Align test columns to train (OHE can create different dummy columns)
    train_feature_cols = [c for c in train_df.columns if c != target_col]
    test_df = test_df.reindex(
        columns=train_feature_cols + [target_col], fill_value=0
    )

    # ── Step 11: Drop collinear features ──────────────────────────────────────
    train_df        = drop_collinear(train_df, target_col, logs)
    final_feat_cols = [c for c in train_df.columns if c != target_col]
    # Apply same drops to test
    test_df = test_df[[c for c in test_df.columns if c in train_df.columns]]

    # ── Step 12: RF pruning (train only) ─────────────────────────────────────
    cols_to_prune = rf_feature_prune(train_df, target_col, logs)
    if cols_to_prune:
        train_df = train_df.drop(columns=cols_to_prune, errors="ignore")
        test_df  = test_df.drop(columns=cols_to_prune, errors="ignore")

    # ── Step 13: Final NaN check ──────────────────────────────────────────────
    remaining_nan = int(train_df.isnull().sum().sum())
    if remaining_nan > 0:
        logs.append(
            f"⚠️ {remaining_nan} NaN values remain in train set after pipeline. "
            "Check datetime-derived columns or re-run with KNN imputation."
        )
    else:
        logs.append("✅ No NaN values in final train set.")

    # ── Step 14: Final score ──────────────────────────────────────────────────
    final_score = compute_readiness_score(train_df, target_col)
    logs.append(f"Final readiness score: {final_score}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    st.info(
        f"📦 Train set: **{train_df.shape[0]:,} rows × {train_df.shape[1]} columns**  \n"
        f"📦 Test set:  **{test_df.shape[0]:,} rows × {test_df.shape[1]} columns**  \n"
        "Test set is saved in `st.session_state.test_data` for evaluation in ML section."
    )

    # Save test set to session state for modeling.py to use
    st.session_state["test_data"] = test_df

    return train_df, logs, init_score, final_score