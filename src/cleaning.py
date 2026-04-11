import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


# ── Constants ──────────────────────────────────────────────────────────────────
DATE_SAMPLE_SIZE       = 50     # rows sampled for datetime detection
DATE_PARSE_THRESHOLD   = 0.7    # 70% of sample must parse as date to convert
ZSCORE_THRESHOLD       = 3.0    # standard deviations for Z-score outlier bound
IQR_MULTIPLIER         = 1.5    # IQR fence multiplier


# ── Logging helper ─────────────────────────────────────────────────────────────
def _log(logs: list, msg: str):
    logs.append(msg)


# ── Step 1: Drop useless columns ───────────────────────────────────────────────
def drop_useless_columns(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    Drop:
    - Fully empty columns (100% missing)
    - Constant columns (zero variance — single unique value)
    - Unnamed index artifact columns
    These carry zero signal for ML.
    """
    df = df.copy()
    dropped = []

    # Fully empty
    fully_empty = df.columns[df.isnull().all()].tolist()
    if fully_empty:
        df.drop(columns=fully_empty, inplace=True)
        dropped.extend(fully_empty)
        _log(logs, f"Dropped fully empty columns: {fully_empty}")

    # Constant (single unique non-null value)
    constant = [
        c for c in df.columns
        if df[c].nunique(dropna=True) <= 1
    ]
    if constant:
        df.drop(columns=constant, inplace=True)
        dropped.extend(constant)
        _log(logs, f"Dropped constant/zero-variance columns: {constant}")

    # Unnamed index artifacts
    unnamed = [c for c in df.columns if str(c).startswith("unnamed")]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)
        dropped.extend(unnamed)
        _log(logs, f"Dropped unnamed index columns: {unnamed}")

    if not dropped:
        _log(logs, "No useless columns found to drop.")

    return df


# ── Step 2: Fix numeric columns stored as strings ──────────────────────────────
def fix_numeric_strings(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    Columns stored as object that are actually numeric get converted.
    Handles common formatting issues:
    - Thousands separators: "1,234" → 1234
    - Currency symbols:     "$45.00" → 45.0
    - Percentage signs:     "78%" → 78.0
    - Whitespace padding:   " 42 " → 42
    """
    df      = df.copy()
    fixed   = []

    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(100)
        if sample.empty:
            continue

        cleaned = (
            sample.astype(str)
            .str.strip()
            .str.replace(r"[\$,£€%\s]", "", regex=True)
            .str.replace(r"^-$", "0", regex=True)
        )

        try:
            pd.to_numeric(cleaned, errors="raise")
            # Full column conversion
            df[col] = (
                df[col].astype(str)
                .str.strip()
                .str.replace(r"[\$,£€%\s]", "", regex=True)
                .str.replace(r"^-$", "0", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
            fixed.append(col)
        except (ValueError, TypeError):
            pass

    if fixed:
        _log(logs, f"Converted numeric-string columns to numeric dtype: {fixed}")
    else:
        _log(logs, "No numeric-string columns found.")

    return df


# ── Step 3: Detect and convert datetime columns ────────────────────────────────
def detect_datetime_columns(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    For each object column, sample up to DATE_SAMPLE_SIZE non-null values.
    Convert to datetime only if ≥ DATE_PARSE_THRESHOLD fraction parse cleanly.
    Logs exactly which columns were converted.
    """
    df           = df.copy()
    converted    = []
    skipped      = []

    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(DATE_SAMPLE_SIZE)
        if sample.empty:
            continue

        try:
            parsed      = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            parse_rate  = parsed.notna().mean()

            if parse_rate >= DATE_PARSE_THRESHOLD:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                converted.append(col)
            else:
                skipped.append(col)

        except Exception:
            skipped.append(col)

    if converted:
        _log(logs, f"Converted to datetime (≥{int(DATE_PARSE_THRESHOLD*100)}% parse rate): {converted}")
    else:
        _log(logs, "No datetime columns detected.")

    return df


# ── Step 4: Remove duplicates ──────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    n_before = len(df)
    df       = df.drop_duplicates()
    n_removed = n_before - len(df)

    if n_removed > 0:
        _log(logs, f"Removed {n_removed:,} duplicate rows ({n_removed/n_before*100:.1f}% of data).")
    else:
        _log(logs, "No duplicate rows found.")

    return df


# ── Step 5: Impute missing values ──────────────────────────────────────────────
def fill_missing_values(
    df:           pd.DataFrame,
    logs:         list,
    num_strategy: str = "Median",
    cat_strategy: str = "Mode",
) -> pd.DataFrame:
    """
    Impute missing values.

    ⚠️  KNN Imputation warning: fitted on the full dataset here (no train/test split).
        In a production ML pipeline, fit the imputer on train set only and
        transform both train and test separately to avoid data leakage.
    """
    df       = df.copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number", "datetime64[ns]"]).columns.tolist()

    # ── Numeric ───────────────────────────────────────────────────────────────
    num_missing = [c for c in num_cols if df[c].isnull().any()]
    if num_missing:
        if num_strategy == "KNN Imputation":
            st.warning(
                "⚠️ KNN Imputation is fitted on the full dataset. "
                "Split your data before fitting in a real ML pipeline to avoid leakage."
            )
            imputer = KNNImputer(n_neighbors=5)
            df[num_cols] = imputer.fit_transform(df[num_cols])
            _log(logs, f"KNN Imputation applied to numeric columns: {num_missing}")

        elif num_strategy == "Forward/Backward Fill (Time Series)":
            df[num_missing] = df[num_missing].ffill().bfill()
            _log(logs, f"Forward/backward fill applied to: {num_missing}")

        elif num_strategy == "Mean":
            means = df[num_missing].mean()
            df[num_missing] = df[num_missing].fillna(means)
            _log(logs, f"Mean imputation applied to: {num_missing}")

        else:  # Median (default — robust to outliers)
            medians = df[num_missing].median()
            df[num_missing] = df[num_missing].fillna(medians)
            _log(logs, f"Median imputation applied to: {num_missing}")
    else:
        _log(logs, "No missing values in numeric columns.")

    # ── Categorical ───────────────────────────────────────────────────────────
    cat_missing = [c for c in cat_cols if df[c].isnull().any()]
    if cat_missing:
        if cat_strategy == "Forward/Backward Fill (Time Series)":
            df[cat_missing] = df[cat_missing].ffill().bfill()
            _log(logs, f"Forward/backward fill applied to categorical: {cat_missing}")
        else:
            for col in cat_missing:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
            _log(logs, f"Mode imputation applied to categorical: {cat_missing}")
    else:
        _log(logs, "No missing values in categorical columns.")

    return df


# ── Step 6: Handle outliers ────────────────────────────────────────────────────
def handle_outliers(
    df:     pd.DataFrame,
    logs:   list,
    method: str = "IQR",
    action: str = "Cap",
) -> pd.DataFrame:
    """
    Detect and handle outliers column by column.

    Remove mode collects all row masks first and applies once —
    avoids the silent compounding row-loss bug of filtering in a loop.
    """
    df       = df.copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    n_before = len(df)

    outlier_counts = {}
    # For Remove mode: accumulate a boolean keep-mask
    keep_mask = pd.Series(True, index=df.index)

    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        if method == "IQR":
            Q1, Q3    = s.quantile(0.25), s.quantile(0.75)
            iqr       = Q3 - Q1
            lower     = Q1 - IQR_MULTIPLIER * iqr
            upper     = Q3 + IQR_MULTIPLIER * iqr
        else:  # Z-score
            mean, std = s.mean(), s.std()
            if std == 0:
                continue
            lower = mean - ZSCORE_THRESHOLD * std
            upper = mean + ZSCORE_THRESHOLD * std

        is_outlier = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(is_outlier.sum())

        if n_outliers == 0:
            continue

        outlier_counts[col] = n_outliers

        if action == "Cap":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif action == "Remove":
            keep_mask &= ~is_outlier   # accumulate — don't filter yet

    # Apply Remove mask once after all columns processed
    if action == "Remove" and not keep_mask.all():
        df = df[keep_mask]
        n_removed = n_before - len(df)
        _log(logs, f"Outlier removal: dropped {n_removed:,} rows ({n_removed/n_before*100:.1f}% of data).")

    if outlier_counts:
        summary = ", ".join(f"{c}: {n}" for c, n in outlier_counts.items())
        _log(logs, f"Outliers detected ({method}, {action}): {summary}")
    else:
        _log(logs, f"No outliers detected using {method}.")

    return df


# ── Main orchestrator ──────────────────────────────────────────────────────────
def auto_clean_data(
    df:             pd.DataFrame,
    num_strategy:   str = "Median",
    outlier_method: str = "IQR",
    outlier_action: str = "Cap",
) -> pd.DataFrame:
    """
    Full cleaning pipeline in correct order:

        1. Drop useless columns (empty, constant, unnamed)
        2. Fix numeric-string columns (commas, $, %, whitespace)
        3. Detect and convert datetime columns
        4. Remove duplicate rows
        5. Impute missing values
        6. Handle outliers

    Returns the cleaned DataFrame and renders a full log to Streamlit.
    """
    logs       = []
    df_cleaned = df.copy()
    n_cols_in  = df_cleaned.shape[1]
    n_rows_in  = df_cleaned.shape[0]

    st.write("### Cleaning pipeline")

    with st.spinner("Running cleaning steps..."):

        # Step 1
        df_cleaned = drop_useless_columns(df_cleaned, logs)

        # Step 2
        df_cleaned = fix_numeric_strings(df_cleaned, logs)

        # Step 3
        df_cleaned = detect_datetime_columns(df_cleaned, logs)

        # Step 4
        df_cleaned = remove_duplicates(df_cleaned, logs)

        # Step 5
        df_cleaned = fill_missing_values(
            df_cleaned, logs,
            num_strategy=num_strategy,
            cat_strategy="Mode",
        )

        # Step 6
        df_cleaned = handle_outliers(
            df_cleaned, logs,
            method=outlier_method,
            action=outlier_action,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_cols_out = df_cleaned.shape[1]
    n_rows_out = df_cleaned.shape[0]

    st.success("✅ Cleaning complete.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in",      f"{n_rows_in:,}")
    c2.metric("Rows out",     f"{n_rows_out:,}",
              delta=f"{n_rows_out - n_rows_in:,}")
    c3.metric("Cols in",      f"{n_cols_in}")
    c4.metric("Cols out",     f"{n_cols_out}",
              delta=f"{n_cols_out - n_cols_in}")

    remaining_missing = int(df_cleaned.isnull().sum().sum())
    if remaining_missing == 0:
        st.success("✅ No missing values remain.")
    else:
        st.warning(
            f"⚠️ {remaining_missing} missing values still remain "
            "(likely in datetime columns — safe to ignore for most ML models)."
        )

    # ── Cleaning log ──────────────────────────────────────────────────────────
    with st.expander("📋 View full cleaning log", expanded=False):
        for i, entry in enumerate(logs, 1):
            st.write(f"**{i}.** {entry}")

    return df_cleaned