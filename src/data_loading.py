import streamlit as st
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")
MAX_ROWS             = 500_000   # warn if dataset is very large
MIN_ROWS             = 5         # reject trivially small files
MIN_COLS             = 2         # reject single-column files


# ── Helpers ────────────────────────────────────────────────────────────────────
def _read_csv(file) -> pd.DataFrame:
    """Try UTF-8 first, then fall back to latin-1 for non-standard encodings."""
    try:
        return pd.read_csv(file, encoding="utf-8")
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


def _read_excel(file) -> pd.DataFrame:
    # FIX: added try/except for openpyxl vs xlrd engine selection
    try:
        return pd.read_excel(file, engine="openpyxl")
    except Exception:
        file.seek(0)
        return pd.read_excel(file)


def _validate(df: pd.DataFrame) -> list[str]:
    """
    Run basic sanity checks on the raw dataframe.
    Returns a list of warning strings (empty = all good).
    """
    warnings = []

    if df.shape[0] < MIN_ROWS:
        warnings.append(
            f"⚠️ Dataset has only {df.shape[0]} rows — too small for meaningful analysis."
        )

    if df.shape[1] < MIN_COLS:
        warnings.append(
            f"⚠️ Dataset has only {df.shape[1]} column(s) — needs at least {MIN_COLS}."
        )

    if df.shape[0] > MAX_ROWS:
        warnings.append(
            f"⚠️ Dataset has {df.shape[0]:,} rows. Performance may be slow. "
            "Consider sampling before running the full pipeline."
        )

    # Fully empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        warnings.append(
            f"⚠️ Completely empty columns detected: `{'`, `'.join(empty_cols)}`. "
            "These will be dropped during cleaning."
        )

    # Fully duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        warnings.append(
            f"⚠️ {n_dupes:,} fully duplicate rows found. "
            "These will be removed during cleaning."
        )

    # All-same-value columns (zero variance — useless for ML)
    zero_var_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if zero_var_cols:
        warnings.append(
            f"⚠️ Zero-variance columns (single unique value): "
            f"`{'`, `'.join(zero_var_cols)}`. These carry no signal for ML."
        )

    # Columns that are >90% missing
    high_missing = [
        c for c in df.columns
        if df[c].isnull().mean() > 0.90
    ]
    if high_missing:
        warnings.append(
            f"⚠️ Columns with >90% missing values: `{'`, `'.join(high_missing)}`. "
            "Consider dropping them."
        )

    # Unnamed index columns (pandas artifact from previous CSV exports)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        warnings.append(
            f"⚠️ Unnamed index columns detected: `{'`, `'.join(unnamed)}`. "
            "These are usually accidental index exports and will be dropped."
        )

    return warnings


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names:
    - Strip leading/trailing whitespace
    - Replace spaces and special chars with underscores
    - Lowercase everything
    """
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are pandas index artifacts (Unnamed: 0, etc.)."""
    unnamed = [c for c in df.columns if c.startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


# ── Main entry point ───────────────────────────────────────────────────────────


def load_data(file) -> pd.DataFrame | None:
    """
    Load a CSV or Excel file into a DataFrame.

    Steps:
        1. Detect file type and read with appropriate engine + encoding fallback
        2. Standardise column names
        3. Drop unnamed index artifact columns
        4. Validate and surface warnings to the user
        5. Return the raw (uncleaned) DataFrame

    Returns None on failure.
    """
    if file is None:
        return None

    
    name = file.name
    ext = "." + file.name.rsplit(".", 1)[-1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        st.error(
            f"Unsupported file type `{ext}`. "
            f"Please upload one of: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
        return None

    # ── Read ──────────────────────────────────────────────────────────────────
    try:
        if ext == ".csv":
            df = _read_csv(file)
        else:
            df = _read_excel(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

    if df.empty:
        st.error("The uploaded file is empty.")
        return None

    # ── Standardise ───────────────────────────────────────────────────────────
    df = _clean_column_names(df)
    df = _drop_unnamed_columns(df)

    # ── Validate ──────────────────────────────────────────────────────────────
    warnings = _validate(df)
    for w in warnings:
        st.warning(w)

    # ── Summary card ──────────────────────────────────────────────────────────
    n_missing_pct = df.isnull().mean().mean() * 100
    n_numeric     = df.select_dtypes(include="number").shape[1]
    n_categorical = df.select_dtypes(include="object").shape[1]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows",        f"{df.shape[0]:,}")
    col2.metric("Columns",     df.shape[1])
    col3.metric("Numeric",     n_numeric)
    col4.metric("Categorical", n_categorical)
    col5.metric("Missing %",   f"{n_missing_pct:.1f}%")

    st.success(f"✅ `{file.name}` loaded successfully.")
    return df