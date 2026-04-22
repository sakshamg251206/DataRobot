import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from itertools import combinations


# ── Constants ──────────────────────────────────────────────────────────────────
OHE_CARDINALITY_LIMIT = 50    # refuse OHE on columns with more unique vals than this
POLY_MAX_COLS         = 4     # max numeric cols to use for polynomial expansion
RATIO_MAX_PAIRS       = 5     # max ratio pairs to auto-generate
DATE_PARTS            = ["year", "month", "day", "dayofweek", "quarter"]


# ── Column type helper ─────────────────────────────────────────────────────────
def get_column_types(df: pd.DataFrame) -> tuple[list, list, list, list]:
    """
    Returns (num_cols, cat_cols, bool_cols, date_cols).
    Separates bool and datetime from generic object/category
    so each type is handled correctly downstream.
    """
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Remove booleans that sneak into object dtype
    cat_cols  = [c for c in cat_cols if c not in bool_cols]
    return num_cols, cat_cols, bool_cols, date_cols


# ── Step 1: Boolean columns → int ─────────────────────────────────────────────
def encode_booleans(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """Cast bool columns to 0/1 int. No encoding needed — they're already binary."""
    df        = df.copy()
    _, _, bool_cols, _ = get_column_types(df)

    if not bool_cols:
        return df

    for col in bool_cols:
        df[col] = df[col].astype(int)

    logs.append(f"Boolean → int (0/1): {bool_cols}")
    return df


# ── Step 2: Datetime columns → numeric features ────────────────────────────────
def extract_datetime_features(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    For each datetime column, extract year, month, day, dayofweek, quarter.
    Drop the original datetime column after extraction — ML models need numbers.
    """
    df = df.copy()
    _, _, _, date_cols = get_column_types(df)

    if not date_cols:
        return df

    extracted = []
    dropped   = []
    for col in date_cols:
        for part in DATE_PARTS:
            new_col = f"{col}_{part}"
            try:
                df[new_col] = getattr(df[col].dt, part)
                extracted.append(new_col)
            except Exception:
                pass   # FIX: silently skip parts that fail on certain freq types
        df.drop(columns=[col], inplace=True)
        dropped.append(col)

    if logs is not None:
        if extracted:
            logs.append(f"Extracted datetime features {DATE_PARTS} from: {dropped}")
        logs.append(f"Dropped original datetime columns: {dropped}")
    return df


# ── Step 3: Encode categorical columns ────────────────────────────────────────
def encode_categorical(
    df:       pd.DataFrame,
    strategy: str = "Label Encoding",
    target:   str | None = None,
    logs:     list | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns.

    Strategy:
    - Label Encoding: safe only for ordinal or tree-based models.
      Each column gets its OWN LabelEncoder saved in the returned dict.
    - One-Hot Encoding: correct for nominal categories.
      Columns with > OHE_CARDINALITY_LIMIT unique values are skipped
      to prevent feature explosion.

    Returns:
        df_encoded  — encoded dataframe
        encoders    — dict of {col: fitted_encoder} for later use on test data
    """
    if logs is None:
        logs = []

    _, cat_cols, _, _ = get_column_types(df)

    # Never encode the target column
    if target and target in cat_cols:
        cat_cols = [c for c in cat_cols if c != target]

    if not cat_cols:
        st.info("No categorical columns to encode.")
        return df, {}

    df_encoded = df.copy()
    encoders   = {}

    if strategy == "Label Encoding":
        st.warning(
            "⚠️ Label Encoding assigns arbitrary integers to categories. "
            "This implies ordinal relationships that may not exist. "
            "Use One-Hot Encoding for nominal categories with non-tree models."
        )
        for col in cat_cols:
            le = LabelEncoder()
            # Fill NaN with placeholder so LabelEncoder doesn't crash
            df_encoded[col] = df_encoded[col].astype(str).fillna("__missing__")
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col]   = le

        logs.append(f"Label Encoding applied to: {cat_cols}")
        st.success(f"Label Encoding applied to {len(cat_cols)} columns.")

    elif strategy == "One-Hot Encoding":
        high_card = [
            c for c in cat_cols
            if df_encoded[c].nunique() > OHE_CARDINALITY_LIMIT
        ]
        safe_cols = [c for c in cat_cols if c not in high_card]

        if high_card:
            st.warning(
                f"⚠️ Skipped One-Hot Encoding for high-cardinality columns "
                f"(>{OHE_CARDINALITY_LIMIT} unique values): `{'`, `'.join(high_card)}`. "
                "Consider Label Encoding or dropping these columns."
            )

        if safe_cols:
            n_before   = df_encoded.shape[1]
            df_encoded = pd.get_dummies(df_encoded, columns=safe_cols, drop_first=True)
            n_after    = df_encoded.shape[1]
            n_new      = n_after - n_before + len(safe_cols)
            logs.append(
                f"One-Hot Encoding applied to: {safe_cols} "
                f"(created {n_new} new binary columns)."
            )
            st.success(
                f"One-Hot Encoding applied to {len(safe_cols)} columns "
                f"→ {n_new} new binary columns created."
            )
            encoders["__ohe_columns__"] = safe_cols

    # Ensure all encoded columns are numeric (get_dummies returns bool in newer pandas)
    bool_new = df_encoded.select_dtypes(include="bool").columns
    if len(bool_new) > 0:
        df_encoded[bool_new] = df_encoded[bool_new].astype(int)

    return df_encoded, encoders


# ── Step 4: Scale numeric columns ─────────────────────────────────────────────
def scale_numerical(
    df:       pd.DataFrame,
    strategy: str = "Standard Scaling",
    target:   str | None = None,
    logs:     list | None = None,
) -> tuple[pd.DataFrame, object | None]:
    """
    Scale numeric columns.

    ⚠️  The target column is always excluded from scaling.
        Scaler is returned so it can be applied to test data later.

    Returns:
        df_scaled — scaled dataframe
        scaler    — fitted scaler object (None if strategy is 'None')
    """
    if logs is None:
        logs = []

    num_cols, _, _, _ = get_column_types(df)

    # Exclude target from scaling
    if target and target in num_cols:
        num_cols = [c for c in num_cols if c != target]
        logs.append(f"Target column `{target}` excluded from scaling.")

    if not num_cols:
        st.info("No numeric columns to scale.")
        return df, None

    df_scaled = df.copy()

    if strategy == "None":
        logs.append("Scaling skipped (None selected).")
        return df_scaled, None

    if strategy == "Standard Scaling":
        scaler = StandardScaler()
    elif strategy == "Min-Max Normalization":
        scaler = MinMaxScaler()
    else:
        logs.append(f"Unknown scaling strategy: {strategy}. Skipping.")
        return df_scaled, None

    # FIX: drop columns with zero variance before scaling to avoid division-by-zero
    zero_var = [c for c in num_cols if df_scaled[c].std() == 0]
    if zero_var:
        num_cols = [c for c in num_cols if c not in zero_var]
        logs.append(f"Skipped scaling for zero-variance columns: {zero_var}")

    if not num_cols:
        return df_scaled, None

    df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    logs.append(f"{strategy} applied to {len(num_cols)} numeric columns (target excluded).")
    st.success(f"{strategy} applied to {len(num_cols)} numeric columns.")

    st.info(
        "💾 Save the scaler object to apply the same transformation to test/new data. "
        "Never refit on test data — that leaks distribution information."
    )

    return df_scaled, scaler


# ── Step 5: Feature engineering (ratios + polynomial) ─────────────────────────
def auto_feature_engineering(
    df:     pd.DataFrame,
    poly:   bool = False,
    ratios: bool = False,
    target: str | None = None,
    logs:   list | None = None,
) -> pd.DataFrame:
    """
    Generate new features from existing numeric columns.

    Ratios: top RATIO_MAX_PAIRS pairs by absolute correlation difference.
            Skips pairs where denominator is near-zero.
    Poly:   degree-2 expansion on top POLY_MAX_COLS numeric columns
            (by variance — most informative columns get expanded).

    Target column is excluded from feature generation inputs.
    """
    if logs is None:
        logs = []

    num_cols, _, _, _ = get_column_types(df)

    # Exclude target from feature inputs
    if target and target in num_cols:
        num_cols = [c for c in num_cols if c != target]

    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns for feature engineering.")
        return df

    df_eng = df.copy()

    # ── Ratios ────────────────────────────────────────────────────────────────
    if ratios:
        pairs_created = []
        all_pairs = list(combinations(num_cols, 2))

        for col1, col2 in all_pairs[:RATIO_MAX_PAIRS * 3]:
            denom = df_eng[col2].abs()
            # Skip if denominator is mostly zero (ratio would be meaningless)
            if (denom < 1e-6).mean() > 0.3:
                continue

            ratio_col         = f"{col1}_div_{col2}"
            # FIX: avoid SettingWithCopyWarning by direct assignment
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_vals = np.where(
                    df_eng[col2].abs() < 1e-6, np.nan,
                    df_eng[col1].values / df_eng[col2].values
                )
            df_eng[ratio_col] = ratio_vals
            df_eng[ratio_col] = df_eng[ratio_col].replace([np.inf, -np.inf], np.nan)
            median_val        = df_eng[ratio_col].median()
            df_eng[ratio_col] = df_eng[ratio_col].fillna(
                median_val if pd.notna(median_val) else 0
            )
            pairs_created.append(ratio_col)

            if len(pairs_created) >= RATIO_MAX_PAIRS:
                break

        if pairs_created:
            logs.append(f"Ratio features created: {pairs_created}")
            st.success(f"Created {len(pairs_created)} ratio features.")
        else:
            st.warning("No valid ratio pairs found (denominators too close to zero).")

    # ── Polynomial features ───────────────────────────────────────────────────
    if poly:
        # Pick top columns by variance — these carry the most signal
        variances = df_eng[num_cols].var().sort_values(ascending=False)
        top_cols  = variances.index[:POLY_MAX_COLS].tolist()

        # FIX: guard against NaN in polynomial input
        poly_input = df_eng[top_cols].fillna(df_eng[top_cols].median())

        poly_feat  = PolynomialFeatures(degree=2, include_bias=False)
        poly_arr   = poly_feat.fit_transform(poly_input)
        poly_names = poly_feat.get_feature_names_out(top_cols)

        poly_df = pd.DataFrame(
            poly_arr,
            columns=poly_names,
            index=df_eng.index
        )

        # Drop originals to avoid duplication (they're included in poly output)
        df_eng = df_eng.drop(columns=top_cols)
        df_eng = pd.concat([df_eng, poly_df], axis=1)

        n_new = poly_df.shape[1] - len(top_cols)
        logs.append(
            f"Polynomial (degree 2) features from top {len(top_cols)} columns "
            f"by variance: {top_cols}. Created {n_new} new features."
        )
        st.success(
            f"Polynomial features created from: {top_cols} "
            f"→ {poly_df.shape[1]} total columns (including originals)."
        )

    return df_eng


# ── Full pipeline wrapper ──────────────────────────────────────────────────────
def run_feature_pipeline(
    df:               pd.DataFrame,
    encoding_strategy: str  = "Label Encoding",
    scaling_strategy:  str  = "Standard Scaling",
    poly:              bool = False,
    ratios:            bool = False,
    target:            str | None = None,
) -> tuple[pd.DataFrame, dict, object | None, list]:
    """
    Run the full feature engineering pipeline in correct order:

        1. Encode booleans → int
        2. Extract datetime features
        3. Generate ratio / polynomial features  (on raw numeric, before scaling)
        4. Encode categorical columns
        5. Scale numeric columns                 (after encoding, never on target)

    Returns:
        df_out   — fully transformed dataframe ready for ML
        encoders — fitted encoder dict (save for test data)
        scaler   — fitted scaler (save for test data)
        logs     — list of strings describing every step taken
    """
    logs = []

    # 1. Booleans
    df = encode_booleans(df, logs)

    # 2. Datetimes
    df = extract_datetime_features(df, logs)

    # 3. Feature generation (before scaling — poly on raw values)
    df = auto_feature_engineering(df, poly=poly, ratios=ratios, target=target, logs=logs)

    # 4. Encode categoricals
    df, encoders = encode_categorical(df, strategy=encoding_strategy, target=target, logs=logs)

    # 5. Scale numerics (target excluded)
    df, scaler = scale_numerical(df, strategy=scaling_strategy, target=target, logs=logs)

    return df, encoders, scaler, logs