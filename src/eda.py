import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ── Constants ──────────────────────────────────────────────────────────────────
HIGH_CARDINALITY_THRESHOLD = 0.95   # unique ratio above this → likely ID column
LOW_CARDINALITY_THRESHOLD  = 10     # nunique at or below this → treat as categorical
SKEW_THRESHOLD             = 1.0    # |skew| above this → flag for log transform
MISSING_CRITICAL           = 0.50   # >50% missing → critical warning


# ── Internal helpers ───────────────────────────────────────────────────────────
def _classify_column(series: pd.Series, total_rows: int) -> str:
    """
    Classify a column into one of:
        ID | Date | Binary | Low-card categorical | High-card categorical |
        Continuous numeric | Discrete numeric
    """
    # FIX: guard against empty series
    if series.dropna().empty:
        return "Empty"

    unique_ratio = series.nunique(dropna=False) / max(total_rows, 1)

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date/Time"

    # Try parsing as date if object
    if series.dtype == object:
        try:
            sample = series.dropna().head(20)
            if not sample.empty:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.7:
                    return "Date/Time (string)"
        except Exception:
            pass

    if unique_ratio >= HIGH_CARDINALITY_THRESHOLD and series.dtype != object:
        return "ID / Index"

    # FIX: use pd.api.types instead of deprecated is_categorical_dtype
    if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
        if series.nunique() <= LOW_CARDINALITY_THRESHOLD:
            return "Low-card categorical"
        elif unique_ratio >= HIGH_CARDINALITY_THRESHOLD:
            return "High-card categorical (free text?)"
        else:
            return "Categorical"

    if pd.api.types.is_bool_dtype(series):
        return "Binary"

    if series.nunique() == 2:
        return "Binary"

    if pd.api.types.is_integer_dtype(series):
        if series.nunique() <= LOW_CARDINALITY_THRESHOLD:
            return "Discrete numeric (low-card)"
        return "Discrete numeric"

    return "Continuous numeric"


def _missing_heatmap(df: pd.DataFrame) -> go.Figure:
    """Mini heatmap showing missingness per column."""
    missing_pct = (df.isnull().mean() * 100).reset_index()
    missing_pct.columns = ["Column", "Missing %"]
    missing_pct = missing_pct.sort_values("Missing %", ascending=False)

    fig = px.bar(
        missing_pct,
        x="Column", y="Missing %",
        color="Missing %",
        color_continuous_scale=["#1d9e75", "#fac775", "#e24b4a"],
        range_color=[0, 100],
        title="Missing values per column (%)",
        height=350,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#888",
        showlegend=False,
        xaxis_tickangle=-35,
        margin=dict(t=50, b=80),
        coloraxis_showscale=False,
    )
    return fig


def _skew_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of skewness for all numeric columns."""
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        return go.Figure()

    skew   = num_df.skew().sort_values(key=abs, ascending=False).reset_index()
    skew.columns = ["Column", "Skewness"]

    # FIX: build color list properly without apply on Series
    colors = ["#e24b4a" if abs(v) >= SKEW_THRESHOLD else "#1d9e75"
              for v in skew["Skewness"]]

    fig = px.bar(
        skew, x="Column", y="Skewness",
        title=f"Skewness per numeric column (|skew| ≥ {SKEW_THRESHOLD} = needs transform)",
        height=350,
    )
    fig.update_traces(marker_color=colors)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#888",
        xaxis_tickangle=-35,
        margin=dict(t=50, b=80),
    )
    return fig


# ── Public functions ───────────────────────────────────────────────────────────
def show_data_info(df: pd.DataFrame):
    """
    High-level dataset overview:
    - Shape, duplicates, memory usage
    - Per-column classification (ID / date / categorical / numeric)
    - Cardinality report
    - Skewness flags
    """
    st.subheader("Dataset Overview")

    # Top metrics
    n_dupes   = int(df.duplicated().sum())
    n_missing = int(df.isnull().sum().sum())
    memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows",           f"{df.shape[0]:,}")
    c2.metric("Columns",        df.shape[1])
    c3.metric("Duplicate rows", f"{n_dupes:,}")
    c4.metric("Missing cells",  f"{n_missing:,}")
    c5.metric("Memory",         f"{memory_mb:.1f} MB")

    if n_dupes > 0:
        st.warning(f"⚠️ {n_dupes:,} duplicate rows found — these should be removed in cleaning.")

    st.divider()

    # Column classification table
    st.subheader("Column Profile")
    st.caption(
        "Understand what kind of data each column contains before cleaning or modelling."
    )

    rows = []
    for col in df.columns:
        s           = df[col]
        col_type    = _classify_column(s, df.shape[0])
        n_unique    = s.nunique(dropna=False)
        missing_pct = s.isnull().mean() * 100

        # FIX: safe sample values — avoid errors on mixed types
        try:
            sample_vals = ", ".join(str(v) for v in s.dropna().unique()[:3])
        except Exception:
            sample_vals = "N/A"

        skew_val = None
        if pd.api.types.is_numeric_dtype(s):
            try:
                skew_val = round(float(s.skew()), 2)
            except Exception:
                pass

        rows.append({
            "Column":        col,
            "Dtype":         str(s.dtype),
            "Kind":          col_type,
            "Unique":        n_unique,
            "Missing %":     round(missing_pct, 1),
            "Skewness":      skew_val if skew_val is not None else "—",
            "Sample values": sample_vals,
        })

    profile_df = pd.DataFrame(rows)
    st.dataframe(profile_df, use_container_width=True, height=350)

    # ML-relevant flags
    id_cols    = profile_df[profile_df["Kind"].str.contains("ID",    na=False)]
    date_cols  = profile_df[profile_df["Kind"].str.contains("Date",  na=False)]
    hcard_cols = profile_df[profile_df["Kind"].str.contains("High-card", na=False)]

    flags = []
    if not id_cols.empty:
        flags.append(
            f"🔴 **Likely ID columns** (drop before ML): "
            f"`{'`, `'.join(id_cols['Column'].tolist())}`"
        )
    if not date_cols.empty:
        flags.append(
            f"🟡 **Date columns** (extract year/month/day features): "
            f"`{'`, `'.join(date_cols['Column'].tolist())}`"
        )
    if not hcard_cols.empty:
        flags.append(
            f"🟠 **High-cardinality categoricals** (consider hashing or dropping): "
            f"`{'`, `'.join(hcard_cols['Column'].tolist())}`"
        )

    if flags:
        st.subheader("⚡ ML Readiness Flags")
        for f in flags:
            st.markdown(f)

    st.divider()

    # Skewness chart
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        st.subheader("Skewness Analysis")
        st.caption(
            "Highly skewed numeric columns often need a log or square-root transform "
            "before tree-free ML models (linear regression, SVMs, neural nets)."
        )
        st.plotly_chart(_skew_chart(df), use_container_width=True)

        skewed = profile_df[
            profile_df["Skewness"].apply(
                lambda v: isinstance(v, float) and abs(v) >= SKEW_THRESHOLD
            )
        ]["Column"].tolist()

        if skewed:
            st.warning(
                f"⚠️ Columns with high skewness (consider log transform): "
                f"`{'`, `'.join(skewed)}`"
            )
        else:
            st.success("✅ No severely skewed numeric columns detected.")


def show_missing_values(df: pd.DataFrame):
    """
    Detailed missing value report with percentages and severity tiers.
    """
    st.subheader("Missing Value Report")

    missing_pct = df.isnull().mean() * 100
    missing_cnt = df.isnull().sum()

    # FIX: build DataFrame without relying on positional index alignment
    missing_df = pd.DataFrame({
        "Column":    df.columns.tolist(),
        "Missing #": missing_cnt.values,
        "Missing %": missing_pct.round(2).values,
    })
    missing_df["Severity"] = missing_df["Missing %"].apply(
        lambda p:
        "🔴 Critical (>50%)"   if p > 50  else
        "🟠 High (20–50%)"     if p > 20  else
        "🟡 Moderate (5–20%)"  if p > 5   else
        "🟢 Low (<5%)"         if p > 0   else
        "✅ None"
    )
    missing_df = missing_df.sort_values("Missing %", ascending=False)

    total_missing_pct = df.isnull().mean().mean() * 100
    cols_with_missing = (missing_pct > 0).sum()

    c1, c2 = st.columns(2)
    c1.metric("Columns with missing data", f"{cols_with_missing} / {df.shape[1]}")
    c2.metric("Overall missing %",         f"{total_missing_pct:.1f}%")

    if cols_with_missing == 0:
        st.success("✅ No missing values found in the dataset.")
        return

    st.dataframe(missing_df, use_container_width=True)
    st.plotly_chart(_missing_heatmap(df), use_container_width=True)

    # Recommendations
    critical = missing_df[missing_df["Missing %"] > 50]["Column"].tolist()
    if critical:
        st.error(
            f"🔴 These columns are >50% missing and should likely be **dropped**: "
            f"`{'`, `'.join(critical)}`"
        )


def show_summary_statistics(df: pd.DataFrame):
    """
    Split summary statistics into numeric and categorical tabs
    so the ML person can read each clearly.
    """
    st.subheader("Summary Statistics")

    num_df = df.select_dtypes(include="number")
    cat_df = df.select_dtypes(include="object")

    tab1, tab2 = st.tabs(["Numeric columns", "Categorical columns"])

    with tab1:
        if num_df.empty:
            st.info("No numeric columns found.")
        else:
            desc = num_df.describe().T
            # FIX: guard against missing skew/kurt when col has constant values
            try:
                desc["skewness"] = num_df.skew().round(3)
            except Exception:
                desc["skewness"] = np.nan
            try:
                desc["kurtosis"] = num_df.kurt().round(3)
            except Exception:
                desc["kurtosis"] = np.nan

            desc = desc.rename(columns={
                "count": "count", "mean": "mean", "std": "std dev",
                "min": "min", "25%": "Q1", "50%": "median",
                "75%": "Q3", "max": "max"
            })
            # FIX: only apply gradient to columns that exist in this describe output
            gradient_cols = [c for c in ["mean", "std dev"] if c in desc.columns]
            styled = desc.style.background_gradient(subset=gradient_cols, cmap="Blues")
            st.dataframe(styled, use_container_width=True)

            st.caption(
                "Skewness > 1 or < -1 = consider log transform. "
                "Kurtosis > 3 = heavy tails / outlier-prone."
            )

    with tab2:
        if cat_df.empty:
            st.info("No categorical columns found.")
        else:
            cat_rows = []
            for col in cat_df.columns:
                s = cat_df[col]
                vc = s.value_counts()
                top_val  = vc.idxmax() if not vc.empty else "—"
                top_freq = int(vc.max())  if not vc.empty else 0
                # FIX: guard div-by-zero when series is all NaN
                top_pct  = round(top_freq / max(len(s), 1) * 100, 1)
                cat_rows.append({
                    "Column":      col,
                    "Unique vals": s.nunique(),
                    "Top value":   str(top_val),
                    "Top freq":    top_freq,
                    "Top %":       top_pct,
                    "Missing %":   round(s.isnull().mean() * 100, 1),
                })
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)
            st.caption(
                "High unique count + high Top % = near-constant column (low signal). "
                "Low unique count = good one-hot encoding candidate."
            )