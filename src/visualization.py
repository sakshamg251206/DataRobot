import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.feature_engineering import get_column_types


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_num_cols(df: pd.DataFrame) -> list[str]:
    num_cols, _, _, _ = get_column_types(df)
    return num_cols


def _get_cat_cols(df: pd.DataFrame) -> list[str]:
    _, cat_cols, bool_cols, _ = get_column_types(df)
    return cat_cols + bool_cols


# ── AI insight generator ───────────────────────────────────────────────────────
def auto_insight_generator(df: pd.DataFrame, context: str, extra: str = ""):
    """
    Generate an AI insight for the current chart context.
    context: "Histogram" | "Correlation" | "Pairplot"
    extra:   additional context string (e.g. column name for histogram)
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.caption("💡 Add your Gemini API key in the sidebar for AI insights.")
        return

    if not LANGCHAIN_AVAILABLE:
        return

    if st.button(f"🤖 Generate AI insight — {context}", key=f"insight_{context}_{extra}"):
        with st.spinner("Generating insight..."):
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

                if context == "Histogram":
                    col_stats = df[extra].describe().round(3).to_string()
                    skew      = round(float(df[extra].skew()), 3)
                    prompt = (
                        f"You are a data scientist. I have a numeric column '{extra}' with these stats:\n"
                        f"{col_stats}\nSkewness: {skew}\n\n"
                        "In 2-3 sentences, explain what this distribution suggests about the data "
                        "and whether any transformation (log, sqrt) or action is recommended."
                    )

                elif context == "Correlation":
                    num_df  = df.select_dtypes(include="number")
                    corr_str = num_df.corr().round(2).to_string()
                    prompt = (
                        f"You are a data scientist. Dataset columns: {list(df.columns)}.\n"
                        f"Pearson correlation matrix:\n{corr_str}\n\n"
                        "Identify the top 2-3 most meaningful correlations and explain "
                        "what they imply in plain business terms. Be concise."
                    )

                elif context == "Pairplot":
                    cols_used = extra
                    prompt = (
                        f"You are a data scientist. I plotted a scatter matrix of columns: {cols_used}. "
                        f"Dataset has {df.shape[0]} rows. "
                        "In 2-3 sentences, explain what patterns a data scientist should look for "
                        "in a scatter matrix and what actionable steps they imply."
                    )

                else:
                    return

                response = llm.invoke(prompt)
                st.info(f"**AI Insight:** {response.content}")

            except Exception as e:
                st.error(f"AI insight failed: {e}")


# ── Histogram ──────────────────────────────────────────────────────────────────
def plot_histogram(df: pd.DataFrame):
    st.subheader("Histograms")

    num_cols = _get_num_cols(df)
    cat_cols = _get_cat_cols(df)

    if not num_cols:
        st.warning("No numeric columns found.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        col_to_plot = st.selectbox("Column", num_cols, key="hist_col")
    with col2:
        color_by = st.selectbox(
            "Color by (optional)", ["None"] + cat_cols, key="hist_color"
        )
    with col3:
        n_bins = st.slider("Bins", min_value=10, max_value=100, value=30, key="hist_bins")

    color_col = None if color_by == "None" else color_by

    fig = px.histogram(
        df, x=col_to_plot,
        color=color_col,
        nbins=n_bins,
        marginal="box",
        title=f"Distribution of {col_to_plot}"
        + (f" colored by {color_col}" if color_col else ""),
        opacity=0.8,
        barmode="overlay",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution stats
    s    = df[col_to_plot].dropna()
    skew = s.skew()
    kurt = s.kurt()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean",     f"{s.mean():.3f}")
    c2.metric("Median",   f"{s.median():.3f}")
    c3.metric("Skewness", f"{skew:.3f}",
              help="|skew| > 1 → consider log/sqrt transform")
    c4.metric("Kurtosis", f"{kurt:.3f}",
              help="Kurtosis > 3 → heavy tails / outlier-prone")

    if abs(skew) > 1:
        st.warning(
            f"⚠️ `{col_to_plot}` is {'right' if skew > 0 else 'left'}-skewed "
            f"(skew = {skew:.2f}). Consider a log transform before modeling."
        )

    auto_insight_generator(df, context="Histogram", extra=col_to_plot)


# ── Correlation heatmap ────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    st.subheader("Correlation Heatmap")

    num_cols = _get_num_cols(df)

    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns for a correlation heatmap.")
        return

    # Let user filter columns if there are many
    if len(num_cols) > 10:
        selected = st.multiselect(
            "Select columns to include (default: all)",
            options=num_cols,
            default=num_cols[:10],
            key="heatmap_cols",
        )
        if len(selected) < 2:
            st.warning("Select at least 2 columns.")
            return
        num_cols = selected

    corr_matrix = df[num_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation matrix",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Highlight strongest correlations
    st.subheader("Strongest correlations")
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            c1 = corr_matrix.columns[i]
            c2 = corr_matrix.columns[j]
            r  = corr_matrix.loc[c1, c2]
            pairs.append({"Feature A": c1, "Feature B": c2, "Correlation": round(r, 4)})

    pairs_df = (
        pd.DataFrame(pairs)
        .sort_values("Correlation", key=abs, ascending=False)
        .head(10)
    )
    st.dataframe(pairs_df, use_container_width=True)

    # Flag multicollinearity risk
    high_corr = pairs_df[pairs_df["Correlation"].abs() > 0.85]
    if not high_corr.empty:
        st.warning(
            "⚠️ High multicollinearity detected (|r| > 0.85). "
            "Consider dropping one column from each pair before modeling:"
        )
        for _, row in high_corr.iterrows():
            st.write(f"  • `{row['Feature A']}` ↔ `{row['Feature B']}` = {row['Correlation']}")

    auto_insight_generator(df, context="Correlation")


# ── Pairplot / scatter matrix ──────────────────────────────────────────────────
def plot_pairplot(df: pd.DataFrame):
    st.subheader("Scatter Matrix (Pairplot)")

    num_cols = _get_num_cols(df)
    cat_cols = _get_cat_cols(df)

    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns for a scatter matrix.")
        return

    col1, col2 = st.columns(2)
    with col1:
        # Default to top 5 by variance — most informative columns
        default_cols = (
            df[num_cols].var()
            .sort_values(ascending=False)
            .index[:5]
            .tolist()
        )
        selected_cols = st.multiselect(
            "Columns to plot (max 6 recommended)",
            options=num_cols,
            default=default_cols,
            key="pairplot_cols",
        )
    with col2:
        color_by = st.selectbox(
            "Color by (optional)", ["None"] + cat_cols, key="pairplot_color"
        )

    if len(selected_cols) < 2:
        st.warning("Select at least 2 columns.")
        return

    if len(selected_cols) > 6:
        st.warning("More than 6 columns may render slowly. Consider reducing selection.")

    color_col = None if color_by == "None" else color_by

    fig = px.scatter_matrix(
        df,
        dimensions=selected_cols,
        color=color_col,
        title="Scatter matrix"
        + (f" colored by {color_col}" if color_col else ""),
        opacity=0.5,
    )
    fig.update_traces(diagonal_visible=True, showupperhalf=False)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    auto_insight_generator(df, context="Pairplot", extra=str(selected_cols))