import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def generate_text_insight(prompt_context: str) -> str:
    """
    Generate a 1-2 sentence AI insight for the given statistical context.
    Returns a fallback string if the API key is absent or the call fails.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "💡 Provide a Google Gemini API Key in the sidebar to unlock automated AI textual insights."

    if not LANGCHAIN_AVAILABLE:
        return "💡 Install `langchain-google-genai` to enable AI insights."

    try:
        llm    = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        prompt = (
            "You are a Senior Data Analyst. Analyze the following local statistical context "
            "and return exactly 1-2 sharp, highly-insightful sentences summarizing what it means.\n\n"
            f"Context:\n{prompt_context}"
        )
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"⚠️ AI Insight Generation failed: {e}"


def render_univariate(df: pd.DataFrame, col: str):
    if pd.api.types.is_numeric_dtype(df[col]):
        st.write(f"#### Distribution of `{col}` (Numerical)")
        fig = px.histogram(
            df, x=col, marginal="box",
            title=f"Histogram with Boxplot: {col}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # FIX: guard against empty/constant column before computing skewness
        valid = df[col].dropna()
        if valid.empty or valid.std() == 0:
            st.info("Column has no variance — skewness not available.")
            return

        skew           = valid.skew()
        mean_val       = valid.mean()
        median_val     = valid.median()
        insight_context = (
            f"Numerical variable '{col}' has skewness={skew:.2f}, "
            f"mean={mean_val:.4g}, median={median_val:.4g}. "
            f"The distribution has {valid.shape[0]:,} non-null observations."
        )

    else:
        st.write(f"#### Frequency of `{col}` (Categorical)")
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]

        # FIX: truncate to top 30 categories to prevent unreadable charts
        if len(counts) > 30:
            st.caption(f"Showing top 30 of {len(counts)} categories.")
            counts = counts.head(30)

        fig = px.bar(counts, x=col, y="Count", title=f"Count Plot: {col}")
        st.plotly_chart(fig, use_container_width=True)

        top_cat = df[col].mode()[0] if not df[col].mode().empty else "N/A"
        insight_context = (
            f"Categorical variable '{col}' has {df[col].nunique()} unique categories. "
            f"Most frequent: '{top_cat}' with {df[col].value_counts().max()} occurrences "
            f"({df[col].value_counts().max() / max(len(df), 1) * 100:.1f}% of data)."
        )

    with st.spinner("Generating AI Insight..."):
        ai_insight = generate_text_insight(insight_context)
    st.info(f"**🤖 AI Insight:** {ai_insight}")


def render_bivariate(df: pd.DataFrame, col1: str, col2: str):
    type1 = "Num" if pd.api.types.is_numeric_dtype(df[col1]) else "Cat"
    type2 = "Num" if pd.api.types.is_numeric_dtype(df[col2]) else "Cat"

    st.write(f"#### Bivariate Analysis: `{col1}` vs `{col2}`")

    if type1 == "Num" and type2 == "Num":
        # FIX: sample without hard-coding state; show user a notice
        plot_df = df
        if len(df) > 10_000:
            plot_df = df.sample(10_000, random_state=42)
            st.caption(f"Sampled 10,000 / {len(df):,} rows for scatter performance.")

        fig = px.scatter(
            plot_df, x=col1, y=col2,
            trendline="ols",
            title="Scatter Plot (Num vs Num)",
            opacity=0.6,
        )
        st.plotly_chart(fig, use_container_width=True)

        # FIX: guard division-by-zero in correlation when std is 0
        try:
            corr = df[col1].corr(df[col2])
            corr_str = f"{corr:.4f}"
        except Exception:
            corr_str = "N/A"
        insight_context = (
            f"Comparing numerical features '{col1}' and '{col2}'. "
            f"Pearson r = {corr_str}."
        )

    elif type1 == "Cat" and type2 == "Cat":
        # FIX: limit crosstab size to prevent huge unreadable heatmaps
        top_c1 = df[col1].value_counts().index[:15]
        top_c2 = df[col2].value_counts().index[:15]
        filtered = df[df[col1].isin(top_c1) & df[col2].isin(top_c2)]
        heatmap_data = pd.crosstab(filtered[col1], filtered[col2])
        fig = px.imshow(
            heatmap_data, text_auto=True, aspect="auto",
            color_continuous_scale="Blues",
            title="Crosstab Heatmap (Cat vs Cat) — top 15 categories each"
        )
        st.plotly_chart(fig, use_container_width=True)
        insight_context = (
            f"Categorical crosstab of '{col1}' × '{col2}'. "
            f"{col1} has {df[col1].nunique()} unique values, "
            f"{col2} has {df[col2].nunique()} unique values."
        )

    else:
        num_col = col1 if type1 == "Num" else col2
        cat_col = col1 if type1 == "Cat" else col2

        # FIX: limit categories in box plot to top 20 by frequency
        top_cats = df[cat_col].value_counts().index[:20]
        plot_df  = df[df[cat_col].isin(top_cats)]

        fig = px.box(
            plot_df, x=cat_col, y=num_col, color=cat_col,
            title="Box Plot Distribution (Cat vs Num) — top 20 categories"
        )
        st.plotly_chart(fig, use_container_width=True)

        avg_per_cat = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
        top3 = avg_per_cat.head(3).to_dict()
        insight_context = (
            f"Numerical '{num_col}' across categorical '{cat_col}'. "
            f"Top 3 categories by mean {num_col}: {top3}."
        )

    with st.spinner("Generating AI Insight..."):
        ai_insight = generate_text_insight(insight_context)
    st.info(f"**🤖 AI Insight:** {ai_insight}")


def render_advanced_analysis(df: pd.DataFrame):
    st.subheader("📊 Advanced Bivariate AI Analysis")
    st.write(
        "Deep dive into specific feature distributions and mapping structures, "
        "augmented with Gemini AI insights."
    )

    analysis_type = st.radio(
        "Select Analysis Type",
        ["Univariate (1 Column)", "Bivariate (2 Columns)"],
        horizontal=True
    )
    st.divider()

    if analysis_type == "Univariate (1 Column)":
        col1 = st.selectbox("Select Feature:", df.columns)
        if st.button("Run Univariate Analysis", type="primary"):
            render_univariate(df, col1)

    else:
        # FIX: guard against datasets with fewer than 2 columns
        if len(df.columns) < 2:
            st.warning("Need at least 2 columns for bivariate analysis.")
            return

        col_left, col_right = st.columns(2)
        with col_left:
            c1 = st.selectbox("Select Feature 1:", df.columns, key="biv_col1")
        with col_right:
            remaining = [c for c in df.columns if c != c1]
            if not remaining:
                st.warning("Need at least 2 distinct columns.")
                return
            c2 = st.selectbox("Select Feature 2:", remaining, key="biv_col2")

        if st.button("Run Bivariate Analysis", type="primary"):
            render_bivariate(df, c1, c2)