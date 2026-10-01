"""Visualize: distributions, correlations, scatter matrix and pairwise relationships."""

import pandas as pd
import streamlit as st

from autods.ai.prompts import column_insight_prompt, correlation_prompt, relationship_prompt
from autods.core.dtypes import boolean_columns, categorical_columns, numeric_columns, skewness
from autods.core.profiling import top_correlations
from autods.ui import charts
from autods.ui.components import READABLE, ai_insight, dataset_picker, page_header, require_data

page_header(
    "Visualize",
    "Interactive charts to see how values are distributed and how columns relate. "
    "Hover for exact values, drag to zoom, double-click to reset.",
    eyebrow="Step 2 · Understand",
)
require_data()

_, df = dataset_picker("viz_version", prefer=READABLE)
nums = numeric_columns(df)
groups = [c for c in categorical_columns(df) + boolean_columns(df) if df[c].nunique() <= 30]

dist_tab, corr_tab, matrix_tab, rel_tab = st.tabs(
    ["Distributions", "Correlations", "Scatter matrix", "Compare two columns"]
)

with dist_tab:
    left, mid, right = st.columns([2, 2, 1])
    column = left.selectbox("Column", list(df.columns), key="dist_col")
    is_numeric = column in nums
    color = mid.selectbox(
        "Split by",
        ["(none)", *[g for g in groups if g != column]],
        key="dist_color",
        disabled=not is_numeric,
    )
    bins = right.slider("Bins", 10, 100, 30, key="dist_bins", disabled=not is_numeric)

    if is_numeric:
        values = df[column].dropna()
        if values.empty:
            st.info("This column has no values.")
        else:
            st.plotly_chart(
                charts.histogram(df, column, None if color == "(none)" else color, bins)
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{values.mean():,.4g}")
            c2.metric("Median", f"{values.median():,.4g}")
            c3.metric("Std dev", f"{values.std():,.4g}")
            skew = skewness(values)
            c4.metric(
                "Skewness",
                f"{skew:.2f}",
                help="0 = symmetric. Above 1 or below −1 = strongly lopsided.",
            )
    else:
        if df[column].nunique() > 25:
            st.caption(f"Showing the 25 most common of {df[column].nunique():,} values.")
        st.plotly_chart(charts.category_bar(df, column))
    if df[column].notna().any():
        ai_insight(column_insight_prompt(df, column), key=f"ai_dist_{column}")

with corr_tab:
    if len(nums) < 2:
        st.info("Correlations need at least two numeric columns.")
    else:
        selected = st.multiselect(
            "Columns",
            nums,
            default=nums[:12],
            key="corr_cols",
            help="Pearson correlation: +1 = move together, −1 = move in opposite directions, 0 = no linear relationship.",
        )
        if len(selected) < 2:
            st.info("Pick at least two columns.")
        else:
            st.plotly_chart(charts.correlation_heatmap(df[selected].corr()))
            pairs = top_correlations(df, selected, n=10)
            st.markdown("**Strongest relationships**")
            st.dataframe(
                pairs,
                hide_index=True,
                column_config={
                    "Correlation": st.column_config.ProgressColumn(
                        "Correlation", min_value=-1, max_value=1, format="%.3f"
                    )
                },
            )
            redundant = pairs[pairs["Correlation"].abs() > 0.9]
            if not redundant.empty:
                st.warning(
                    "Some pairs are almost perfectly correlated (|r| > 0.9); one column of each "
                    "pair is probably redundant for modelling.",
                    icon=":material/warning:",
                )
            ai_insight(correlation_prompt(pairs), key="ai_corr")

with matrix_tab:
    if len(nums) < 2:
        st.info("The scatter matrix needs at least two numeric columns.")
    else:
        default = df[nums].var().sort_values(ascending=False).index[:4].tolist()
        left, right = st.columns([3, 1])
        selected = left.multiselect(
            "Columns (2–6)", nums, default=default, max_selections=6, key="sm_cols"
        )
        color = right.selectbox("Color by", ["(none)", *groups], key="sm_color")
        if len(selected) < 2:
            st.info("Pick at least two columns.")
        else:
            if len(df) > charts.MAX_POINTS:
                st.caption(f"Plotting a random sample of {charts.MAX_POINTS:,} rows for speed.")
            st.plotly_chart(
                charts.scatter_matrix(df, selected, None if color == "(none)" else color)
            )

with rel_tab:
    columns = list(df.columns)
    if len(columns) < 2:
        st.info("Need at least two columns.")
    else:
        left, right = st.columns(2)
        a = left.selectbox("First column", columns, key="rel_a")
        b = right.selectbox("Second column", [c for c in columns if c != a], key="rel_b")
        pair = df[[a, b]].dropna()
        if pair.empty:
            st.info("These two columns never have values in the same row.")
        else:
            a_num, b_num = a in nums, b in nums
            kind = (
                "scatter plot"
                if a_num and b_num
                else "box plot"
                if (a_num or b_num)
                else "count heatmap"
            )
            st.caption(f"Showing a {kind} — the chart type is chosen from the column types.")
            st.plotly_chart(charts.relationship(pd.DataFrame(pair), a, b))
            if a_num and b_num:
                st.metric("Correlation (Pearson r)", f"{pair[a].corr(pair[b]):.3f}")
            ai_insight(relationship_prompt(pair, a, b), key=f"ai_rel_{a}_{b}")
