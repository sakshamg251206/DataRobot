"""Profile: what's in each column, how complete it is and what needs attention."""

import pandas as pd
import streamlit as st

from autods.core.profiling import (
    SKEW_THRESHOLD,
    categorical_summary,
    column_profile,
    missing_report,
    numeric_summary,
    overview,
    readiness_flags,
    skewed_columns,
)
from autods.ui import charts
from autods.ui.components import READABLE, column_list, dataset_picker, page_header, require_data

page_header(
    "Profile",
    "A health check of every column: its type, how much is missing, and what to fix "
    "before modelling.",
    eyebrow="Step 2 · Understand",
)
require_data()

_, df = dataset_picker("profile_version", prefer=READABLE)


@st.cache_data(show_spinner="Profiling columns…", max_entries=8)
def _profile(data: pd.DataFrame) -> pd.DataFrame:
    return column_profile(data)


stats = overview(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{stats['rows']:,}")
c2.metric("Columns", stats["columns"])
c3.metric(
    "Missing cells", f"{stats['missing_cells']:,}", help=f"{stats['missing_pct']:.1f}% of all cells"
)
c4.metric("Duplicate rows", f"{stats['duplicate_rows']:,}")
c5.metric("Memory", f"{stats['memory_mb']:.1f} MB")

profile = _profile(df)
flags = {reason: cols for reason, cols in readiness_flags(profile).items() if cols}

overview_tab, missing_tab, numeric_tab, text_tab = st.tabs(
    ["Columns", "Missing values", "Numeric statistics", "Text statistics"]
)

with overview_tab:
    if flags:
        with st.container(border=True):
            st.markdown("**Needs attention before modelling**")
            for reason, cols in flags.items():
                st.markdown(f"- {reason}: {column_list(cols)}")
    st.dataframe(
        profile,
        hide_index=True,
        height=min(38 * (len(profile) + 1), 520),
        column_config={
            "Missing %": st.column_config.ProgressColumn(
                "Missing %", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Unique": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption(
        "**Kind** is inferred from the values, not just the storage type — for example, text "
        "that holds dates is reported as *Date / time*."
    )

with missing_tab:
    missing = missing_report(df)
    if missing.empty:
        st.success("No missing values anywhere in the dataset.", icon=":material/verified:")
    else:
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.plotly_chart(charts.missing_bar(missing))
        with right:
            st.dataframe(missing, hide_index=True)
            critical = missing.loc[missing["Missing %"] > 50, "Column"].tolist()
            if critical:
                st.warning(
                    f"More than half empty, consider dropping: {column_list(critical)}",
                    icon=":material/warning:",
                )
            st.caption("The **Clean** and **Smart Auto Prep** pages fill or drop these gaps.")

with numeric_tab:
    summary = numeric_summary(df)
    if summary.empty:
        st.info("No numeric columns.")
    else:
        st.dataframe(summary)
        skews = skewed_columns(df, threshold=0)
        if not skews.empty:
            st.markdown("**Skewness**: how lopsided each distribution is")
            st.plotly_chart(charts.skew_bar(skews.head(30), SKEW_THRESHOLD))
            heavy = skews[skews.abs() >= SKEW_THRESHOLD].index.tolist()
            if heavy:
                st.caption(
                    f"Bars in red have |skew| ≥ {SKEW_THRESHOLD}: {column_list(heavy)}. "
                    "Linear models may benefit from a log transform; tree models don't care."
                )

with text_tab:
    cats = categorical_summary(df)
    if cats.empty:
        st.info("No text or categorical columns.")
    else:
        st.dataframe(cats, hide_index=True)
        st.caption(
            "A very high **Share %** means one value dominates (little signal). "
            "Few unique values make a column a good candidate for one-hot encoding."
        )

st.page_link(
    "autods/ui/pages/visualize.py", label="Next: visualize", icon=":material/arrow_forward:"
)
