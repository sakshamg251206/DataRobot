"""Clean: choose how gaps and outliers are handled, then review exactly what changed."""

import streamlit as st

from autods.core.cleaning import (
    CleaningOptions,
    ImputeStrategy,
    OutlierAction,
    OutlierMethod,
    clean_dataset,
)
from autods.ui import state
from autods.ui.components import csv_download, page_header, require_data, show_log

page_header(
    "Clean",
    "Fix the common problems in raw data with choices you control. Columns that are empty "
    "or constant are dropped, numbers and dates stored as text are converted, and "
    "duplicate rows are removed automatically.",
    eyebrow="Step 3 · Prepare",
)
require_data()

original = state.get_version(state.ORIGINAL)
assert original is not None
df = original.df

with st.container(border=True):
    left, right = st.columns(2, gap="large")
    with left:
        impute = st.radio(
            "Fill missing numbers with",
            list(ImputeStrategy),
            format_func=lambda s: s.value,
            help="Median is robust to outliers and the safest default. KNN estimates each gap "
            "from the 5 most similar rows. Fill copies the previous value (for time-ordered "
            "data). Missing text is filled with the most common value.",
        )
    with right:
        method = st.radio(
            "Detect outliers using",
            list(OutlierMethod),
            format_func=lambda s: s.value,
            help="IQR works for skewed data; Z-score assumes a roughly bell-shaped distribution.",
        )
        action = st.radio(
            "Then",
            list(OutlierAction),
            format_func=lambda s: s.value,
            horizontal=True,
            help="Capping keeps every row but limits extreme values. Removing drops whole rows.",
        )
    protect = st.selectbox(
        "Protect a target column (optional)",
        ["(none)", *df.columns],
        help="A protected column is never filled in or capped, and rows where it is missing "
        "are dropped instead.",
    )
    run = st.button("Clean data", type="primary", icon=":material/cleaning_services:")

if run:
    with st.spinner("Cleaning…"):
        cleaned, report = clean_dataset(
            df,
            CleaningOptions(
                impute=impute or ImputeStrategy.MEDIAN,
                outlier_method=method or OutlierMethod.IQR,
                outlier_action=action or OutlierAction.CAP,
            ),
            target=None if protect == "(none)" else protect,
        )
    if cleaned.empty:
        st.error(
            "Cleaning removed every row. Try capping outliers instead of removing them.",
            icon=":material/error:",
        )
        st.stop()
    state.set_version(
        state.CLEANED, cleaned, report.log, target=None if protect == "(none)" else protect
    )
    st.toast("Cleaned data saved as the **Cleaned** version", icon=":material/check_circle:")

result = state.get_version(state.CLEANED)
if result is None:
    st.stop()

st.divider()
cleaned = result.df
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(cleaned):,}", delta=f"{len(cleaned) - len(df):,}")
c2.metric("Columns", cleaned.shape[1], delta=f"{cleaned.shape[1] - df.shape[1]:+d}")
remaining = int(cleaned.isna().sum().sum())
c3.metric("Missing cells left", f"{remaining:,}")
if remaining:
    st.caption(
        "Remaining gaps are in date columns, which are not imputed. Models handle them automatically."
    )

show_log(result.log, expanded=True)
st.dataframe(cleaned.head(100), height=320)
left, right = st.columns([1, 3])
with left:
    csv_download(cleaned, "cleaned.csv", key="dl_clean")
with right:
    st.page_link(
        "autods/ui/pages/train.py", label="Next: train models", icon=":material/arrow_forward:"
    )
