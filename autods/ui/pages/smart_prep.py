"""Smart Auto Prep: one click to a clean, fully numeric, model-ready table."""

import streamlit as st

from autods.core.smart import smart_prepare
from autods.ui import state
from autods.ui.components import csv_download, page_header, require_data, shape_caption, show_log

page_header(
    "Smart Auto Prep",
    "Pick the column you want to predict and let the app apply sensible defaults: fix types, "
    "remove duplicates and unusable columns, fill gaps, tame outliers and encode text as numbers.",
    eyebrow="Step 3 · Prepare",
)
require_data()

source = state.get_version(state.CLEANED) or state.get_version(state.ORIGINAL)
assert source is not None
df = source.df
existing = state.get_version(state.SMART)

with st.container(border=True):
    columns = list(df.columns)
    default_target = state.default_target(columns)
    target = st.selectbox(
        "What do you want to predict? (target column)",
        columns,
        index=columns.index(default_target),
        help="The target is protected: it is never filled in, capped, encoded or dropped.",
    )
    st.caption(
        f"Starting from the **{'Cleaned' if state.get_version(state.CLEANED) else 'Original'}** "
        f"data ({shape_caption(df)})."
    )
    run = st.button("Run Smart Auto Prep", type="primary", icon=":material/bolt:")

if run:
    with st.spinner("Preparing data…"):
        try:
            result = smart_prepare(df, target)
        except ValueError as exc:
            st.error(str(exc), icon=":material/error:")
            st.stop()
    state.set_version(
        state.SMART,
        result.df,
        result.log,
        target=target,
        meta={"before": result.score_before, "after": result.score_after},
    )
    existing = state.get_version(state.SMART)

if existing is None:
    st.info(
        "Smart Auto Prep creates a new **Smart prep** version of the data; the original is "
        "kept so you can always compare or start again.",
        icon=":material/info:",
    )
    st.stop()

st.divider()
before, after = existing.meta.get("before"), existing.meta.get("after")
c1, c2, c3 = st.columns(3)
if before is not None and after is not None:
    c1.metric(
        "ML readiness score",
        f"{after:.0f} / 100",
        delta=f"{after - before:+.0f} points",
        help="Penalises missing values, duplicates, non-numeric, constant and mostly-empty "
        "columns. 100 means a model can use the table as-is.",
    )
c2.metric("Rows", f"{len(existing.df):,}", delta=f"{len(existing.df) - len(df):,}")
c3.metric("Columns", existing.df.shape[1], delta=f"{existing.df.shape[1] - df.shape[1]:+d}")

show_log(existing.log, expanded=True)
st.dataframe(existing.df.head(100), height=320)
left, right = st.columns([1, 3])
with left:
    csv_download(existing.df, "smart_prepared.csv", key="dl_smart")
with right:
    st.page_link(
        "autods/ui/pages/train.py", label="Next: train models", icon=":material/arrow_forward:"
    )
st.caption(
    "Tip: the **Train** page re-learns all preprocessing on the training rows only, so it "
    "works equally well on the Original or Cleaned version."
)
