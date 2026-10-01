"""Engineer features: encode, scale and create new columns for export or analysis."""

import streamlit as st

from autods.core.features import Encoding, FeatureOptions, Scaling, engineer_features
from autods.ui import state
from autods.ui.components import (
    READABLE,
    csv_download,
    dataset_picker,
    page_header,
    require_data,
    show_log,
)

page_header(
    "Engineer features",
    "Turn every column into numbers a model can use and optionally create new ones. "
    "Use this to export a model-ready file for other tools.",
    eyebrow="Step 3 · Prepare",
)
require_data()

source_name, df = dataset_picker("features_version", label="Start from", prefer=READABLE)
if source_name == state.ENGINEERED:
    st.caption(
        "Engineering on top of already engineered data — usually you want Original or Cleaned."
    )

with st.container(border=True):
    left, right = st.columns(2, gap="large")
    with left:
        encoding = st.radio(
            "Encode text columns as",
            list(Encoding),
            format_func=lambda e: e.value,
            help="One-hot is correct for unordered categories such as colours or cities. "
            "Ordinal is compact but implies an order; fine for tree models.",
        )
        scaling = st.radio(
            "Scale numeric columns",
            list(Scaling),
            format_func=lambda s: s.value,
            help="Scaling matters for distance- and gradient-based models, not for trees.",
        )
    with right:
        ratios = st.toggle(
            "Add ratio features", help="Up to 5 new a ÷ b columns from pairs of numeric columns."
        )
        polynomial = st.toggle(
            "Add polynomial features",
            help="Squares and pairwise products of the 4 most variable numeric columns.",
        )
        target_options = ["(none)", *df.columns]
        known = state.default_target(list(df.columns)) if state.has_known_target() else None
        target = st.selectbox(
            "Keep this column unchanged (target)",
            target_options,
            index=target_options.index(known) if known else 0,
            help="The target is excluded from encoding, scaling and feature generation, so new "
            "features can never contain the answer you want to predict.",
        )
    run = st.button("Apply", type="primary", icon=":material/tune:")

if run:
    with st.spinner("Engineering features…"):
        result = engineer_features(
            df,
            FeatureOptions(
                encoding=encoding or Encoding.ONE_HOT,
                scaling=scaling or Scaling.NONE,
                ratios=ratios,
                polynomial=polynomial,
            ),
            target=None if target == "(none)" else target,
        )
    log = [f"Started from the {source_name} version.", *result.log]
    state.set_version(
        state.ENGINEERED, result.df, log, target=None if target == "(none)" else target
    )
    st.toast("Saved as the **Engineered** version", icon=":material/check_circle:")

engineered = state.get_version(state.ENGINEERED)
if engineered is None:
    st.stop()

st.divider()
st.metric("Columns", engineered.df.shape[1])
show_log(engineered.log, expanded=True)
st.dataframe(engineered.df.head(100), height=320)
csv_download(engineered.df, "engineered.csv", key="dl_features")
st.caption(
    "Scalers and encoders here are fitted on the whole table, which is fine for export. "
    "For honest model scores, the **Train** page refits its own preprocessing on training rows only."
)
