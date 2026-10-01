"""Load a dataset from a file or a built-in sample, then preview it."""

import streamlit as st

from autods.config import get_settings
from autods.core.io import SUPPORTED_EXTENSIONS, DataLoadError, load_dataset
from autods.core.profiling import overview
from autods.core.samples import SAMPLES
from autods.ui import state
from autods.ui.components import page_header

page_header(
    "Load data",
    "Upload a table where each row is one record (a customer, a sale, a patient…) "
    "and each column is one attribute.",
    eyebrow="Step 1",
)

upload_tab, sample_tab = st.tabs(
    [":material/upload_file: Upload a file", ":material/dataset: Use a sample"]
)

with upload_tab:
    uploaded = st.file_uploader(
        "CSV, TSV or Excel file",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        help="The first row must contain column names. Delimiters (, ; tab |) and "
        "text encodings are detected automatically.",
    )
    if uploaded is not None and st.session_state.get("loaded_file_id") != uploaded.file_id:
        with st.spinner(f"Reading {uploaded.name}…"):
            try:
                result = load_dataset(uploaded.getvalue(), uploaded.name, get_settings().max_rows)
            except DataLoadError as exc:
                st.error(str(exc), icon=":material/error:")
            else:
                state.set_original(
                    result.df, uploaded.name, result.warnings, result.renamed_columns
                )
                st.session_state.loaded_file_id = uploaded.file_id
                st.toast(f"Loaded {uploaded.name}", icon=":material/check_circle:")

with sample_tab:
    options = {s.title: s for s in SAMPLES.values()}
    choice = st.radio(
        "Sample dataset",
        list(options),
        captions=[s.description for s in options.values()],
        label_visibility="collapsed",
    )
    if st.button("Load sample", type="primary", icon=":material/download:"):
        sample = options[choice]
        state.load_sample_dataset(sample.key)
        st.session_state.pop("loaded_file_id", None)
        st.toast(f"Loaded {sample.title}", icon=":material/check_circle:")

if not state.has_data():
    st.info("Load a file or a sample to continue.", icon=":material/info:")
    st.stop()

original = state.get_version(state.ORIGINAL)
assert original is not None
df = original.df
stats = overview(df)

st.divider()
st.subheader(state.dataset_name(), anchor=False)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{stats['rows']:,}")
c2.metric("Columns", stats["columns"])
c3.metric("Numeric", stats["numeric"])
c4.metric("Text / categorical", stats["categorical"])
c5.metric("Missing cells", f"{stats['missing_pct']:.1f}%")

warnings = state.load_warnings()
if warnings:
    with st.container(border=True):
        st.markdown("**Things to look at**")
        for warning in warnings:
            st.markdown(f":material/warning: {warning}")
else:
    st.success("No obvious data quality problems found.", icon=":material/verified:")

renamed = state.renamed_columns()
if renamed:
    with st.expander(f"{len(renamed)} column names were tidied to snake_case"):
        st.dataframe({"Original": list(renamed), "Now": list(renamed.values())}, hide_index=True)

st.dataframe(df.head(100), height=360)
st.caption("Showing up to the first 100 rows.")
st.page_link(
    "autods/ui/pages/profile.py", label="Next: profile the data", icon=":material/arrow_forward:"
)
