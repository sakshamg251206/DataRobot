"""Report: download a shareable summary of the data and any trained models."""

import re

import streamlit as st

from autods.core.reporting import ReportContext, build_html_report, build_pdf_report
from autods.ui import state
from autods.ui.components import READABLE, dataset_picker, page_header, require_data

page_header(
    "Report",
    "Download a summary to share with people who don't use the app: data overview, "
    "processing steps, missing values, charts and model scores.",
    eyebrow="Step 5 · Share",
)
require_data()

version_name, df = dataset_picker("report_version", prefer=READABLE)
version = state.get_version(version_name)
original = state.get_version(state.ORIGINAL)
assert version is not None and original is not None

training = state.training()
include_model = False
if training is not None:
    include_model = st.toggle(
        f"Include model results ({training.best_model} predicting {training.target})", value=True
    )

ctx = ReportContext(
    df=df,
    dataset_name=state.dataset_name(),
    raw_shape=original.df.shape if version_name != state.ORIGINAL else None,
    processing_log=[line.replace("`", "") for line in version.log],
    model_summary=training.leaderboard if include_model and training else None,
    model_target=training.target if include_model and training else None,
    model_task=training.task.value if include_model and training else None,
)
cache = state.page_cache().setdefault("report", {})
report_key = (version_name, include_model, id(training))
if cache.get("key") != report_key:  # settings changed: previously built files are stale
    cache.clear()
    cache["key"] = report_key
slug = re.sub(r"[^\w]+", "_", state.dataset_name()).strip("_").lower() or "dataset"

left, right = st.columns(2, gap="large")
with left, st.container(border=True, height="stretch"):
    st.markdown("#### :material/web: Interactive HTML")
    st.caption(
        "Charts you can hover and zoom. Opens in any browser; best for sharing by email or chat."
    )
    if st.button("Build HTML report", type="primary"):
        with st.spinner("Building report…"):
            cache["html"] = build_html_report(ctx).encode("utf-8")
    if cache.get("html"):
        st.download_button(
            "Download HTML",
            cache["html"],
            file_name=f"{slug}_report.html",
            mime="text/html",
            icon=":material/download:",
        )
with right, st.container(border=True, height="stretch"):
    st.markdown("#### :material/picture_as_pdf: PDF summary")
    st.caption("A compact text summary of the same information, ready to print or attach.")
    if st.button("Build PDF report"):
        with st.spinner("Building PDF…"):
            try:
                cache["pdf"] = build_pdf_report(ctx)
            except Exception as exc:
                st.error(f"Could not build the PDF: {exc}", icon=":material/error:")
    if cache.get("pdf"):
        st.download_button(
            "Download PDF",
            cache["pdf"],
            file_name=f"{slug}_report.pdf",
            mime="application/pdf",
            icon=":material/download:",
        )
