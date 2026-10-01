"""Landing page: what the app is, who it is for and how to use it."""

import streamlit as st

from autods.core.samples import SAMPLES
from autods.ui import state
from autods.ui.components import page_header

page_header(
    "Turn a spreadsheet into insights and a working model",
    "Upload a CSV or Excel file and walk through the same steps a data scientist would: "
    "understand the data, clean it, explore it, train and compare models, and share a report. "
    "Every automatic decision is explained in plain English.",
    eyebrow="Auto Data Science",
)

STEPS = [
    ("Load", "Upload CSV / Excel or pick a sample.", "autods/ui/pages/data.py"),
    ("Understand", "Profile columns, gaps and distributions.", "autods/ui/pages/profile.py"),
    (
        "Prepare",
        "Clean and encode data in one click or step by step.",
        "autods/ui/pages/smart_prep.py",
    ),
    ("Model", "Train and explain models, or forecast.", "autods/ui/pages/train.py"),
    ("Share", "Ask questions and download a report.", "autods/ui/pages/report.py"),
]

cols = st.columns(len(STEPS))
for i, (col, (title, text, page)) in enumerate(zip(cols, STEPS, strict=True), start=1):
    with col, st.container(border=True, height="stretch"):
        st.markdown(f"<span class='ads-step-n'>{i}</span>**{title}**", unsafe_allow_html=True)
        st.caption(text)
        st.page_link(page, label="Open", icon=":material/arrow_forward:")

st.space("small")
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Try it in 10 seconds", anchor=False)
    st.caption("No file handy? Load a real public dataset and follow the steps above.")
    for sample in SAMPLES.values():
        with st.container(border=True):
            text_col, button_col = st.columns([4, 1], vertical_alignment="center")
            text_col.markdown(f"**{sample.title}**")
            text_col.caption(sample.description)
            if button_col.button("Load", key=f"home_{sample.key}", width="stretch"):
                state.load_sample_dataset(sample.key)
                st.switch_page("autods/ui/pages/profile.py")

with right:
    st.subheader("Good to know", anchor=False)
    st.markdown(
        """
- **Who it's for:** analysts, students and anyone who wants a fast, honest first pass
  at a dataset without writing code.
- **Your data stays in this session.** Nothing is saved to disk; closing the tab clears it.
- **AI is optional.** With a Gemini key, the app sends *summary statistics* (never the
  full file) to Google to write explanations and answer questions.
- **Results you can trust:** models are evaluated on rows they never saw, preprocessing
  included, and every exported model works on raw data with the same columns.
"""
    )
    if state.has_data():
        st.success(f"**{state.dataset_name()}** is loaded. Continue with **Profile**.")
        st.page_link(
            "autods/ui/pages/profile.py", label="Go to Profile", icon=":material/arrow_forward:"
        )
