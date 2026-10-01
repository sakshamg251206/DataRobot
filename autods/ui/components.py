"""Reusable UI building blocks so every page looks and behaves the same."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import pandas as pd
import streamlit as st

from autods.ai.client import AIError, GeminiClient
from autods.ai.prompts import ANALYST_SYSTEM
from autods.config import get_settings
from autods.ui import state

CSS = """
<style>
.block-container { padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1200px; }
.ads-eyebrow { color: #4F46E5; font-weight: 600; font-size: .78rem; letter-spacing: .08em;
               text-transform: uppercase; margin-bottom: .1rem; }
.ads-sub { color: rgba(128,128,128,.95); font-size: 1.02rem; margin-top: -.4rem; margin-bottom: .6rem; }
.ads-step-n { display:inline-flex; width:1.6rem; height:1.6rem; border-radius:50%;
              align-items:center; justify-content:center; background:#4F46E5; color:#fff;
              font-weight:600; font-size:.85rem; margin-right:.4rem; }
div[data-testid="stMetric"] { padding: .2rem 0; }
@media (max-width: 640px) {
  .block-container { padding-top: 2.5rem; }
  h1 { font-size: 2rem !important; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str | None = None) -> None:
    if eyebrow:
        st.markdown(f"<div class='ads-eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(f"<div class='ads-sub'>{subtitle}</div>", unsafe_allow_html=True)


def empty_state(
    message: str, page: str | None = "autods/ui/pages/data.py", label: str = "Load data"
) -> None:
    """Friendly placeholder shown when a page has nothing to work with yet."""
    with st.container(border=True):
        st.markdown(f"#### {message}")
        if page:
            st.page_link(page, label=label, icon=":material/arrow_forward:")


def require_data() -> None:
    """Stop rendering the page (with guidance) if no dataset is loaded."""
    if not state.has_data():
        empty_state("No dataset loaded yet. Upload a file or pick a sample to get started.")
        st.stop()


READABLE = (state.CLEANED, state.ORIGINAL)


def dataset_picker(
    key: str, label: str = "Dataset version", prefer: tuple[str, ...] | None = None
) -> tuple[str, pd.DataFrame]:
    """Let the user choose which version of the data a page works on.

    The default is the first available name in ``prefer`` (e.g. human-readable
    versions for exploration pages), otherwise the most recent version.
    """
    available = state.versions()
    names = list(available)
    if len(names) == 1:
        return names[0], available[names[0]].df
    default = next((n for n in prefer or () if n in names), names[-1])
    choice = st.segmented_control(
        label,
        names,
        default=default,
        key=key,
        help=" · ".join(f"**{n}**: {state.VERSION_HELP[n]}" for n in names),
    )
    choice = choice or default
    return choice, available[choice].df


def shape_caption(df: pd.DataFrame) -> str:
    return f"{len(df):,} rows × {df.shape[1]} columns"


def show_log(log: Iterable[str], title: str = "What was done", expanded: bool = False) -> None:
    entries = list(log)
    if not entries:
        return
    with st.expander(
        f"{title} ({len(entries)} steps)", expanded=expanded, icon=":material/checklist:"
    ):
        st.markdown("\n".join(f"{i}. {line}" for i, line in enumerate(entries, 1)))


def csv_download(
    df: pd.DataFrame, filename: str, label: str = "Download CSV", key: str | None = None
) -> None:
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        icon=":material/download:",
        key=key,
    )


def column_list(cols: Iterable[str]) -> str:
    return ", ".join(f"`{c}`" for c in cols)


# ── AI helpers ─────────────────────────────────────────────────────────────────
def ai_client() -> GeminiClient | None:
    key = state.api_key()
    if not key:
        return None
    try:
        return GeminiClient(api_key=key, model=get_settings().gemini_model)
    except AIError:
        return None


def ai_insight(prompt: str, key: str, label: str = "Explain with AI") -> None:
    """A button that asks Gemini to explain something; answers are cached per session."""
    cache = state.ai_cache()
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if digest in cache:
        st.info(cache[digest], icon=":material/auto_awesome:")
        return
    if not state.api_key():
        st.caption(
            ":material/auto_awesome: Add a Gemini API key in the sidebar to get plain-English AI explanations here."
        )
        return
    if st.button(label, key=key, icon=":material/auto_awesome:"):
        client = ai_client()
        if client is None:
            st.error("Could not start the AI client. Check the API key.")
            return
        with st.spinner("Asking Gemini…"):
            try:
                answer = client.generate(prompt, system=ANALYST_SYSTEM)
            except AIError as exc:
                st.error(str(exc))
                return
        cache[digest] = answer
        st.info(answer, icon=":material/auto_awesome:")
