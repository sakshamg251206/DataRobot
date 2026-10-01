"""Application shell: page config, navigation and the shared sidebar."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from autods import APP_NAME, __version__
from autods.ui import state
from autods.ui.components import inject_css, shape_caption

PAGES_DIR = "autods/ui/pages"
ASSETS = Path(__file__).resolve().parents[2] / "assets"


def _page(file: str, title: str, icon: str, default: bool = False) -> st.Page:
    return st.Page(f"{PAGES_DIR}/{file}", title=title, icon=f":material/{icon}:", default=default)


def _sidebar() -> None:
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")
    with st.sidebar:
        st.caption("CURRENT DATA")
        with st.container(border=True):
            if state.has_data():
                st.markdown(f"**{state.dataset_name()}**")
                for name, version in state.versions().items():
                    st.caption(f":material/check_circle: {name} · {shape_caption(version.df)}")
                result = state.training()
                if result is not None:
                    score = result.leaderboard.loc[0, result.primary_metric]
                    st.caption(
                        f":material/model_training: {result.best_model} · "
                        f"{result.primary_metric} {score:.3f}"
                    )
                if st.button("Start over", icon=":material/restart_alt:", width="stretch"):
                    state.reset()
                    st.rerun()
            else:
                st.caption(":material/radio_button_unchecked: No dataset loaded")

        with st.expander("AI settings", icon=":material/auto_awesome:", expanded=False):
            st.text_input(
                "Gemini API key",
                type="password",
                key="api_key_input",
                on_change=lambda: state.set_api_key(st.session_state.api_key_input),
                placeholder="Paste key and press Enter",
                help="Optional. Enables plain-English explanations and the dataset chat. "
                "Get a free key at aistudio.google.com. The key lives only in this browser "
                "session and is never stored.",
            )
            if state.api_key():
                st.caption(":material/check: AI features enabled")
            else:
                st.caption("AI features are optional; everything else works without a key.")

        st.caption(f"v{__version__}")


def run() -> None:
    load_dotenv()  # optional local .env; real environment variables take precedence
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=str(ASSETS / "icon.svg"),
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_css()

    navigation = st.navigation(
        {
            "Start": [
                _page("home.py", "Home", "home", default=True),
                _page("data.py", "Load data", "upload_file"),
            ],
            "Understand": [
                _page("profile.py", "Profile", "fact_check"),
                _page("visualize.py", "Visualize", "bar_chart"),
            ],
            "Prepare": [
                _page("smart_prep.py", "Smart Auto Prep", "bolt"),
                _page("clean.py", "Clean", "cleaning_services"),
                _page("features.py", "Engineer features", "tune"),
            ],
            "Model": [
                _page("train.py", "Train & compare", "model_training"),
                _page("forecast.py", "Forecast", "timeline"),
            ],
            "Share": [
                _page("assistant.py", "AI assistant", "forum"),
                _page("report.py", "Report", "description"),
            ],
        }
    )
    _sidebar()
    navigation.run()
