"""Headless smoke tests: every page renders without exceptions, empty and loaded.

Uses Streamlit's AppTest harness, which runs the real scripts (navigation
included) without a browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
PAGES = [
    "home.py",
    "data.py",
    "profile.py",
    "visualize.py",
    "smart_prep.py",
    "clean.py",
    "features.py",
    "train.py",
    "forecast.py",
    "assistant.py",
    "report.py",
]
SAMPLE_TITLES = {"breast_cancer": "Breast cancer diagnosis", "co2": "Mauna Loa CO₂"}


def _open(at: AppTest, page: str) -> AppTest:
    at.switch_page(f"autods/ui/pages/{page}")
    return at.run()


def _click(at: AppTest, label: str) -> AppTest:
    return next(b for b in at.button if b.label == label).click().run()


def _loaded_app(sample: str = "breast_cancer") -> AppTest:
    at = _open(AppTest.from_file(str(APP), default_timeout=180).run(), "data.py")
    at.radio[0].set_value(SAMPLE_TITLES[sample]).run()
    at = _click(at, "Load sample")
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_without_data(page):
    at = _open(AppTest.from_file(str(APP), default_timeout=60).run(), page)
    assert not at.exception, at.exception


@pytest.mark.parametrize("sample", list(SAMPLE_TITLES))
@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_sample(page, sample):
    at = _open(_loaded_app(sample), page)
    assert not at.exception, at.exception


def test_prepare_train_and_report_flow():
    at = _open(_loaded_app(), "clean.py")
    at = _click(at, "Clean data")
    assert not at.exception, at.exception
    assert "Cleaned" in at.session_state["versions"]

    at = _click(_open(at, "smart_prep.py"), "Run Smart Auto Prep")
    assert not at.exception, at.exception
    assert "Smart prep" in at.session_state["versions"]

    at = _click(_open(at, "train.py"), "Train models")
    assert not at.exception, at.exception
    assert at.session_state["training"] is not None

    at = _open(at, "report.py")
    at = _click(at, "Build HTML report")
    at = _click(at, "Build PDF report")
    assert not at.exception, at.exception


def test_forecast_flow():
    at = _click(_open(_loaded_app("co2"), "forecast.py"), "Forecast")
    assert not at.exception, at.exception
