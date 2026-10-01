"""Shared fixtures: small, deterministic datasets that exercise edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Typical real-world mess: money as text, dates as text, gaps, a duplicate row."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 6],
            "signup_date": [
                "2024-01-05",
                "2024-02-10",
                "2024-03-15",
                None,
                "2024-05-20",
                "2024-06-25",
                "2024-06-25",
            ],
            "revenue": ["$1,200", "$950", "$15,000", "$1,100", None, "$1,050", "$1,050"],
            "plan": ["basic", "pro", "pro", None, "basic", "basic", "basic"],
            "visits": [10.0, 12.0, np.nan, 11.0, 300.0, 9.0, 9.0],
            "constant": ["x"] * 7,
            "empty": [None] * 7,
        }
    )


@pytest.fixture
def classification_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    color = rng.choice(["red", "green", "blue"], size=n)
    signal = x1 + 0.5 * x2 + (color == "red") * 1.5
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": np.where(rng.random(n) < 0.05, np.nan, x2),
            "color": color,
            "when": pd.date_range("2023-01-01", periods=n, freq="D"),
            "label": np.where(signal > 0.5, "yes", "no"),
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 250
    x1 = rng.uniform(0, 10, size=n)
    x2 = rng.uniform(0, 5, size=n)
    group = rng.choice(["a", "b"], size=n)
    y = 3 * x1 - 2 * x2 + (group == "a") * 4 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "group": group, "y": y})


@pytest.fixture
def monthly_series_df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    dates = pd.date_range("2018-01-01", periods=60, freq="MS")
    trend = np.linspace(100, 160, 60)
    season = 10 * np.sin(2 * np.pi * np.arange(60) / 12)
    return pd.DataFrame(
        {"date": dates.astype(str), "sales": trend + season + rng.normal(scale=2, size=60)}
    )
