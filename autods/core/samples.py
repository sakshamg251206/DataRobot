"""Built-in sample datasets so new users can try every feature without their own file.

All samples are real, public datasets that ship inside scikit-learn and
statsmodels, so no network access is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Sample:
    key: str
    title: str
    description: str
    suggested_target: str
    loader: Callable[[], pd.DataFrame]


def _breast_cancer() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["diagnosis"] = df.pop("target").map({0: "malignant", 1: "benign"})
    return df


def _diabetes() -> pd.DataFrame:
    from sklearn.datasets import load_diabetes

    df = load_diabetes(as_frame=True, scaled=False).frame.copy()
    df["sex"] = df["sex"].map({1.0: "group_1", 2.0: "group_2"})
    return df.rename(columns={"target": "disease_progression", "bp": "blood_pressure"})


def _co2() -> pd.DataFrame:
    from statsmodels.datasets import co2

    df = co2.load_pandas().data.rename_axis("date").reset_index()
    return df.rename(columns={"co2": "co2_ppm"})


SAMPLES: dict[str, Sample] = {
    s.key: s
    for s in (
        Sample(
            "breast_cancer",
            "Breast cancer diagnosis",
            "569 tumour measurements labelled malignant or benign "
            "(UCI Wisconsin dataset). A classification problem.",
            "diagnosis",
            _breast_cancer,
        ),
        Sample(
            "diabetes",
            "Diabetes progression",
            "442 patients with age, BMI, blood pressure and blood-serum readings, "
            "and a measure of disease progression one year later. A regression problem.",
            "disease_progression",
            _diabetes,
        ),
        Sample(
            "co2",
            "Mauna Loa CO₂",
            "Weekly atmospheric CO₂ readings from 1958 to 2001, with some missing weeks. "
            "A time series with a strong trend and yearly seasonality.",
            "co2_ppm",
            _co2,
        ),
    )
}


def load_sample(key: str) -> pd.DataFrame:
    return SAMPLES[key].loader()
