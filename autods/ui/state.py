"""Typed access to Streamlit session state.

The app keeps several *versions* of the dataset (original, cleaned, …) so
users can compare and roll back. Every page reads and writes state through
these helpers instead of touching ``st.session_state`` keys directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import streamlit as st

from autods.ai.client import ChatTurn
from autods.config import get_settings
from autods.core.modeling import TrainingResult

ORIGINAL = "Original"
CLEANED = "Cleaned"
SMART = "Smart prep"
ENGINEERED = "Engineered"
VERSION_ORDER = (ORIGINAL, CLEANED, SMART, ENGINEERED)
VERSION_HELP = {
    ORIGINAL: "The data exactly as loaded (column names tidied).",
    CLEANED: "Output of the Clean page.",
    SMART: "Fully numeric output of Smart Auto Prep.",
    ENGINEERED: "Output of the Engineer features page.",
}


@dataclass
class DatasetVersion:
    df: pd.DataFrame
    log: list[str] = field(default_factory=list)
    target: str | None = None
    meta: dict[str, float] = field(default_factory=dict)


def _state() -> Any:
    ss = st.session_state
    ss.setdefault("versions", {})
    ss.setdefault("dataset_name", None)
    ss.setdefault("load_warnings", [])
    ss.setdefault("renamed_columns", {})
    ss.setdefault("suggested_target", None)
    ss.setdefault("training", None)
    ss.setdefault("training_source", None)
    ss.setdefault("chat", [])
    ss.setdefault("api_key", "")
    ss.setdefault("ai_cache", {})
    ss.setdefault("training_cache", {})
    ss.setdefault("page_cache", {})
    return ss


# ── Dataset versions ───────────────────────────────────────────────────────────
def has_data() -> bool:
    return ORIGINAL in _state().versions


def dataset_name() -> str:
    return _state().dataset_name or "dataset"


def load_warnings() -> list[str]:
    return list(_state().load_warnings)


def versions() -> dict[str, DatasetVersion]:
    stored = _state().versions
    return {name: stored[name] for name in VERSION_ORDER if name in stored}


def get_version(name: str) -> DatasetVersion | None:
    return _state().versions.get(name)


def latest_version_name() -> str | None:
    names = list(versions())
    return names[-1] if names else None


def renamed_columns() -> dict[str, str]:
    return dict(_state().renamed_columns)


def load_sample_dataset(key: str) -> None:
    from autods.core.io import find_quality_issues
    from autods.core.samples import SAMPLES, load_sample

    df = load_sample(key)
    set_original(df, SAMPLES[key].title, find_quality_issues(df))
    _state().suggested_target = SAMPLES[key].suggested_target


def set_original(
    df: pd.DataFrame,
    name: str,
    warnings: list[str] | None = None,
    renamed: dict[str, str] | None = None,
) -> None:
    """Load a new dataset. Everything derived from the previous one is discarded."""
    reset()
    ss = _state()
    ss.versions = {ORIGINAL: DatasetVersion(df=df)}
    ss.dataset_name = name
    ss.load_warnings = warnings or []
    ss.renamed_columns = renamed or {}


def set_version(
    name: str,
    df: pd.DataFrame,
    log: list[str],
    target: str | None = None,
    meta: dict[str, float] | None = None,
) -> None:
    ss = _state()
    ss.versions[name] = DatasetVersion(df=df, log=log, target=target, meta=meta or {})
    ss.page_cache = {}
    if ss.training_source == name:  # results were computed on data that just changed
        ss.training = None
        ss.training_source = None
        ss.training_cache = {}


def reset() -> None:
    ss = _state()
    ss.versions = {}
    ss.dataset_name = None
    ss.load_warnings = []
    ss.renamed_columns = {}
    ss.suggested_target = None
    ss.training = None
    ss.training_source = None
    ss.training_cache = {}
    ss.chat = []
    ss.ai_cache = {}
    ss.page_cache = {}


def has_known_target() -> bool:
    """Whether the user (or a sample) has indicated what they want to predict."""
    ss = _state()
    return bool(ss.training or ss.suggested_target or any(v.target for v in versions().values()))


def default_target(columns: list[str]) -> str:
    """Best guess for the column to predict, consistent across pages.

    Order: the target of the most recent training run, a target chosen on any
    preparation page, the sample's suggested target, the original's last column.
    """
    ss = _state()
    candidates: list[str | None] = [ss.training.target if ss.training else None]
    candidates += [v.target for v in reversed(list(versions().values()))]
    candidates.append(ss.suggested_target)
    original = ss.versions.get(ORIGINAL)
    if original is not None and len(original.df.columns):
        candidates.append(str(original.df.columns[-1]))
    return next((c for c in candidates if c and c in columns), columns[-1])


# ── Modelling ──────────────────────────────────────────────────────────────────
def training() -> TrainingResult | None:
    return _state().training


def training_source() -> str | None:
    return _state().training_source


def set_training(result: TrainingResult, source: str) -> None:
    ss = _state()
    ss.training = result
    ss.training_source = source
    ss.training_cache = {}


def training_cache() -> dict[str, Any]:
    """Per-result cache for slow, derived artefacts (SHAP values, learning curves, exports)."""
    cache: dict[str, Any] = _state().training_cache
    return cache


# ── AI ─────────────────────────────────────────────────────────────────────────
def api_key() -> str:
    """Per-session key typed by the user, falling back to a server-side key."""
    typed = _state().api_key
    if typed:
        return str(typed)
    configured = get_settings().google_api_key
    if configured:
        return configured
    try:
        return str(st.secrets.get("GOOGLE_API_KEY", ""))
    except Exception:  # no secrets file
        return ""


def set_api_key(value: str) -> None:
    _state().api_key = value.strip()


def chat_history() -> list[ChatTurn]:
    return list(_state().chat)


def append_chat(turn: ChatTurn) -> None:
    _state().chat.append(turn)


def clear_chat() -> None:
    _state().chat = []


def page_cache() -> dict[str, Any]:
    """Scratch space for page results (forecasts, built reports); cleared with the dataset."""
    cache: dict[str, Any] = _state().page_cache
    return cache


def ai_cache() -> dict[str, str]:
    cache: dict[str, str] = _state().ai_cache
    return cache
