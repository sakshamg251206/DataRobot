"""Runtime configuration read from environment variables.

Every setting has a safe default so the app runs with zero configuration.
See ``.env.example`` for documentation of each variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Application settings. Construct with :func:`get_settings`."""

    google_api_key: str | None
    gemini_model: str
    enable_code_agent: bool
    max_rows: int
    random_state: int


def get_settings() -> Settings:
    """Read settings from the environment (re-read on every call; cheap)."""
    return Settings(
        google_api_key=os.environ.get("GOOGLE_API_KEY") or None,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        enable_code_agent=_env_bool("ENABLE_CODE_AGENT", default=False),
        max_rows=_env_int("AUTODS_MAX_ROWS", 1_000_000),
        random_state=_env_int("AUTODS_RANDOM_STATE", 42),
    )
