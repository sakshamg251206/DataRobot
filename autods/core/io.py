"""Reading uploaded files into a tidy DataFrame and flagging data quality issues."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import IO

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx")
MIN_ROWS = 5
LARGE_DATASET_ROWS = 200_000
HIGH_MISSING_RATIO = 0.90


class DataLoadError(ValueError):
    """Raised when a file cannot be turned into a usable DataFrame."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    filename: str
    warnings: list[str] = field(default_factory=list)
    renamed_columns: dict[str, str] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)


# ── Reading ────────────────────────────────────────────────────────────────────
def _extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DataLoadError("Could not decode the file. Please save it as UTF-8.")


def _sniff_delimiter(text: str) -> str:
    sample = text[:20_000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_table(file: IO[bytes] | bytes, filename: str) -> pd.DataFrame:
    """Read CSV/TSV/Excel bytes into a DataFrame.

    Handles UTF-8 (with or without BOM) and Windows encodings, and detects
    ``,``, ``;``, tab and ``|`` delimiters automatically.
    """
    ext = _extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file type '{ext or filename}'. "
            f"Please upload one of: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    raw = file if isinstance(file, bytes) else file.read()
    if not raw:
        raise DataLoadError("The uploaded file is empty.")

    try:
        if ext == ".xlsx":
            return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        text = _decode(raw)
        return pd.read_csv(io.StringIO(text), sep=_sniff_delimiter(text))
    except DataLoadError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The uploaded file has no data.") from exc
    except Exception as exc:
        raise DataLoadError(f"Could not read '{filename}': {exc}") from exc


# ── Standardisation ────────────────────────────────────────────────────────────
def standardize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return ``snake_case`` column names, guaranteed unique and non-empty.

    Example: ``" Total Sales ($)"`` -> ``"total_sales"``.
    Returns the renamed frame and a mapping of ``{old: new}`` for changed names.
    """
    cleaned = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    seen: dict[str, int] = {}
    new_names: list[str] = []
    for position, name in enumerate(cleaned):
        base = name or f"column_{position + 1}"
        if base in seen:
            seen[base] += 1
            candidate = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            candidate = base
        new_names.append(candidate)

    renamed = {
        str(old): new for old, new in zip(df.columns, new_names, strict=True) if str(old) != new
    }
    out = df.copy()
    out.columns = new_names
    return out, renamed


def drop_index_artifacts(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop ``Unnamed: 0``-style columns left behind by ``DataFrame.to_csv()``."""
    artifacts = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    return df.drop(columns=artifacts), artifacts


# ── Validation ─────────────────────────────────────────────────────────────────
def _fmt_cols(cols: list[str], limit: int = 6) -> str:
    shown = ", ".join(f"`{c}`" for c in cols[:limit])
    return shown + (f" and {len(cols) - limit} more" if len(cols) > limit else "")


def find_quality_issues(df: pd.DataFrame) -> list[str]:
    """Human-readable warnings about the raw data. An empty list means no issues."""
    issues: list[str] = []
    n_rows, n_cols = df.shape

    if n_rows < MIN_ROWS:
        issues.append(f"Only {n_rows} rows — too few for reliable statistics or modelling.")
    if n_cols < 2:
        issues.append("Only one column — most analyses need at least two.")
    if n_rows > LARGE_DATASET_ROWS:
        issues.append(
            f"{n_rows:,} rows is a large dataset; some charts are sampled and "
            "model training may take a while."
        )

    empty = df.columns[df.isna().all()].tolist()
    if empty:
        issues.append(f"Completely empty columns: {_fmt_cols(empty)}. Cleaning will drop them.")

    mostly_missing = [
        c for c in df.columns if c not in empty and df[c].isna().mean() > HIGH_MISSING_RATIO
    ]
    if mostly_missing:
        issues.append(
            f"Columns more than {HIGH_MISSING_RATIO:.0%} empty: {_fmt_cols(mostly_missing)}. "
            "Consider dropping them."
        )

    constant = [c for c in df.columns if c not in empty and df[c].nunique(dropna=True) == 1]
    if constant:
        issues.append(f"Columns with a single value carry no information: {_fmt_cols(constant)}.")

    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        issues.append(f"{n_dupes:,} duplicate rows. Cleaning will remove them.")

    return issues


# ── Entry point ────────────────────────────────────────────────────────────────
def load_dataset(file: IO[bytes] | bytes, filename: str, max_rows: int | None = None) -> LoadResult:
    """Read, standardise and validate an uploaded dataset.

    Raises :class:`DataLoadError` with a user-friendly message on failure.
    """
    df = read_table(file, filename)

    df = df.dropna(how="all")  # spreadsheet exports often have trailing blank rows
    if df.empty or df.shape[1] == 0:
        raise DataLoadError("The file contains no data rows.")

    df, dropped = drop_index_artifacts(df)
    if df.shape[1] == 0:
        raise DataLoadError("The file only contains index columns.")
    df, renamed = standardize_column_names(df)
    df = df.reset_index(drop=True)

    warnings = find_quality_issues(df)
    if max_rows is not None and len(df) > max_rows:
        warnings.insert(
            0,
            f"Dataset truncated to the first {max_rows:,} of {len(df):,} rows "
            "(limit set by AUTODS_MAX_ROWS).",
        )
        df = df.head(max_rows)

    return LoadResult(
        df=df,
        filename=filename,
        warnings=warnings,
        renamed_columns=renamed,
        dropped_columns=dropped,
    )
