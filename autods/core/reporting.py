"""Shareable reports: a self-contained interactive HTML page and a compact PDF."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from autods import APP_NAME, __version__
from autods.core.dtypes import categorical_columns, numeric_columns
from autods.core.profiling import missing_report, overview, top_correlations

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    FPDF_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    FPDF_AVAILABLE = False

MAX_CHART_COLUMNS = 4


@dataclass
class ReportContext:
    """Everything a report needs. Only ``df`` is required."""

    df: pd.DataFrame
    dataset_name: str = "dataset"
    raw_shape: tuple[int, int] | None = None
    processing_log: list[str] = field(default_factory=list)
    model_summary: pd.DataFrame | None = None
    model_target: str | None = None
    model_task: str | None = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _esc(value: object) -> str:
    return html.escape(str(value))


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="table", escape=True, na_rep="—")


_CSS = """
:root { --fg:#0f172a; --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc; --card:#fff; --accent:#4f46e5; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e2e8f0; --muted:#94a3b8; --line:#334155; --bg:#0f172a; --card:#1e293b; --accent:#818cf8; }
}
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px; font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;
       background:var(--bg); color:var(--fg); }
main { max-width:1100px; margin:0 auto; }
header { margin-bottom:24px; }
h1 { font-size:1.8rem; margin:0 0 4px; }
h2 { font-size:1.2rem; margin:32px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--line); }
.muted { color:var(--muted); font-size:.9rem; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.card .v { font-size:1.5rem; font-weight:650; }
.card .l { color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }
.table { width:100%; border-collapse:collapse; background:var(--card); font-size:.9rem; }
.table th, .table td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; }
.scroll { overflow-x:auto; }
ol.log li { margin:4px 0; }
.chart { background:var(--card); border:1px solid var(--line); border-radius:10px; margin:12px 0; padding:8px; }
footer { margin-top:40px; color:var(--muted); font-size:.8rem; }
"""


def build_html_report(ctx: ReportContext) -> str:
    """Return a standalone HTML document. All user-provided text is escaped."""
    df = ctx.df
    stats = overview(df)
    nums = numeric_columns(df)
    cards = [
        ("Rows", f"{stats['rows']:,}"),
        ("Columns", stats["columns"]),
        ("Numeric", stats["numeric"]),
        ("Text / categorical", stats["categorical"]),
        ("Missing cells", f"{stats['missing_cells']:,}"),
        ("Duplicate rows", f"{stats['duplicate_rows']:,}"),
    ]
    if ctx.raw_shape:
        cards.insert(0, ("Original shape", f"{ctx.raw_shape[0]:,} × {ctx.raw_shape[1]}"))

    parts: list[str] = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_esc(ctx.dataset_name)} — data report</title><style>{_CSS}</style></head><body><main>",
        f"<header><h1>Data report: {_esc(ctx.dataset_name)}</h1>",
        f"<div class='muted'>Generated {_timestamp()} by {APP_NAME} v{__version__}</div></header>",
        "<h2>Overview</h2><div class='grid'>",
        *(
            f"<div class='card'><div class='v'>{_esc(v)}</div><div class='l'>{_esc(k)}</div></div>"
            for k, v in cards
        ),
        "</div>",
    ]

    if ctx.processing_log:
        parts.append("<h2>Processing steps</h2><ol class='log'>")
        parts += [f"<li>{_esc(line)}</li>" for line in ctx.processing_log]
        parts.append("</ol>")

    missing = missing_report(df)
    parts.append("<h2>Missing values</h2>")
    parts.append(
        f"<div class='scroll'>{_table(missing)}</div>"
        if not missing.empty
        else "<p>No missing values.</p>"
    )

    if ctx.model_summary is not None and not ctx.model_summary.empty:
        parts.append("<h2>Model comparison</h2>")
        if ctx.model_target:
            parts.append(
                f"<p class='muted'>{_esc(ctx.model_task or '')} model predicting "
                f"<b>{_esc(ctx.model_target)}</b>, scored on a held-out test set.</p>"
            )
        parts.append(f"<div class='scroll'>{_table(ctx.model_summary)}</div>")

    figures: list[go.Figure] = []
    if len(nums) >= 2:
        corr_cols = nums[:15]
        fig = px.imshow(
            df[corr_cols].corr().round(2),
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Correlation matrix",
        )
        figures.append(fig)
        top = top_correlations(df, corr_cols, n=5)
        if not top.empty:
            parts.append(f"<h2>Strongest correlations</h2><div class='scroll'>{_table(top)}</div>")
    for col in nums[:MAX_CHART_COLUMNS]:
        figures.append(px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}"))
    for col in categorical_columns(df)[:3]:
        counts = df[col].value_counts().head(20).rename_axis(col).reset_index(name="count")
        figures.append(px.bar(counts, x=col, y="count", title=f"Most common values: {col}"))

    if figures:
        parts.append("<h2>Charts</h2>")
        for i, fig in enumerate(figures):
            fig.update_layout(template="plotly_white", margin={"t": 50, "l": 40, "r": 20, "b": 40})
            include = "cdn" if i == 0 else False
            parts.append(
                f"<div class='chart'>{fig.to_html(full_html=False, include_plotlyjs=include)}</div>"
            )

    parts.append(
        "<footer>Charts are interactive: hover for values, drag to zoom, double-click to reset. "
        "An internet connection is needed to load the charting library.</footer></main></body></html>"
    )
    return "\n".join(parts)


# ── PDF ────────────────────────────────────────────────────────────────────────
def _latin1(value: object) -> str:
    """The core PDF fonts are Latin-1 only; replace anything else."""
    text = str(value).replace("—", "-").replace("–", "-").replace("→", "->").replace("×", "x")
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf_report(ctx: ReportContext) -> bytes:
    """Return a compact, text-only PDF summary."""
    if not FPDF_AVAILABLE:  # pragma: no cover
        raise RuntimeError("Install `fpdf2` to build PDF reports.")
    df = ctx.df
    stats = overview(df)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def heading(text: str) -> None:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)

    def line(text: str) -> None:
        pdf.multi_cell(0, 6, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(
        0, 12, _latin1(f"Data report: {ctx.dataset_name}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(
        0,
        6,
        _latin1(f"Generated {_timestamp()} by {APP_NAME} v{__version__}"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    heading("Overview")
    if ctx.raw_shape:
        line(f"Original: {ctx.raw_shape[0]:,} rows x {ctx.raw_shape[1]} columns")
    line(f"Current: {stats['rows']:,} rows x {stats['columns']} columns")
    line(
        f"Numeric columns: {stats['numeric']}   Text columns: {stats['categorical']}   "
        f"Missing cells: {stats['missing_cells']:,}   Duplicate rows: {stats['duplicate_rows']:,}"
    )

    if ctx.processing_log:
        heading("Processing steps")
        for i, entry in enumerate(ctx.processing_log, 1):
            line(f"{i}. {entry.replace('`', '')}")

    heading("Missing values")
    missing = missing_report(df)
    if missing.empty:
        line("No missing values.")
    for _, row in missing.head(15).iterrows():
        line(f"{row['Column']}: {int(row['Missing']):,} ({row['Missing %']}%)")

    heading("Numeric columns")
    nums = numeric_columns(df)
    if not nums:
        line("No numeric columns.")
    for col in nums[:12]:
        s = df[col].dropna()
        if s.empty:
            continue
        line(
            f"{col}: mean {s.mean():.4g}, std {s.std():.4g}, "
            f"min {s.min():.4g}, median {s.median():.4g}, max {s.max():.4g}"
        )

    cats = categorical_columns(df)
    if cats:
        heading("Text / categorical columns")
        for col in cats[:10]:
            counts = df[col].value_counts()
            top = f"'{counts.index[0]}' ({counts.iloc[0]:,} rows)" if not counts.empty else "n/a"
            line(f"{col}: {df[col].nunique():,} distinct values, most common {top}")

    if ctx.model_summary is not None and not ctx.model_summary.empty:
        heading(f"Model comparison ({ctx.model_task or ''} - target: {ctx.model_target or ''})")
        for _, row in ctx.model_summary.iterrows():
            metrics = ", ".join(
                f"{k} {v:.3f}" for k, v in row.items() if k != "Model" and isinstance(v, float)
            )
            line(f"{row['Model']}: {metrics}")

    return bytes(pdf.output())
