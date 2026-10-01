"""Plotly figure builders.

Figures inherit Streamlit's light/dark theme when rendered, so these builders
only set layout details and a consistent accent palette.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ACCENT = "#4F46E5"
POSITIVE = "#10B981"
NEGATIVE = "#EF4444"
NEUTRAL = "#94A3B8"
SEQUENCE = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#06B6D4", "#8B5CF6", "#EC4899", "#84CC16"]
DIVERGING = "RdBu_r"
MAX_POINTS = 5_000


def _finish(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"t": 48 if title else 16, "l": 8, "r": 8, "b": 8},
        title=title,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        colorway=SEQUENCE,
    )
    return fig


def sample_for_plot(df: pd.DataFrame, n: int = MAX_POINTS) -> pd.DataFrame:
    return df.sample(n, random_state=0) if len(df) > n else df


def missing_bar(report: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        report,
        x="Missing %",
        y="Column",
        orientation="h",
        color="Missing %",
        color_continuous_scale=[POSITIVE, "#F59E0B", NEGATIVE],
        range_color=[0, 100],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return _finish(fig, height=max(240, 28 * len(report) + 60))


def skew_bar(skews: pd.Series, threshold: float) -> go.Figure:
    frame = skews.rename("Skewness").rename_axis("Column").reset_index()
    colors = [NEGATIVE if abs(v) >= threshold else POSITIVE for v in frame["Skewness"]]
    fig = go.Figure(go.Bar(x=frame["Column"], y=frame["Skewness"], marker_color=colors))
    fig.add_hline(y=threshold, line_dash="dot", line_color=NEUTRAL)
    fig.add_hline(y=-threshold, line_dash="dot", line_color=NEUTRAL)
    return _finish(fig, height=320)


def histogram(df: pd.DataFrame, column: str, color: str | None, bins: int) -> go.Figure:
    fig = px.histogram(
        df,
        x=column,
        color=color,
        nbins=bins,
        marginal="box",
        opacity=0.8,
        barmode="overlay",
        color_discrete_sequence=SEQUENCE,
    )
    return _finish(fig, height=420)


def category_bar(df: pd.DataFrame, column: str, top: int = 25) -> go.Figure:
    counts = df[column].value_counts().head(top).rename_axis(column).reset_index(name="Count")
    fig = px.bar(counts, x=column, y="Count", color_discrete_sequence=[ACCENT])
    return _finish(fig, height=400)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr.round(2),
        text_auto=len(corr) <= 15,
        aspect="auto",
        color_continuous_scale=DIVERGING,
        zmin=-1,
        zmax=1,
    )
    return _finish(fig, height=max(380, 32 * len(corr) + 120))


def scatter_matrix(df: pd.DataFrame, columns: list[str], color: str | None) -> go.Figure:
    fig = px.scatter_matrix(
        sample_for_plot(df), dimensions=columns, color=color, color_discrete_sequence=SEQUENCE
    )
    fig.update_traces(
        diagonal_visible=False, showupperhalf=False, marker={"size": 4, "opacity": 0.6}
    )
    return _finish(fig, height=180 * len(columns) + 80)


def relationship(df: pd.DataFrame, a: str, b: str) -> go.Figure:
    """Pick the right chart for two columns: scatter, box plot or heatmap."""
    a_num = pd.api.types.is_numeric_dtype(df[a])
    b_num = pd.api.types.is_numeric_dtype(df[b])
    if a_num and b_num:
        fig = px.scatter(
            sample_for_plot(df), x=a, y=b, opacity=0.6, color_discrete_sequence=[ACCENT]
        )
        clean = df[[a, b]].dropna()
        if len(clean) > 2 and clean[a].nunique() > 1:
            slope, intercept = np.polyfit(clean[a], clean[b], 1)
            xs = np.linspace(clean[a].min(), clean[a].max(), 50)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=slope * xs + intercept,
                    mode="lines",
                    name="Linear trend",
                    line={"color": NEGATIVE, "dash": "dash"},
                )
            )
        return _finish(fig, height=440)
    if a_num or b_num:
        num, cat = (a, b) if a_num else (b, a)
        top = df[cat].value_counts().index[:20]
        subset = df[df[cat].isin(top)]
        fig = px.box(subset, x=cat, y=num, color=cat, color_discrete_sequence=SEQUENCE)
        fig.update_layout(showlegend=False)
        return _finish(fig, height=440)
    top_a = df[a].value_counts().index[:15]
    top_b = df[b].value_counts().index[:15]
    subset = df[df[a].isin(top_a) & df[b].isin(top_b)]
    fig = px.imshow(
        pd.crosstab(subset[a], subset[b]),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
    )
    return _finish(fig, height=440)


def metric_bar(leaderboard: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.bar(
        leaderboard,
        x=metric,
        y="Model",
        orientation="h",
        text=metric,
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return _finish(fig, height=80 + 50 * len(leaderboard))


def confusion_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        labels={"x": "Predicted", "y": "Actual", "color": "Rows"},
    )
    fig.update_layout(coloraxis_showscale=False)
    return _finish(fig, height=max(320, 40 * len(matrix) + 120))


def actual_vs_predicted(actual: pd.Series, predicted: np.ndarray) -> go.Figure:
    fig = px.scatter(
        x=actual,
        y=predicted,
        opacity=0.6,
        labels={"x": "Actual", "y": "Predicted"},
        color_discrete_sequence=[ACCENT],
    )
    low = float(min(np.min(actual), np.min(predicted)))
    high = float(max(np.max(actual), np.max(predicted)))
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Perfect prediction",
            line={"color": NEGATIVE, "dash": "dash"},
        )
    )
    return _finish(fig, height=420)


def importance_bar(importance: pd.DataFrame, x: str = "Importance") -> go.Figure:
    fig = px.bar(importance, x=x, y="Feature", orientation="h", color_discrete_sequence=[ACCENT])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return _finish(fig, height=60 + 26 * len(importance))


def contribution_bar(contrib: pd.DataFrame) -> go.Figure:
    """Signed per-feature SHAP contributions for one prediction."""
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in contrib["Contribution"]]
    fig = go.Figure(
        go.Bar(
            x=contrib["Contribution"], y=contrib["Feature"], orientation="h", marker_color=colors
        )
    )
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": contrib["Feature"][::-1]})
    return _finish(fig, height=60 + 26 * len(contrib))


def learning_curve(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, std, color in (
        ("Train score", "Train std", POSITIVE),
        ("Validation score", "Validation std", ACCENT),
    ):
        fig.add_trace(
            go.Scatter(
                x=curve["Training rows"],
                y=curve[name],
                name=name,
                mode="lines+markers",
                line={"color": color},
                error_y={"type": "data", "array": curve[std], "visible": True},
            )
        )
    fig.update_layout(xaxis_title="Training rows", yaxis_title="Score")
    return _finish(fig, height=380)


def line(series: pd.Series) -> go.Figure:
    fig = px.line(
        series.reset_index(), x=series.index.name, y=series.name, color_discrete_sequence=[ACCENT]
    )
    return _finish(fig, height=380)


def decomposition(parts: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, subplot_titles=list(parts.columns), vertical_spacing=0.06
    )
    for i, col in enumerate(parts.columns, start=1):
        fig.add_trace(
            go.Scatter(
                x=parts.index,
                y=parts[col],
                name=col,
                mode="lines",
                line={"color": SEQUENCE[i - 1], "width": 1.5},
            ),
            row=i,
            col=1,
        )
    fig.update_layout(showlegend=False)
    return _finish(fig, height=640)


def forecast(history: pd.Series, prediction: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index, y=history, name="History", mode="lines", line={"color": NEUTRAL}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(prediction.index) + list(prediction.index[::-1]),
            y=list(prediction["upper"]) + list(prediction["lower"][::-1]),
            fill="toself",
            fillcolor="rgba(79,70,229,0.15)",
            line={"color": "rgba(0,0,0,0)"},
            name="95% interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prediction.index,
            y=prediction["forecast"],
            name="Forecast",
            mode="lines+markers",
            line={"color": ACCENT, "width": 2.5},
        )
    )
    fig.update_layout(hovermode="x unified")
    return _finish(fig, height=440)
