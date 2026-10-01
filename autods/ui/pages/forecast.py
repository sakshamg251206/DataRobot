"""Forecast: trend, seasonality, stationarity and an ARIMA forecast for one metric over time."""

import pandas as pd
import streamlit as st

from autods.core.dtypes import numeric_columns
from autods.core.timeseries import (
    AGGREGATIONS,
    FREQUENCIES,
    TimeSeriesError,
    adf_test,
    arima_forecast,
    build_series,
    date_column_candidates,
    decompose,
)
from autods.ui import charts, state
from autods.ui.components import READABLE, dataset_picker, page_header, require_data

page_header(
    "Forecast",
    "Track one number over time (sales, visitors, temperature…), see its trend and seasonal "
    "pattern, and project it forward with an ARIMA model.",
    eyebrow="Step 4 · Model",
)
require_data()

version_name, df = dataset_picker("ts_version", prefer=READABLE)
cache = state.page_cache()


@st.cache_data(show_spinner=False, max_entries=8)
def _date_candidates(data: pd.DataFrame) -> list[str]:
    return date_column_candidates(data)


dates = _date_candidates(df)
values = numeric_columns(df)
if not dates:
    st.info(
        "Forecasting needs a date or time column, and none was found in this version of the data. "
        "Try the **Mauna Loa CO₂** sample to see how it works.",
        icon=":material/event_busy:",
    )
    st.stop()
if not values:
    st.info("Forecasting needs a numeric column to track over time.", icon=":material/info:")
    st.stop()

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    date_col = c1.selectbox("Date column", dates)
    value_col = c2.selectbox("Value to track", [v for v in values if v != date_col])
    freq_label = c3.selectbox("Group by", list(FREQUENCIES), index=2)
    agg_label = c4.selectbox(
        "Combine rows with",
        list(AGGREGATIONS),
        help="Use Sum for totals (sales), Mean for levels (temperature, price).",
    )

freq = FREQUENCIES[freq_label]
try:
    info = build_series(df, date_col, value_col, freq, AGGREGATIONS[agg_label])
except TimeSeriesError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()
series = info.series

if len(series) < 3:
    st.warning(
        f"Only {len(series)} {freq_label.lower()} period(s) — choose a finer grouping.",
        icon=":material/warning:",
    )
    st.stop()

st.caption(
    f"{info.raw_points:,} rows → {len(series):,} {freq_label.lower()} periods "
    f"from {series.index.min():%Y-%m-%d} to {series.index.max():%Y-%m-%d}."
    + (f" {info.filled_gaps} empty period(s) were interpolated." if info.filled_gaps else "")
)
st.plotly_chart(charts.line(series))

trend_tab, forecast_tab = st.tabs(["Trend & seasonality", "Forecast"])

with trend_tab:
    try:
        parts = decompose(series, freq)
    except TimeSeriesError as exc:
        st.info(str(exc))
    else:
        st.plotly_chart(charts.decomposition(parts))
        st.caption(
            "**Trend** is the long-run direction, **Seasonal** the repeating yearly/weekly "
            "pattern, **Residual** what's left over (noise)."
        )

    try:
        adf = adf_test(series)
    except TimeSeriesError as exc:
        st.caption(str(exc))
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Stationary?", "Yes" if adf.is_stationary else "No")
        c2.metric("ADF p-value", f"{adf.p_value:.4f}")
        c3.metric("Suggested d", adf.recommended_d)
        st.caption(
            "A stationary series has a stable mean and variance. ARIMA's *d* parameter "
            "differences the series that many times to make it stationary."
        )

with forecast_tab:
    try:
        suggested_d = adf_test(series).recommended_d
    except TimeSeriesError:
        suggested_d = 1
    c1, c2, c3, c4 = st.columns(4)
    p = c1.number_input("p · autoregressive terms", 0, 5, 1)
    d = c2.number_input(
        "d · differencing", 0, 2, suggested_d, help="Pre-filled from the stationarity test."
    )
    q = c3.number_input("q · moving-average terms", 0, 5, 1)
    steps = c4.number_input(f"Periods ahead ({freq_label.lower()})", 1, 60, 12)

    if st.button("Forecast", type="primary", icon=":material/timeline:"):
        with st.spinner(f"Fitting ARIMA({p},{d},{q})…"):
            try:
                cache["forecast"] = (
                    (version_name, date_col, value_col, freq, agg_label),
                    arima_forecast(series, (int(p), int(d), int(q)), int(steps)),
                )
            except TimeSeriesError as exc:
                st.error(str(exc), icon=":material/error:")

    saved = cache.get("forecast")
    if saved and saved[0] == (version_name, date_col, value_col, freq, agg_label):
        result = saved[1]
        st.plotly_chart(charts.forecast(series, result.forecast))
        if result.backtest_mae is not None and result.naive_mae is not None:
            better = result.backtest_mae < result.naive_mae
            c1, c2 = st.columns(2)
            c1.metric(
                f"Back-test error (last {result.backtest_points} periods)",
                f"{result.backtest_mae:,.4g}",
                delta=f"{result.backtest_mae - result.naive_mae:+,.3g} vs naive",
                delta_color="inverse",
            )
            c2.metric(
                "Naive forecast error",
                f"{result.naive_mae:,.4g}",
                help="Error of simply repeating the last observed value.",
            )
            st.caption(
                "The model was refit without the most recent periods and asked to predict them. "
                + (
                    "It beat the naive baseline."
                    if better
                    else "It did **not** beat the naive baseline — treat this forecast with caution "
                    "or try different p, d, q values."
                )
            )
        table = result.forecast.rename(
            columns={"forecast": "Forecast", "lower": "Low (95%)", "upper": "High (95%)"}
        )
        st.dataframe(table.round(3))
        with st.expander("Model details"):
            st.text(result.summary)
