import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────────────────
MIN_POINTS_ARIMA      = 20    # minimum data points required for ARIMA
MIN_POINTS_DECOMPOSE  = 24    # minimum for seasonal decomposition
ADF_SIGNIFICANCE      = 0.05  # p-value threshold for stationarity test


# ── Frequency options ──────────────────────────────────────────────────────────
FREQ_OPTIONS = {
    "Daily":     "D",
    "Weekly":    "W",
    "Monthly":   "ME",    # "ME" replaces deprecated "M" in pandas >= 2.2
    "Quarterly": "QE",
    "Yearly":    "YE",
}

AGG_OPTIONS = {
    "Mean":  "mean",
    "Sum":   "sum",
    "Last":  "last",
    "Max":   "max",
    "Min":   "min",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return all datetime-dtype columns."""
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def _resample(
    series: pd.Series,
    freq:   str,
    agg:    str,
) -> pd.Series:
    """Resample a time-indexed series with the chosen aggregation function."""
    resampled = series.resample(freq)
    return getattr(resampled, agg)().dropna()


def _generate_future_dates(
    last_date: pd.Timestamp,
    freq:      str,
    steps:     int,
) -> pd.DatetimeIndex:
    """
    Generate `steps` future period-end dates after `last_date`.
    Uses pd.date_range with the same frequency as the resampled series.
    """
    return pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]


# ── Stationarity check ─────────────────────────────────────────────────────────
def _adf_test(series: pd.Series) -> tuple[bool, float, float]:
    """
    Augmented Dickey-Fuller test for stationarity.
    Returns (is_stationary, adf_stat, p_value).
    """
    result       = adfuller(series.dropna(), autolag="AIC")
    adf_stat     = result[0]
    p_value      = result[1]
    is_stationary = p_value < ADF_SIGNIFICANCE
    return is_stationary, adf_stat, p_value


def show_stationarity(series: pd.Series, label: str):
    """Display ADF test result and recommend d parameter for ARIMA."""
    is_stat, adf_stat, p_val = _adf_test(series)

    col1, col2, col3 = st.columns(3)
    col1.metric("ADF Statistic", f"{adf_stat:.4f}")
    col2.metric("p-value",       f"{p_val:.4f}")
    col3.metric("Stationary?",   "Yes ✅" if is_stat else "No ❌")

    if is_stat:
        st.success(
            f"`{label}` is stationary (p={p_val:.4f} < {ADF_SIGNIFICANCE}). "
            "Use d=0 in ARIMA."
        )
    else:
        st.warning(
            f"`{label}` is NOT stationary (p={p_val:.4f}). "
            "Use d=1 in ARIMA to difference the series. "
            "If still non-stationary after d=1, try d=2."
        )

    return is_stat


# ── Decomposition ──────────────────────────────────────────────────────────────
def show_decomposition(series: pd.Series, freq_label: str):
    """
    Decompose the series into trend, seasonality, and residual.
    Requires at least MIN_POINTS_DECOMPOSE data points.
    """
    if len(series) < MIN_POINTS_DECOMPOSE:
        st.info(
            f"Seasonal decomposition requires at least {MIN_POINTS_DECOMPOSE} "
            f"data points (current: {len(series)})."
        )
        return

    try:
        # period: number of observations per seasonal cycle
        period_map = {"D": 7, "W": 52, "ME": 12, "QE": 4, "YE": 1}
        period     = period_map.get(freq_label, 12)

        if len(series) < 2 * period:
            st.info(
                f"Need at least {2 * period} data points for decomposition "
                f"with period={period}."
            )
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decomp = seasonal_decompose(series, model="additive", period=period)

        fig = go.Figure()
        components = {
            "Observed":   decomp.observed,
            "Trend":      decomp.trend,
            "Seasonal":   decomp.seasonal,
            "Residual":   decomp.resid,
        }
        colors = ["#1d9e75", "#378add", "#e24b4a", "#888780"]

        for (name, comp), color in zip(components.items(), colors):
            fig.add_trace(go.Scatter(
                x=comp.index, y=comp.values,
                name=name, mode="lines",
                line=dict(color=color, width=1.5),
            ))

        fig.update_layout(
            title="Seasonal decomposition (additive)",
            xaxis_title="Date",
            yaxis_title="Value",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Trend = long-run direction. "
            "Seasonal = repeating periodic pattern. "
            "Residual = unexplained noise."
        )

    except Exception as e:
        st.warning(f"Decomposition failed: {e}")


# ── ARIMA forecast ─────────────────────────────────────────────────────────────
def run_arima_forecast(
    series:     pd.Series,
    freq:       str,
    steps:      int,
    p: int, d: int, q: int,
    target_col: str,
    dt_col:     str,
):
    """
    Fit ARIMA(p,d,q) on the full resampled series and forecast `steps` periods.
    Plots historical data + point forecast + 95% confidence interval.
    """
    if not STATSMODELS_AVAILABLE:
        st.error("Install `statsmodels` to use ARIMA forecasting.")
        return

    if len(series) < MIN_POINTS_ARIMA:
        st.warning(
            f"ARIMA requires at least {MIN_POINTS_ARIMA} data points "
            f"(current: {len(series)} after resampling). "
            "Try a finer frequency or provide more data."
        )
        return

    with st.spinner(f"Fitting ARIMA({p},{d},{q})..."):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model     = ARIMA(series, order=(p, d, q))
                fit       = model.fit()
                forecast  = fit.get_forecast(steps=steps)

            f_mean = forecast.predicted_mean
            f_ci   = forecast.conf_int(alpha=0.05)  # 95% CI

            # Generate proper future date index
            future_dates = _generate_future_dates(series.index[-1], freq, steps)
            f_mean.index = future_dates
            f_ci.index   = future_dates

            # Build figure
            fig = go.Figure()

            # Historical
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values,
                name="Historical", mode="lines",
                line=dict(color="#1d9e75", width=2),
            ))

            # Confidence interval band
            fig.add_trace(go.Scatter(
                x=pd.concat([
                    pd.Series(future_dates),
                    pd.Series(future_dates[::-1])
                ]),
                y=pd.concat([f_ci.iloc[:, 1], f_ci.iloc[:, 0].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(55,138,221,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% Confidence interval",
                showlegend=True,
            ))

            # Point forecast
            fig.add_trace(go.Scatter(
                x=future_dates, y=f_mean.values,
                name=f"Forecast ({steps} periods)",
                mode="lines+markers",
                line=dict(color="#378add", width=2, dash="dot"),
                marker=dict(size=7),
            ))

            fig.update_layout(
                title=f"ARIMA({p},{d},{q}) forecast — {target_col}",
                xaxis_title=dt_col,
                yaxis_title=target_col,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table
            st.write("**Forecast values:**")
            forecast_df = pd.DataFrame({
                "Period":       future_dates,
                "Forecast":     f_mean.values.round(4),
                "Lower 95% CI": f_ci.iloc[:, 0].values.round(4),
                "Upper 95% CI": f_ci.iloc[:, 1].values.round(4),
            })
            st.dataframe(forecast_df, use_container_width=True)

            # Model summary in expander
            with st.expander("View ARIMA model summary"):
                st.text(fit.summary().as_text())

        except Exception as e:
            st.error(
                f"ARIMA fitting failed: {e}  \n"
                "Try adjusting p, d, q values or using a different frequency."
            )


# ── Main entry point ───────────────────────────────────────────────────────────
def run_time_series_analysis(df: pd.DataFrame):
    st.subheader("Time Series Analysis & Forecasting")

    if not STATSMODELS_AVAILABLE:
        st.error(
            "Install `statsmodels` to use time series features: "
            "`pip install statsmodels`"
        )

    # ── Column selection ───────────────────────────────────────────────────────
    dt_cols = _get_datetime_columns(df)

    if not dt_cols:
        st.warning(
            "No datetime columns detected. "
            "Run the cleaning pipeline first (Step 3 or Step 4) to convert "
            "date-like columns to datetime format."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        dt_col = st.selectbox(
            "Date/time column", dt_cols, key="ts_dt_col"
        )
    with col2:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.error("No numeric columns available to analyse over time.")
            return
        target_col = st.selectbox(
            "Metric to analyse", num_cols, key="ts_target"
        )

    # ── Resampling options ─────────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        freq_label = st.selectbox(
            "Resample frequency", list(FREQ_OPTIONS.keys()),
            index=2, key="ts_freq"
        )
    with col4:
        agg_label = st.selectbox(
            "Aggregation method", list(AGG_OPTIONS.keys()),
            index=0, key="ts_agg",
            help="Sum for totals (sales), Mean for averages (temp), Last for snapshots (price).",
        )

    freq = FREQ_OPTIONS[freq_label]
    agg  = AGG_OPTIONS[agg_label]

    # ── Build time series ──────────────────────────────────────────────────────
    ts_df = df[[dt_col, target_col]].copy()
    ts_df = ts_df.sort_values(dt_col).set_index(dt_col)
    ts    = _resample(ts_df[target_col], freq, agg)

    st.write(
        f"Resampled to **{freq_label}** using **{agg_label}** "
        f"→ **{len(ts)} data points**"
    )

    if len(ts) < 5:
        st.warning(
            "Too few data points after resampling. "
            "Try a finer frequency (e.g. Daily instead of Monthly)."
        )
        return

    # ── Trend chart ────────────────────────────────────────────────────────────
    fig = px.line(
        ts.reset_index(),
        x=dt_col, y=target_col,
        title=f"{target_col} over time ({freq_label} {agg_label})",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Decomposition ──────────────────────────────────────────────────────────
    with st.expander("📊 Seasonal Decomposition", expanded=False):
        show_decomposition(ts, freq)

    # ── Stationarity ──────────────────────────────────────────────────────────
    with st.expander("📐 Stationarity Test (ADF)", expanded=False):
        st.caption(
            "ARIMA requires a stationary series. "
            "The ADF test checks whether your series has a unit root (non-stationary)."
        )
        if STATSMODELS_AVAILABLE:
            is_stat = show_stationarity(ts, target_col)
            recommended_d = 0 if is_stat else 1
        else:
            recommended_d = 1

    # ── ARIMA forecast ─────────────────────────────────────────────────────────
    st.subheader("ARIMA Forecast")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        p = st.number_input("p (AR order)",    min_value=0, max_value=5,
                            value=1, step=1, key="arima_p",
                            help="Number of autoregressive terms.")
    with col6:
        d = st.number_input("d (differencing)", min_value=0, max_value=2,
                            value=recommended_d if STATSMODELS_AVAILABLE else 1,
                            step=1, key="arima_d",
                            help="Degree of differencing. ADF test recommends this value above.")
    with col7:
        q = st.number_input("q (MA order)",    min_value=0, max_value=5,
                            value=1, step=1, key="arima_q",
                            help="Number of moving average terms.")
    with col8:
        steps = st.number_input("Forecast periods", min_value=1, max_value=36,
                                value=6, step=1, key="arima_steps")

    if st.button("📈 Run ARIMA Forecast", type="primary"):
        run_arima_forecast(
            series=ts,
            freq=freq,
            steps=int(steps),
            p=int(p), d=int(d), q=int(q),
            target_col=target_col,
            dt_col=dt_col,
        )