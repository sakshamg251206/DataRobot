"""Time series: resampling, decomposition, stationarity testing and ARIMA forecasting."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from autods.core.dtypes import looks_like_datetime, numeric_columns, to_datetime

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller

    STATSMODELS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    STATSMODELS_AVAILABLE = False

FREQUENCIES = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
AGGREGATIONS = {"Mean": "mean", "Sum": "sum", "Last": "last", "Max": "max", "Min": "min"}
SEASONAL_PERIOD = {"D": 7, "W": 52, "ME": 12, "QE": 4}
MIN_POINTS = 8
MIN_POINTS_ARIMA = 20
ADF_ALPHA = 0.05


class TimeSeriesError(ValueError):
    """Raised with a user-facing message when an analysis cannot run."""


def date_column_candidates(df: pd.DataFrame) -> list[str]:
    """Datetime columns first, then text columns that parse as dates."""
    return [c for c in df.columns if looks_like_datetime(df[c])]


@dataclass
class SeriesInfo:
    series: pd.Series
    filled_gaps: int
    raw_points: int


def build_series(
    df: pd.DataFrame, date_col: str, value_col: str, freq: str, agg: str
) -> SeriesInfo:
    """Aggregate ``value_col`` per ``freq`` period into a regular time series.

    Several rows on the same timestamp are combined by ``agg`` (e.g. all sales
    on one day are summed). Empty periods are interpolated for mean-like
    aggregations and treated as zero for sums.
    """
    if value_col not in numeric_columns(df):
        raise TimeSeriesError(f"`{value_col}` is not numeric.")
    dates = to_datetime(df[date_col])
    frame = pd.DataFrame({"date": dates, "value": df[value_col]}).dropna()
    if frame.empty:
        raise TimeSeriesError(f"No rows have both a valid date and a `{value_col}` value.")

    resampled = frame.set_index("date")["value"].sort_index().resample(freq)
    series: pd.Series = pd.Series(
        resampled.sum(min_count=0) if agg == "sum" else resampled.agg(agg)
    )
    gaps = int(series.isna().sum())
    if gaps:
        series = series.interpolate(method="time").ffill().bfill()
    series.name = value_col
    series.index.name = date_col
    return SeriesInfo(series=series.astype(float), filled_gaps=gaps, raw_points=len(frame))


@dataclass
class StationarityResult:
    statistic: float
    p_value: float
    is_stationary: bool

    @property
    def recommended_d(self) -> int:
        return 0 if self.is_stationary else 1


def adf_test(series: pd.Series) -> StationarityResult:
    """Augmented Dickey–Fuller test: a small p-value means the series is stationary."""
    values = series.dropna()
    if len(values) < MIN_POINTS or values.nunique() < 2:
        raise TimeSeriesError("Need at least 8 varying points for a stationarity test.")
    statistic, p_value, *_ = adfuller(values, autolag="AIC")
    return StationarityResult(float(statistic), float(p_value), bool(p_value < ADF_ALPHA))


def decompose(series: pd.Series, freq: str) -> pd.DataFrame:
    """Additive trend / seasonal / residual decomposition."""
    period = SEASONAL_PERIOD.get(freq)
    if period is None:
        raise TimeSeriesError("Seasonal decomposition needs a sub-yearly frequency.")
    if len(series) < 2 * period:
        raise TimeSeriesError(
            f"Need at least {2 * period} points (two full cycles) for decomposition; "
            f"have {len(series)}. Try a finer frequency."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = seasonal_decompose(series, model="additive", period=period)
    return pd.DataFrame(
        {
            "Observed": result.observed,
            "Trend": result.trend,
            "Seasonal": result.seasonal,
            "Residual": result.resid,
        }
    )


@dataclass
class Forecast:
    forecast: pd.DataFrame  # columns: forecast, lower, upper
    summary: str
    backtest_mae: float | None
    naive_mae: float | None
    backtest_points: int


def _fit_arima(series: pd.Series, order: tuple[int, int, int]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(series, order=order).fit()


def arima_forecast(series: pd.Series, order: tuple[int, int, int], steps: int) -> Forecast:
    """Fit ARIMA(p, d, q) and forecast ``steps`` periods with a 95% interval.

    Also back-tests the same order on the last ``steps`` points (capped at 20%
    of the series) and reports its MAE next to a naive "repeat the last value"
    forecast, so users can judge whether the model adds anything.
    """
    if not STATSMODELS_AVAILABLE:  # pragma: no cover
        raise TimeSeriesError("Install `statsmodels` to use forecasting.")
    if len(series) < MIN_POINTS_ARIMA:
        raise TimeSeriesError(
            f"ARIMA needs at least {MIN_POINTS_ARIMA} points; have {len(series)}. "
            "Use a finer frequency or more data."
        )

    try:
        fitted = _fit_arima(series, order)
        prediction = fitted.get_forecast(steps=steps)
    except Exception as exc:
        raise TimeSeriesError(f"ARIMA{order} could not be fitted: {exc}") from exc

    interval = prediction.conf_int(alpha=0.05)
    forecast = pd.DataFrame(
        {
            "forecast": np.asarray(prediction.predicted_mean),
            "lower": interval.iloc[:, 0].to_numpy(),
            "upper": interval.iloc[:, 1].to_numpy(),
        },
        index=prediction.predicted_mean.index,
    )

    holdout = min(steps, len(series) // 5)
    backtest_mae = naive_mae = None
    if holdout >= 1:
        train, test = series.iloc[:-holdout], series.iloc[-holdout:]
        try:
            predicted = _fit_arima(train, order).forecast(steps=holdout)
            backtest_mae = float(np.mean(np.abs(test.to_numpy() - np.asarray(predicted))))
            naive_mae = float(np.mean(np.abs(test.to_numpy() - train.iloc[-1])))
        except Exception:
            backtest_mae = naive_mae = None

    return Forecast(
        forecast=forecast,
        summary=str(fitted.summary()),
        backtest_mae=backtest_mae,
        naive_mae=naive_mae,
        backtest_points=holdout,
    )
