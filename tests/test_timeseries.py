import pandas as pd
import pytest

from autods.core.timeseries import (
    TimeSeriesError,
    adf_test,
    arima_forecast,
    build_series,
    date_column_candidates,
    decompose,
)


def test_build_series_parses_text_dates(monthly_series_df):
    info = build_series(monthly_series_df, "date", "sales", "ME", "mean")
    assert len(info.series) == 60
    assert info.filled_gaps == 0
    assert "date" in date_column_candidates(monthly_series_df)


def test_sum_aggregation_keeps_same_day_rows():
    df = pd.DataFrame(
        {"day": ["2024-01-01", "2024-01-01", "2024-01-03"], "amount": [10.0, 5.0, 7.0]}
    )
    series = build_series(df, "day", "amount", "D", "sum").series
    assert series.tolist() == [15.0, 0.0, 7.0]


def test_mean_aggregation_interpolates_gaps():
    df = pd.DataFrame({"day": ["2024-01-01", "2024-01-03"], "v": [1.0, 3.0]})
    info = build_series(df, "day", "v", "D", "mean")
    assert info.series.tolist() == [1.0, 2.0, 3.0]
    assert info.filled_gaps == 1


def test_decompose_and_adf(monthly_series_df):
    series = build_series(monthly_series_df, "date", "sales", "ME", "mean").series
    parts = decompose(series, "ME")
    assert list(parts.columns) == ["Observed", "Trend", "Seasonal", "Residual"]
    assert adf_test(series).recommended_d in (0, 1)
    with pytest.raises(TimeSeriesError, match="sub-yearly"):
        decompose(series, "YE")


def test_arima_forecast_with_backtest(monthly_series_df):
    series = build_series(monthly_series_df, "date", "sales", "ME", "mean").series
    result = arima_forecast(series, (1, 1, 1), steps=6)
    assert len(result.forecast) == 6
    assert (result.forecast["lower"] <= result.forecast["upper"]).all()
    assert result.forecast.index[0] > series.index[-1]
    assert result.backtest_mae is not None and result.naive_mae is not None


def test_arima_needs_enough_points():
    series = pd.Series(range(10), index=pd.date_range("2024", periods=10, freq="D"), dtype=float)
    with pytest.raises(TimeSeriesError, match="at least"):
        arima_forecast(series, (1, 0, 0), 3)
