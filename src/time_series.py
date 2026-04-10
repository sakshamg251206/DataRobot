import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import warnings

def has_datetime_column(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True, col
    return False, None

def run_time_series_analysis(df):
    st.write("### 📅 Time Series & Forecasting")
    
    is_ts, dt_col = has_datetime_column(df)
    if not is_ts:
        st.warning("No datetime column detected. Please ensure your dataset has a time column and is auto-cleaned in Step 3 to convert to Datetime.")
        return
        
    st.success(f"Time Series detected on column: **{dt_col}**")
    
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not num_cols:
        st.error("No numerical columns available for forecasting over time.")
        return
        
    target_col = st.selectbox("Select metric to forecast over time:", num_cols)
    
    ts_df = df.copy()
    ts_df = ts_df.sort_values(dt_col).set_index(dt_col)
    
    freq = st.selectbox("Resample Frequency", ["D", "W", "M", "Q", "Y"], index=2, help="D=Day, W=Week, M=Month, Q=Quarter, Y=Year")
    ts_agg = ts_df[target_col].resample(freq).mean().dropna()
    
    fig = px.line(ts_agg.reset_index(), x=dt_col, y=target_col, title=f"Trend over time ({freq})")
    st.plotly_chart(fig, use_container_width=True)
    
    if len(ts_agg) < 10:
        st.warning("Not enough time periods for a statistically significant forecast (ARIMA requires more data points).")
        return
        
    if st.button("Forecast Next 5 Periods (ARIMA)"):
        with st.spinner("Calculating ARIMA Forecast..."):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(ts_agg, order=(1, 1, 1))
                fit_model = model.fit()
                forecast = fit_model.forecast(steps=5)
            
            past_df = pd.DataFrame({dt_col: ts_agg.index, "Value": ts_agg.values, "Type": "Historical"})
            future_df = pd.DataFrame({dt_col: forecast.index, "Value": forecast.values, "Type": "Forecast"})
            combined_df = pd.concat([past_df, future_df])
            
            fig2 = px.line(combined_df, x=dt_col, y="Value", color="Type", title=f"ARIMA Forecast for {target_col}")
            fig2.update_traces(line=dict(dash="dot"), selector=dict(name="Forecast"))
            st.plotly_chart(fig2, use_container_width=True)
