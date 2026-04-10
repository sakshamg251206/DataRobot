import streamlit as st
import pandas as pd
from src.data_loading import load_data
from src.eda import show_data_info, show_missing_values, show_summary_statistics
from src.cleaning import auto_clean_data
from src.feature_engineering import encode_categorical, scale_numerical, auto_feature_engineering
from src.visualization import plot_histogram, plot_correlation_heatmap, plot_pairplot
from src.modeling import prepare_modeling
from src.time_series import run_time_series_analysis
from src.report_generator import render_report_section
from src.ai_assistant import render_ai_assistant
import os

st.set_page_config(page_title="Auto Data Science Max", layout="wide", page_icon="📊")

def init_session():
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'encoded_data' not in st.session_state:
        st.session_state.encoded_data = None

def main():
    init_session()
    
    st.sidebar.title("📊 Auto Data Science Pro")
    st.sidebar.write("Upload a dataset and deploy an entire Data Science pipeline natively.")
    
    st.sidebar.divider()
    google_key = st.sidebar.text_input("Keys: Google Gemini API Key", type="password")
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
    st.sidebar.divider()
    
    app_mode = st.sidebar.radio("Dashboard Sections", [
        "1. Upload Dataset",
        "2. exploratory Data Analysis",
        "3. Advanced Cleaning",
        "4. Feature Engineering",
        "5. Visualization",
        "6. Time Series Analysis",
        "7. Machine Learning / Explainable AI",
        "8. AI Assistant Chat 🤖",
        "9. Report Generator"
    ])
    
    # 1. Upload
    if app_mode == "1. Upload Dataset":
        st.header("Upload your CSV dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.raw_data = df
                st.session_state.cleaned_data = None
                st.session_state.encoded_data = None
                
        if st.session_state.raw_data is not None:
            st.write("### Preview of Dataset")
            st.dataframe(st.session_state.raw_data.head(10), use_container_width=True)
            st.write("Dataset Shape:", st.session_state.raw_data.shape)

    # 2. EDA
    elif app_mode == "2. exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        if st.session_state.raw_data is not None:
            df = st.session_state.raw_data
            show_data_info(df)
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                show_missing_values(df)
            with col2:
                show_summary_statistics(df)
        else:
            st.warning("Please upload a dataset in Step 1.")

    # 3. Data Cleaning
    elif app_mode == "3. Advanced Cleaning":
        st.header("Automated Advanced Data Cleaning")
        col1, col2 = st.columns(2)
        with col1:
            num_strategy = st.selectbox("Numerical Imputation Method", ["Median", "Mean", "KNN Imputation", "Forward/Backward Fill (Time Series)"])
        with col2:
            outlier_method = st.selectbox("Outlier Detection Method", ["IQR", "Z-score"])
            outlier_action = st.selectbox("Outlier Handling", ["Cap", "Remove"])
            
        if st.session_state.raw_data is not None:
            if st.button("Run Auto-Cleaning Sequence", type="primary"):
                cleaned = auto_clean_data(st.session_state.raw_data, num_strategy=num_strategy, outlier_method=outlier_method, outlier_action=outlier_action)
                st.session_state.cleaned_data = cleaned
                
            if st.session_state.cleaned_data is not None:
                st.write("### Cleaned Dataset")
                st.dataframe(st.session_state.cleaned_data.head(), use_container_width=True)
                csv = st.session_state.cleaned_data.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Cleaned Dataset (.csv)", data=csv, file_name='cleaned_dataset.csv', mime='text/csv')
        else:
            st.warning("Please upload a dataset first.")

    # 4. Feature Engineering
    elif app_mode == "4. Feature Engineering":
        st.header("Feature Engineering Engine")
        data_to_use = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.raw_data
        
        if data_to_use is not None:
            col1, col2 = st.columns(2)
            with col1:
                encoding_strategy = st.selectbox("Categorical Encoding", ["Label Encoding", "One-Hot Encoding"])
                add_poly = st.checkbox("Add Polynomial Features (Degree 2)")
            with col2:
                scaling_strategy = st.selectbox("Numerical Scaling", ["None", "Standard Scaling", "Min-Max Normalization"])
                add_ratios = st.checkbox("Generate Top Interaction Ratios")
            
            if st.button("Apply Transformations", type="primary"):
                df_engineered = auto_feature_engineering(data_to_use, poly=add_poly, ratios=add_ratios)
                df_encoded = encode_categorical(df_engineered, strategy=encoding_strategy)
                df_scaled = scale_numerical(df_encoded, strategy=scaling_strategy)
                st.session_state.encoded_data = df_scaled
                
            if st.session_state.encoded_data is not None:
                st.write("### Trandformed feature space")
                st.dataframe(st.session_state.encoded_data.head(), use_container_width=True)
        else:
            st.warning("Please cleanly preprocess a dataset first.")

    # 5. Visualization
    elif app_mode == "5. Visualization":
        st.header("Interactive Visualizations (Plotly)")
        data_to_plot = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.raw_data
        
        if data_to_plot is not None:
            tab1, tab2, tab3 = st.tabs(["Histograms", "Correlation Matrix", "Pairplot Scatter"])
            with tab1:
                plot_histogram(data_to_plot)
            with tab2:
                plot_correlation_heatmap(data_to_plot)
            with tab3:
                plot_pairplot(data_to_plot)
        else:
            st.warning("Please upload a dataset first.")

    # 6. Time Series
    elif app_mode == "6. Time Series Analysis":
        st.header("Time Series Analysis")
        data_to_ts = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.raw_data
        if data_to_ts is not None:
            run_time_series_analysis(data_to_ts)
        else:
            st.warning("Please upload a dataset first.")

    # 7. Machine Learning / Explainable AI
    elif app_mode == "7. Machine Learning / Explainable AI":
        st.header("Machine Learning & SHAP Explanation")
        if st.session_state.encoded_data is not None:
            prepare_modeling(st.session_state.encoded_data)
        elif st.session_state.cleaned_data is not None:
            st.warning("Data not strictly fully encoded, standard preprocessing is highly suggested but will proceed mapping logic:")
            prepare_modeling(st.session_state.cleaned_data)
        else:
            st.warning("Please upload a dataset first.")

    # 8. AI Assistant
    elif app_mode == "8. AI Assistant Chat 🤖":
        data_to_chat = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.raw_data
        render_ai_assistant(data_to_chat)
        
    # 9. PDF Report
    elif app_mode == "9. Report Generator":
        render_report_section()

if __name__ == "__main__":
    main()
