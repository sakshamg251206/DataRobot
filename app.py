import streamlit as st
import pandas as pd
from src.data_loading import load_data
from src.eda import show_data_info, show_missing_values, show_summary_statistics
from src.cleaning import auto_clean_data
from src.encoding import encode_categorical, scale_numerical
from src.visualization import plot_histogram, plot_correlation_heatmap, plot_pairplot
from src.modeling import prepare_modeling

# Page Config
st.set_page_config(page_title="Auto Data Science", layout="wide", page_icon="📊")

def init_session():
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'encoded_data' not in st.session_state:
        st.session_state.encoded_data = None

def main():
    init_session()
    
    st.sidebar.title("📊 Auto Data Science App")
    st.sidebar.write("Upload a dataset and go through the pipeline automatically!")
    
    app_mode = st.sidebar.radio("Pipeline Steps", [
        "1. Upload Dataset",
        "2. exploratory Data Analysis (EDA)",
        "3. Data Cleaning",
        "4. Feature Engineering",
        "5. Visualization",
        "6. Machine Learning"
    ])
    
    # 1. Upload Dataset
    if app_mode == "1. Upload Dataset":
        st.header("Upload your CSV dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.raw_data = df
                # Reset downstream states if new data is uploaded
                st.session_state.cleaned_data = None
                st.session_state.encoded_data = None
                
        if st.session_state.raw_data is not None:
            st.write("### Preview of Dataset")
            st.dataframe(st.session_state.raw_data.head(10), use_container_width=True)
            st.write("Dataset Shape:", st.session_state.raw_data.shape)

    # 2. EDA
    elif app_mode == "2. exploratory Data Analysis (EDA)":
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
    elif app_mode == "3. Data Cleaning":
        st.header("Automated Data Cleaning")
        if st.session_state.raw_data is not None:
            if st.button("Run Auto-Cleaning", type="primary"):
                cleaned = auto_clean_data(st.session_state.raw_data)
                st.session_state.cleaned_data = cleaned
                
            if st.session_state.cleaned_data is not None:
                st.write("### Cleaned Dataset")
                st.dataframe(st.session_state.cleaned_data.head(), use_container_width=True)
                
                # Allow download of cleaned data
                csv = st.session_state.cleaned_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Dataset (.csv)",
                    data=csv,
                    file_name='cleaned_dataset.csv',
                    mime='text/csv',
                )
        else:
            st.warning("Please upload a dataset in Step 1.")

    # 4. Feature Engineering
    elif app_mode == "4. Feature Engineering":
        st.header("Feature Engineering (Encoding & Scaling)")
        
        data_to_use = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.raw_data
        
        if data_to_use is not None:
            col1, col2 = st.columns(2)
            with col1:
                encoding_strategy = st.selectbox("Categorical Encoding Strategy", ["Label Encoding", "One-Hot Encoding"])
            with col2:
                scaling_strategy = st.selectbox("Numerical Scaling Strategy", ["None", "Standard Scaling", "Min-Max Normalization"])
            
            if st.button("Apply Transformations", type="primary"):
                df_encoded = encode_categorical(data_to_use, strategy=encoding_strategy)
                df_scaled = scale_numerical(df_encoded, strategy=scaling_strategy)
                st.session_state.encoded_data = df_scaled
                
            if st.session_state.encoded_data is not None:
                st.write("### Transformed Dataset")
                st.dataframe(st.session_state.encoded_data.head(), use_container_width=True)
        else:
            st.warning("Please upload a dataset first. It is also highly recommended to clean it in Step 3.")

    # 5. Visualization
    elif app_mode == "5. Visualization":
        st.header("Data Visualizations")
        data_to_plot = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.raw_data
        
        if data_to_plot is not None:
            tab1, tab2, tab3 = st.tabs(["Histograms", "Correlation Heatmap", "Pairplot"])
            with tab1:
                plot_histogram(data_to_plot)
            with tab2:
                plot_correlation_heatmap(data_to_plot)
            with tab3:
                plot_pairplot(data_to_plot)
        else:
            st.warning("Please upload a dataset first.")

    # 6. Machine Learning
    elif app_mode == "6. Machine Learning":
        st.header("Machine Learning Modeling")
        
        # ML is best done on encoded data
        if st.session_state.encoded_data is not None:
            prepare_modeling(st.session_state.encoded_data)
        elif st.session_state.cleaned_data is not None:
            st.warning("Warning: Using partially processed data. It is highly recommended to encode categorical variables in Step 4 before modeling.")
            prepare_modeling(st.session_state.cleaned_data)
        elif st.session_state.raw_data is not None:
            st.error("Please clean and encode features before attempting machine learning.")
        else:
            st.warning("Please upload a dataset first.")

if __name__ == "__main__":
    main()
