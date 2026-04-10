import streamlit as st
import pandas as pd

def show_data_info(df):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Shape:**")
        st.write(df.shape)
    with col2:
        st.write("**Duplicate Rows:**")
        st.write(df.duplicated().sum())
    
    st.write("**Data Types:**")
    dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
    dtype_df = dtype_df.rename(columns={'index': 'Column'})
    dtype_df['Data Type'] = dtype_df['Data Type'].astype(str)
    st.dataframe(dtype_df, use_container_width=True)

def show_missing_values(df):
    st.write("**Missing Values Per Column:**")
    missing_data = df.isnull().sum()
    missing_df = missing_data[missing_data > 0].reset_index()
    if not missing_df.empty:
        missing_df.columns = ['Column', 'Missing Values']
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("No missing values found in the dataset!")

def show_summary_statistics(df):
    st.write("**Summary Statistics:**")
    st.dataframe(df.describe(include='all'), use_container_width=True)
