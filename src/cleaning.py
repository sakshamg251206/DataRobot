import streamlit as st
import pandas as pd
import numpy as np

def remove_duplicates(df):
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    final_shape = df.shape[0]
    st.success(f"Removed {initial_shape - final_shape} duplicate rows.")
    return df

def fill_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
                st.info(f"Filled missing values in numerical column **{col}** with its median.")
            else:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
                    st.info(f"Filled missing values in categorical column **{col}** with its mode.")
    return df

def auto_clean_data(df):
    st.write("### Automating Data Cleaning...")
    df_cleaned = df.copy()
    if df_cleaned.duplicated().sum() > 0:
        df_cleaned = remove_duplicates(df_cleaned)
    else:
        st.info("No duplicates found.")
        
    if df_cleaned.isnull().sum().sum() > 0:
        df_cleaned = fill_missing_values(df_cleaned)
    else:
        st.info("No missing values found to fill.")
        
    st.success("Data Cleaning Complete!")
    return df_cleaned
