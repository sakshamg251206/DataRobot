import streamlit as st
import pandas as pd

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            return df
        else:
            st.error("Please upload a valid CSV file.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None
