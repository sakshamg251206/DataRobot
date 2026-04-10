import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.encoding import get_column_types

def plot_histogram(df):
    st.write("### Histograms for Numerical Columns")
    num_cols, _ = get_column_types(df)
    
    if not num_cols:
        st.warning("No numerical columns found to plot histograms.")
        return
        
    col_to_plot = st.selectbox("Select column for Histogram", num_cols)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col_to_plot], kde=True, ax=ax)
    ax.set_title(f"Histogram of {col_to_plot}")
    st.pyplot(fig)

def plot_correlation_heatmap(df):
    st.write("### Correlation Heatmap")
    num_cols, _ = get_column_types(df)
    
    if len(num_cols) < 2:
        st.warning("Heatmap requires at least 2 numerical columns.")
        return
        
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

def plot_pairplot(df):
    st.write("### Pairplot")
    num_cols, _ = get_column_types(df)
    
    if len(num_cols) < 2:
        st.warning("Pairplot requires at least 2 numerical columns.")
        return
        
    # Limit to top 5 numerical columns to speed up processing
    cols_to_plot = num_cols[:5]
    if len(num_cols) > 5:
        st.info("Pairplot limited to top 5 numerical columns for better performance.")
        
    fig = sns.pairplot(df[cols_to_plot])
    st.pyplot(fig)
