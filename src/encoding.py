import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

def get_column_types(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return num_cols, cat_cols

def encode_categorical(df, strategy="Label Encoding"):
    _, cat_cols = get_column_types(df)
    
    if not cat_cols:
        st.info("No categorical columns to encode.")
        return df
        
    df_encoded = df.copy()
    
    if strategy == "Label Encoding":
        le = LabelEncoder()
        for col in cat_cols:
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = le.fit_transform(df_encoded[col])
        st.success("Applied Label Encoding to categorical columns.")
        
    elif strategy == "One-Hot Encoding":
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)
        st.success("Applied One-Hot Encoding format to categorical columns.")
        
    return df_encoded

def scale_numerical(df, strategy="Standard Scaling"):
    num_cols, _ = get_column_types(df)
    
    if not num_cols:
        st.info("No numerical columns to scale.")
        return df
        
    df_scaled = df.copy()
    
    if strategy == "None":
        st.info("Numerical features kept as original.")
        return df_scaled
        
    if strategy == "Standard Scaling":
        scaler = StandardScaler()
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
        st.success("Standard Scaling applied to numerical columns.")
        
    elif strategy == "Min-Max Normalization":
        scaler = MinMaxScaler()
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
        st.success("Min-Max Normalization applied to numerical columns.")
        
    return df_scaled
