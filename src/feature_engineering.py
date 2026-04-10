import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures

def get_column_types(df):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
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

def auto_feature_engineering(df, poly=False, ratios=False):
    num_cols, _ = get_column_types(df)
    if len(num_cols) < 2:
        return df
    
    df_eng = df.copy()
    
    if ratios:
        if len(num_cols) >= 2:
            col1, col2 = num_cols[0], num_cols[1]
            # Epsilon addition to prevent Inf issues
            df_eng[f"{col1}_div_{col2}"] = df_eng[col1] / (df_eng[col2] + 1e-9)
            st.success(f"Created engineered interaction ratio: {col1} / {col2}")
            
    if poly:
        st.success("Generating Polynomial Features (Degree 2) on top numerical columns.")
        top_cols = num_cols[:3] # keep small
        poly_feat = PolynomialFeatures(degree=2, include_bias=False)
        poly_df = pd.DataFrame(poly_feat.fit_transform(df_eng[top_cols]), 
                               columns=poly_feat.get_feature_names_out(top_cols),
                               index=df_eng.index)
        df_eng = df_eng.drop(columns=top_cols)
        df_eng = pd.concat([df_eng, poly_df], axis=1)
        
    return df_eng
