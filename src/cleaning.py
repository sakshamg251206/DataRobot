import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def remove_duplicates(df):
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    final_shape = df.shape[0]
    if initial_shape > final_shape:
        st.success(f"Removed {initial_shape - final_shape} duplicate rows.")
    return df

def detect_datetime_columns(df):
    df_dt = df.copy()
    dt_converted = 0
    for col in df_dt.columns:
        if df_dt[col].dtype == 'object':
            try:
                # Use errors='raise' so it doesn't randomly cast text data
                # Test using a subset to avoid slow scanning
                pd.to_datetime(df_dt[col].dropna().head(10))
                # If passes without throwing exception
                df_dt[col] = pd.to_datetime(df_dt[col], errors='coerce')
                dt_converted += 1
            except (ValueError, TypeError):
                pass
    if dt_converted > 0:
        st.success(f"Smart Detection: Converted {dt_converted} columns to Datetime format.")
    return df_dt

def fill_missing_values(df, num_strategy="Median", cat_strategy="Mode"):
    df_filled = df.copy()
    
    num_cols = df_filled.select_dtypes(include=['number']).columns
    if len(num_cols) > 0 and df_filled[num_cols].isnull().any().any():
        if num_strategy == "KNN Imputation":
            st.info("Applying KNN Imputation to numerical missing values.")
            imputer = KNNImputer(n_neighbors=5)
            # Imputer returns numpy array, assign it back
            df_filled[num_cols] = imputer.fit_transform(df_filled[num_cols])
        elif num_strategy == "Forward/Backward Fill (Time Series)":
            st.info("Applying Ffill/Bfill to numerical missing values.")
            df_filled[num_cols] = df_filled[num_cols].ffill().bfill()
        elif num_strategy == "Mean":
            st.info("Applying Mean imputation to numerical missing values.")
            df_filled[num_cols] = df_filled[num_cols].fillna(df_filled[num_cols].mean())
        else: # Median
            st.info("Applying Median imputation to numerical missing values.")
            df_filled[num_cols] = df_filled[num_cols].fillna(df_filled[num_cols].median())

    cat_cols = df_filled.select_dtypes(exclude=['number', 'datetime']).columns
    if len(cat_cols) > 0 and df_filled[cat_cols].isnull().any().any():
        if cat_strategy == "Forward/Backward Fill (Time Series)":
            df_filled[cat_cols] = df_filled[cat_cols].ffill().bfill()
        else:
            for col in cat_cols:
                if not df_filled[col].mode().empty:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
                    
    return df_filled

def handle_outliers(df, method="IQR", action="Cap"):
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include=['number']).columns
    
    outliers_handled = False
    for col in num_cols:
        if method == "IQR":
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # check if any outliers exist
            if ((df_out[col] < lower_bound) | (df_out[col] > upper_bound)).any():
                outliers_handled = True
                if action == "Cap":
                    df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
                elif action == "Remove":
                    df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
                    
        elif method == "Z-score":
            mean, std = df_out[col].mean(), df_out[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            if ((df_out[col] < lower_bound) | (df_out[col] > upper_bound)).any():
                outliers_handled = True
                if action == "Cap":
                    df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
                elif action == "Remove":
                    df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
    
    if outliers_handled:
        st.success(f"Outliers processed using {method} ({action} applied).")
        
    return df_out

def auto_clean_data(df, num_strategy="Median", outlier_method="IQR", outlier_action="Cap"):
    st.write("### Automating Data Cleaning...")
    df_cleaned = df.copy()
    
    # 1. Datetime detection
    df_cleaned = detect_datetime_columns(df_cleaned)
    
    # 2. Removing Duplicates
    if df_cleaned.duplicated().sum() > 0:
        df_cleaned = remove_duplicates(df_cleaned)
    else:
        st.info("No duplicates found.")
        
    # 3. Missing Values
    if df_cleaned.isnull().sum().sum() > 0:
        df_cleaned = fill_missing_values(df_cleaned, num_strategy=num_strategy)
    else:
        st.info("No missing values found to fill.")
        
    # 4. Outliers
    df_cleaned = handle_outliers(df_cleaned, method=outlier_method, action=outlier_action)
        
    st.success("Data Cleaning Complete!")
    return df_cleaned
