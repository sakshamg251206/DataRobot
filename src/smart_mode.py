import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.cleaning import detect_datetime_columns, handle_outliers
from src.feature_engineering import auto_feature_engineering
from src.modeling import detect_task_type

def compute_readiness_score(df):
    score = 100
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    score -= missing_ratio * 100
    
    dup_ratio = df.duplicated().sum() / float(df.shape[0])
    score -= dup_ratio * 50
    return max(0, min(100, score))

def smart_auto_pipeline(raw_df, target_col):
    df = raw_df.copy()
    logs = []
    
    init_score = compute_readiness_score(df)
    
    # 1. Date Detection
    df = detect_datetime_columns(df)
    
    # 2. Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates()
        logs.append(f"✅ Removed {dupes} duplicate rows.")
        
    # 3. Drop Constants
    cols_dropped = []
    for col in df.columns:
        if col == target_col: continue
        uniques = df[col].nunique()
        if uniques == 1:
            cols_dropped.append(col)
        elif df[col].dtype == 'object' and uniques == len(df):
            cols_dropped.append(col)
            
    if cols_dropped:
        df = df.drop(columns=cols_dropped)
        logs.append(f"✅ Dropped completely constant/ID columns: {', '.join(cols_dropped)}")
        
    # 4. Handle Missing Values
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns
    
    for col in num_cols:
        if df[col].isnull().any() and col != target_col:
            skew = df[col].skew()
            if abs(skew) > 1:
                df[col] = df[col].fillna(df[col].median())
                logs.append(f"✅ Filled missing '{col}' with Median (skewed).")
            else:
                df[col] = df[col].fillna(df[col].mean())
                logs.append(f"✅ Filled missing '{col}' with Mean (normal distribution).")
                
    for col in cat_cols:
        if df[col].isnull().any() and col != target_col:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
                logs.append(f"✅ Filled missing '{col}' with Mode.")

    # Target Drop Guarantee
    if df[target_col].isnull().any():
        drops = df[target_col].isnull().sum()
        df = df.dropna(subset=[target_col])
        logs.append(f"✅ Dropped {drops} trailing rows where Target variable is missing natively.")
        
    # 5. Outliers
    if len(num_cols) > 0:
        df = handle_outliers(df, method="IQR", action="Cap")
        logs.append(f"✅ Cap'd extreme outliers dynamically via IQR metrics bounds.")

    # 6. Encoding
    current_cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns
    for col in current_cat_cols:
        if col == target_col: continue
        if df[col].nunique() < 10:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            logs.append(f"✅ Applied One-Hot Encoding on '{col}' (Low Cardinality).")
        else:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            logs.append(f"✅ Applied Label Encoding on '{col}' (High Cardinality).")
            
    # 7. Auto Feature Engineering limit check
    if len(df) < 5000:
        df = auto_feature_engineering(df, poly=True, ratios=True)
        logs.append(f"✅ Deployed autonomous polynomial/interaction combinations.")
        
    # 8. Highly Collinear
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 2:
        corr_matrix = df[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90) and column != target_col]
        if to_drop:
            df = df.drop(columns=to_drop)
            logs.append(f"✅ Dropped excessive collinear parameters natively: {', '.join(to_drop)}")
            
    # 9. RF Feature Selection Check
    final_num_cols = df.select_dtypes(include=['number']).columns
    temp_target = df[target_col]
    task_type = detect_task_type(df[target_col])
    
    if task_type == "Classification" and (temp_target.dtype == 'object' or temp_target.dtype.name == 'category'):
        le_t = LabelEncoder()
        temp_target = pd.Series(le_t.fit_transform(temp_target), index=df.index)
        
    features = [c for c in final_num_cols if c != target_col]
    if len(features) > 2:
        X = df[features]
        # Double check no NaNs snuck through into the checking block
        if not X.isnull().any().any():
            try:
                if task_type == "Classification":
                    rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
                else:
                    rf = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
                rf.fit(X, temp_target)
                importances = rf.feature_importances_
                useless_features = [features[i] for i, imp in enumerate(importances) if imp == 0.0]
                if useless_features:
                    df = df.drop(columns=useless_features)
                    logs.append(f"✅ Pruned zero-weight significance features statically: {', '.join(useless_features)}")
            except Exception as e:
                pass # Silent fail RF formatting guard

    final_score = compute_readiness_score(df)
    return df, logs, init_score, final_score
