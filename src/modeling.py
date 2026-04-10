import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def detect_task_type(series):
    if pd.api.types.is_numeric_dtype(series):
        # Even if numerical, if very few unique values, it might be classification
        if series.nunique() < 20: 
            return "Classification"
        return "Regression"
    return "Classification"

def prepare_modeling(df):
    target_col = st.selectbox("Select Target Column", df.columns.tolist())
    
    if not target_col:
        return
        
    task_type = detect_task_type(df[target_col])
    st.write(f"**Detected Task Type:** {task_type}")
    
    # Drop rows where target is missing
    if df[target_col].isnull().any():
        st.warning("Target column has missing values. Dropping those rows for modeling.")
        df = df.dropna(subset=[target_col])
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Check if all features in X are numerical, since we need encoded data
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        st.error("Some features are strictly categorical and haven't been encoded. Please encode them in the Feature Engineering step before modeling.")
        return
        
    test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.write(f"**Training Set:** {X_train.shape[0]} rows | **Test Set:** {X_test.shape[0]} rows")
    
    if task_type == "Classification":
        model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest Classifier"])
        if st.button("Train Model"):
            with st.spinner("Training..."):
                if model_name == "Logistic Regression":
                    # increase max_iter for convergence
                    model = LogisticRegression(max_iter=1000)
                else:
                    model = RandomForestClassifier(random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                st.success(f"**Accuracy:** {acc:.4f}")
                
                st.write("**Confusion Matrix:**")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
                st.pyplot(fig)
                
    else:  # Regression
        model_name = st.selectbox("Select Model", ["Linear Regression", "Random Forest Regressor"])
        if st.button("Train Model"):
            with st.spinner("Training..."):
                if model_name == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)
                
                st.success(f"**RMSE:** {rmse:.4f}")
                st.success(f"**R² Score:** {r2:.4f}")
                
                # Plot actual vs predicted
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y_test, y_pred, alpha=0.7)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)
