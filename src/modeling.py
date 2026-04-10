import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import shap
from langchain_google_genai import ChatGoogleGenerativeAI

def detect_task_type(series):
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() < 20: 
            return "Classification"
        return "Regression"
    return "Classification"

def plot_learning_curve(model, X, y):
    st.write("### 📈 Learning Curves")
    st.info("Learning graphs show if the model is underfitting (needs complexity) or overfitting (needs more data).")
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    st.pyplot(fig)

def explain_features(model, feature_names, target_col, task_type):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        
    if importances is None:
        return
        
    st.write("### 🔥 Feature Importance Analysis")
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)
    
    fig = px.bar(feat_df, x="Importance", y="Feature", orientation='h', title="Top 10 Most Important Features", color="Importance", color_continuous_scale="Viridis")
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.info("💡 Enter your Google API Key in the sidebar to get an AI-generated explanation of why these features are important.")
        return
        
    with st.spinner("Generating AI Explanation for these features..."):
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            top_features_list = feat_df["Feature"].tolist()
            prompt = f"You are an expert Data Scientist analyzing a dataset. I trained a {task_type} model to predict '{target_col}'. The top most important features determined by the model in order of importance are: {', '.join(top_features_list)}. In 2-3 short, clear paragraphs, explain to a business user *why* these features matter and *why* the model likely chose them to predict '{target_col}'. Do not mention any code."
            response = llm.invoke(prompt)
            st.markdown(f"**🤖 AI Insights - Why this matters:**\n\n{response.content}")
        except Exception as e:
            st.error(f"Error generating explanation: {e}")

def explain_shap(model, X_train, X_test):
    st.write("### 🧠 SHAP Explainable AI (Local Explainability)")
    st.info("SHAP explains exactly how much each feature contributed to a **single, specific prediction**.")
    
    if not hasattr(model, "feature_importances_"):
        st.warning("SHAP integration currently supports tree-based models (Random Forest, XGBoost) in this app format.")
        return
        
    with st.spinner("Computing SHAP values..."):
        try:
            explainer = shap.TreeExplainer(model)
            sample_idx = 0
            instance = X_test.iloc[[sample_idx]]
            shap_values = explainer.shap_values(instance)
            
            if isinstance(shap_values, list):
                sv = shap_values[1] # positive class for classification logic usually
            else:
                sv = shap_values
                
            fig, ax = plt.subplots(figsize=(8,3))
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
                base_val = base_val[1]
                
            explanation = shap.Explanation(values=sv[0], base_values=base_val, data=instance.iloc[0], feature_names=X_test.columns.tolist())
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(plt.gcf())
            plt.clf() # clean memory buffer
            st.write("*(This waterfall plot breaks down the exact baseline model offsets made by each feature for predicting Instance 0)*")
        except Exception as e:
            st.error(f"Could not render SHAP for this specific model configuration: {e}")

def run_model_comparison(task_type, X_train, X_test, y_train, y_test):
    st.write(f"### ⚡ Model Comparison Leaderboard")
    results = []
    models = {}
    
    if task_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        
        for name, clf in models.items():
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            results.append({"Model": name, "Score": acc, "Metric": "Accuracy"})
            
    else:  # Regression
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "XGBoost Regressor": XGBRegressor(random_state=42)
        }
        for name, reg in models.items():
            reg.fit(X_train, y_train)
            r2 = r2_score(y_test, reg.predict(X_test))
            results.append({"Model": name, "Score": r2, "Metric": "R² Score"})
            
    res_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    fig = px.bar(res_df, x="Score", y="Model", orientation='h', color="Model", title=f"Comparison of Models ({res_df['Metric'].iloc[0]})")
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    best_model_name = res_df.iloc[0]["Model"]
    st.success(f"🏆 Best Performing Model: **{best_model_name}** with {res_df['Metric'].iloc[0]} of {res_df.iloc[0]['Score']:.4f}")
    
    return models[best_model_name], best_model_name

def prepare_modeling(df):
    target_col = st.selectbox("Select Target Column", df.columns.tolist())
    
    if not target_col:
        return
        
    task_type = detect_task_type(df[target_col])
    st.write(f"**Detected Task Type:** {task_type}")
    
    if df[target_col].isnull().any():
        st.warning("Target column has missing values. Dropping those rows for modeling.")
        df = df.dropna(subset=[target_col])
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        st.error("Some features are strictly categorical. Please encode them in the Feature Engineering step before modeling.")
        return
        
    # Standardize Classification target logic for XGBoost safety
    if task_type == "Classification" and (y.dtype == 'object' or y.dtype.name == 'category'):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
        
    test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.write(f"**Training Set:** {X_train.shape[0]} rows | **Test Set:** {X_test.shape[0]} rows")
    
    if st.button("Train & Compare Models", type="primary"):
        with st.spinner("Training models for comparison..."):
            best_model, best_name = run_model_comparison(task_type, X_train, X_test, y_train, y_test)
            
            st.divider()
            explain_features(best_model, X.columns, target_col, task_type)
            st.divider()
            plot_learning_curve(best_model, X, y)
            st.divider()
            explain_shap(best_model, X_train, X_test)
