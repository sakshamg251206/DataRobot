import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_text_insight(prompt_context):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "💡 Provide a Google Gemini API Key in the sidebar to unlock automated AI textual insights."
        
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        prompt = f"You are a Senior Data Analyst. Analyze the following local statistical context and return exactly 1-2 sharp, highly-insightful sentences summarizing what it means.\n\nContext:\n{prompt_context}"
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Warning: AI Insight Generation failed ({e})"

def render_univariate(df, col):
    if pd.api.types.is_numeric_dtype(df[col]):
        st.write(f"#### Distribution of {col} (Numerical)")
        fig = px.histogram(df, x=col, marginal="box", title=f"Histogram with Boxplot: {col}")
        st.plotly_chart(fig, use_container_width=True)
        
        skew = df[col].skew()
        insight_context = f"Numerical variable '{col}' has a skewness of {skew:.2f}. Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}."
        
    else:
        st.write(f"#### Frequency of {col} (Categorical)")
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, 'Count']
        fig = px.bar(counts, x=col, y='Count', title=f"Count Plot: {col}")
        st.plotly_chart(fig, use_container_width=True)
        
        top_cat = df[col].mode()[0]
        insight_context = f"Categorical variable '{col}' has {df[col].nunique()} unique categories. The most frequent category is '{top_cat}' with {df[col].value_counts().max()} occurrences."
        
    with st.spinner("Generating AI Insight..."):
        ai_insight = generate_text_insight(insight_context)
    st.info(f"**🤖 AI Insight:** {ai_insight}")

def render_bivariate(df, col1, col2):
    type1 = "Num" if pd.api.types.is_numeric_dtype(df[col1]) else "Cat"
    type2 = "Num" if pd.api.types.is_numeric_dtype(df[col2]) else "Cat"
    
    st.write(f"#### Bivariate Analysis: `{col1}` vs `{col2}`")
    
    if type1 == "Num" and type2 == "Num":
        # Check sample size to prevent scatter map blow out natively
        if len(df) > 10000:
            df = df.sample(10000, random_state=42)
            
        fig = px.scatter(df, x=col1, y=col2, trendline="ols", title=f"Scatter Plot (Num vs Num)")
        st.plotly_chart(fig, use_container_width=True)
        
        corr = df[col1].corr(df[col2])
        insight_context = f"Comparing two numerical features '{col1}' and '{col2}'. Their Pearson correlation coefficient is {corr:.3f}."
        
    elif type1 == "Cat" and type2 == "Cat":
        heatmap_data = pd.crosstab(df[col1], df[col2])
        fig = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale="Blues", title=f"Crosstab Heatmap (Cat vs Cat)")
        st.plotly_chart(fig, use_container_width=True)
        
        insight_context = f"Comparing two categorical features '{col1}' and '{col2}'. The crosstab isolates standard occurrences."
        
    else:
        num_col = col1 if type1 == "Num" else col2
        cat_col = col1 if type1 == "Cat" else col2
        fig = px.box(df, x=cat_col, y=num_col, color=cat_col, title=f"Box Plot Distribution (Cat vs Num)")
        st.plotly_chart(fig, use_container_width=True)
        
        avg_per_cat = df.groupby(cat_col)[num_col].mean().to_dict()
        insight_context = f"Analyzing numerical '{num_col}' across categorical '{cat_col}'. The raw averages per category represent significant thresholds mappings."
        
    with st.spinner("Generating AI Insight..."):
        ai_insight = generate_text_insight(insight_context)
    st.info(f"**🤖 AI Insight:** {ai_insight}")

def render_advanced_analysis(df):
    st.header("📊 Advanced Bivariate AI Analysis")
    st.write("Deep dive into specific feature distributions and mapping structures natively augmented with Gemini Flash Contexts.")
    
    analysis_type = st.radio("Select Analysis Type", ["Univariate (1 Column)", "Bivariate (2 Columns)"], horizontal=True)
    st.divider()
    
    if analysis_type == "Univariate (1 Column)":
        col1 = st.selectbox("Select Feature Matrix:", df.columns)
        if st.button("Run Isolation Analysis", type="primary"):
            render_univariate(df, col1)
            
    else:
        col1, col2 = st.columns(2)
        with col1:
            c1 = st.selectbox("Select Feature 1:", df.columns)
        with col2:
            c2 = st.selectbox("Select Feature 2:", [c for c in df.columns if c != c1])
            
        if st.button("Run Joint Impact Analysis", type="primary"):
            render_bivariate(df, c1, c2)
