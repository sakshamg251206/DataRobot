import streamlit as st
import plotly.express as px
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.feature_engineering import get_column_types

def plot_histogram(df):
    st.write("### 📊 Interactive Histograms")
    num_cols, _ = get_column_types(df)
    
    if not num_cols:
        st.warning("No numerical columns found to plot histograms.")
        return
        
    col_to_plot = st.selectbox("Select column for Histogram", num_cols)
    
    fig = px.histogram(df, x=col_to_plot, marginal="box", title=f"Histogram of {col_to_plot}")
    st.plotly_chart(fig, use_container_width=True)

def auto_insight_generator(df, context):
    st.write("#### 🤖 Auto Insight Generator")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.info("💡 Enter your Google API Key in the sidebar to unlock automated textual insights.")
        return
        
    if st.button(f"Generate Insight for {context}"):
        with st.spinner("Analyzing graph context..."):
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
                prompt = f"Act as a top-tier Data Scientist. I have a dataset with columns {list(df.columns)}. "
                if context == "Correlation":
                    num_cols_only = df.select_dtypes(include=['number'])
                    corr_matrix_str = num_cols_only.corr().round(2).to_string()
                    prompt += f"Here is the Pearson correlation matrix:\n{corr_matrix_str}\n\nList the top 2-3 most interesting relationships and what they imply in business terms (e.g. 'Age is negatively correlated with Income, meaning younger users tend to...'). Keep it brief and insightful."
                
                response = llm.invoke(prompt)
                st.info(response.content)
            except Exception as e:
                st.error(f"Error generating insight: {e}")

def plot_correlation_heatmap(df):
    st.write("### 🌡️ Correlation Heatmap")
    num_cols, _ = get_column_types(df)
    
    if len(num_cols) < 2:
        st.warning("Heatmap requires at least 2 numerical columns.")
        return
        
    corr_matrix = df[num_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    text_auto=".2f", 
                    aspect="auto", 
                    color_continuous_scale="RdBu_r",
                    title="Interactive Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    auto_insight_generator(df, context="Correlation")

def plot_pairplot(df):
    st.write("### 🔗 Scatter Matrix (Pairplot)")
    num_cols, _ = get_column_types(df)
    
    if len(num_cols) < 2:
        st.warning("Scatter matrix requires at least 2 numerical columns.")
        return
        
    cols_to_plot = num_cols[:5]
    if len(num_cols) > 5:
        st.info("Scatter Matrix limited to top 5 numerical columns for rendering performance.")
        
    fig = px.scatter_matrix(df, dimensions=cols_to_plot, title="Scatter Matrix")
    st.plotly_chart(fig, use_container_width=True)
