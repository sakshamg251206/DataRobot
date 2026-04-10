import streamlit as st
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

def render_ai_assistant(df):
    st.header("🤖 AI Assistant: Chat with your Dataset")
    
    # Needs valid dataframe
    if df is None or not isinstance(df, pd.DataFrame):
        st.warning("Please upload and process a dataset in previous steps before querying the AI.")
        return
        
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ Please enter your Google API Key in the sidebar to use the AI Assistant.")
        return
        
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        
    st.markdown("Ask anything about your dataset! For example: *'Why did sales drop?'* or *'Which category is the most profitable?'*")
    
    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Chat Input
    prompt = st.chat_input("Ask a question about your data...")
    if prompt:
        # Add user message to state and view
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing dataset... (this may take a moment)"):
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
                    
                    # Ensure agent executes python code under the hood allowing iteratives
                    agent = create_pandas_dataframe_agent(
                        llm, 
                        df, 
                        verbose=False, 
                        allow_dangerous_code=True,
                        handle_parsing_errors=True
                    )
                    
                    response = agent.invoke(prompt)
                    output_text = response.get("output", str(response))
                    
                    st.markdown(output_text)
                    st.session_state.chat_messages.append({"role": "assistant", "content": output_text})
                except Exception as e:
                    st.error(f"Error querying AI Assistant: {e}")
