import streamlit as st
import pandas as pd
import os

try:
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ── Dataset context builder ────────────────────────────────────────────────────
def _build_dataset_context(df: pd.DataFrame) -> str:
    """
    Build a concise text description of the dataset to include in the
    agent's system prompt. Better context = dramatically better answers.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing  = df.isnull().sum()
    missing_info = {c: int(missing[c]) for c in df.columns if missing[c] > 0}

    context = (
        f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns.\n"
        f"Numeric columns ({len(num_cols)}): {num_cols}\n"
        f"Categorical columns ({len(cat_cols)}): {cat_cols}\n"
        f"Missing values: {missing_info if missing_info else 'None'}\n"
        f"Sample (first 3 rows):\n{df.head(3).to_string()}\n"
        f"Summary statistics:\n{df.describe(include='all').round(2).to_string()}"
    )
    return context


# ── Agent factory (cached per dataframe hash) ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def _build_agent(df_hash: int, api_key: str):
    """
    Build and cache the pandas dataframe agent.
    Cached on df_hash so it rebuilds only when data changes.
    The agent is NOT recreated on every message.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )
    return llm


# ── Main entry point ───────────────────────────────────────────────────────────
def render_ai_assistant(df: pd.DataFrame | None):
    st.subheader("AI Assistant — Chat with your Dataset")

    # ── Guards ─────────────────────────────────────────────────────────────────
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.warning(
            "Please upload and process a dataset in previous steps "
            "before using the AI Assistant."
        )
        return

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Enter your Google Gemini API key in the sidebar to use the AI Assistant.")
        return

    if not LANGCHAIN_AVAILABLE:
        st.error(
            "Required packages not installed. Run: "
            "`pip install langchain-experimental langchain-google-genai`"
        )
        return

    # ── Security warning ───────────────────────────────────────────────────────
    st.warning(
        "⚠️ **Security notice:** The AI Assistant executes Python code against your dataset "
        "to answer questions. Only use this with data you own and trust. "
        "Do not use on sensitive or personally identifiable data on shared deployments."
    )

    # ── Dataset summary ────────────────────────────────────────────────────────
    with st.expander("📋 Dataset context given to AI", expanded=False):
        st.text(_build_dataset_context(df))

    # ── Chat controls ──────────────────────────────────────────────────────────
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()

    st.caption(
        "Ask anything about your dataset. Examples:  \n"
        "• *'What are the top 5 rows by sales?'*  \n"
        "• *'Which category has the highest average revenue?'*  \n"
        "• *'Are there any strong correlations between numeric columns?'*"
    )

    # ── Session state init ─────────────────────────────────────────────────────
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # ── Display chat history ───────────────────────────────────────────────────
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ─────────────────────────────────────────────────────────────
    user_prompt = st.chat_input("Ask a question about your data...")

    if user_prompt:
        # Add to history and display immediately
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Build dataset context for system prompt
                    dataset_context = _build_dataset_context(df)

                    # Build conversation history string to give agent memory
                    history_text = ""
                    if len(st.session_state.chat_messages) > 1:
                        history_lines = []
                        for m in st.session_state.chat_messages[:-1]:  # exclude current
                            role = "User" if m["role"] == "user" else "Assistant"
                            history_lines.append(f"{role}: {m['content']}")
                        history_text = (
                            "\n\nPrevious conversation:\n"
                            + "\n".join(history_lines[-10:])  # last 5 turns
                            + "\n\nCurrent question:"
                        )

                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0,
                        google_api_key=api_key,
                    )

                    agent = create_pandas_dataframe_agent(
                        llm,
                        df,
                        verbose=False,
                        allow_dangerous_code=True,
                        handle_parsing_errors=True,
                        prefix=(
                            "You are a Senior Data Analyst assistant. "
                            "Answer questions about the user's dataset clearly and concisely. "
                            "Always show your reasoning. If asked to plot, describe the chart instead.\n\n"
                            f"Dataset context:\n{dataset_context}"
                        ),
                    )

                    # Include history in the prompt for conversational memory
                    full_prompt = history_text + user_prompt if history_text else user_prompt
                    response    = agent.invoke(full_prompt)
                    output_text = response.get("output", str(response))

                    st.markdown(output_text)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": output_text}
                    )

                except Exception as e:
                    error_msg = (
                        f"I couldn't answer that question. Error: `{e}`  \n\n"
                        "Try rephrasing, or ask a simpler question like "
                        "'What are the column names?' to verify the connection works."
                    )
                    st.error(error_msg)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": error_msg}
                    )