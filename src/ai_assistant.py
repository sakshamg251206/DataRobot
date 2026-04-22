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

    # FIX: truncate describe output for very wide datasets to avoid token blowout
    try:
        stats_str = df.describe(include="all").round(2).to_string()
        if len(stats_str) > 3000:
            stats_str = df.describe(include="all").round(2).iloc[:, :10].to_string()
            stats_str += f"\n... (truncated — showing first 10 of {df.shape[1]} columns)"
    except Exception:
        stats_str = "Statistics unavailable."

    context = (
        f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns.\n"
        f"Numeric columns ({len(num_cols)}): {num_cols}\n"
        f"Categorical columns ({len(cat_cols)}): {cat_cols}\n"
        f"Missing values: {missing_info if missing_info else 'None'}\n"
        f"Sample (first 3 rows):\n{df.head(3).to_string()}\n"
        f"Summary statistics:\n{stats_str}"
    )
    return context


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

    # ── Session state init (safe fallback) ────────────────────────────────────
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
                    dataset_context = _build_dataset_context(df)

                    # Build conversation history string for memory
                    history_text = ""
                    prior_msgs   = st.session_state.chat_messages[:-1]   # exclude current
                    if prior_msgs:
                        history_lines = []
                        for m in prior_msgs[-10:]:  # last 5 turns (10 messages)
                            role = "User" if m["role"] == "user" else "Assistant"
                            history_lines.append(f"{role}: {m['content']}")
                        history_text = (
                            "\n\nPrevious conversation:\n"
                            + "\n".join(history_lines)
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
                        # FIX: use agent_type string for compatibility with newer langchain
                        agent_type="openai-tools",
                        prefix=(
                            "You are a Senior Data Analyst assistant. "
                            "Answer questions about the user's dataset clearly and concisely. "
                            "Always show your reasoning. If asked to plot, describe the chart instead.\n\n"
                            f"Dataset context:\n{dataset_context}"
                        ),
                    )

                    full_prompt = history_text + user_prompt if history_text else user_prompt
                    response    = agent.invoke(full_prompt)
                    output_text = response.get("output", str(response))

                    st.markdown(output_text)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": output_text}
                    )

                except Exception as e:
                    # FIX: try a simpler fallback without agent if agent construction fails
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash",
                            temperature=0,
                            google_api_key=api_key,
                        )
                        fallback_prompt = (
                            f"You are a data analyst. Here is a dataset overview:\n"
                            f"{_build_dataset_context(df)}\n\n"
                            f"Answer this question based only on the above context "
                            f"(you cannot run code):\n{user_prompt}"
                        )
                        resp        = llm.invoke(fallback_prompt)
                        output_text = resp.content + "\n\n*(answered without code execution — install langchain-experimental for full analysis)*"
                        st.markdown(output_text)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": output_text}
                        )
                    except Exception as e2:
                        error_msg = (
                            f"I couldn't answer that question. Error: `{e}` / `{e2}`  \n\n"
                            "Try rephrasing, or ask a simpler question like "
                            "'What are the column names?' to verify the connection works."
                        )
                        st.error(error_msg)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": error_msg}
                        )