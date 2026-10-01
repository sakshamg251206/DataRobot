"""AI assistant: ask questions about the dataset in plain English."""

import streamlit as st

from autods.ai.agent import agent_available, run_code_agent
from autods.ai.client import AIError, ChatTurn
from autods.ai.prompts import assistant_system_prompt, dataset_context
from autods.config import get_settings
from autods.ui import state
from autods.ui.components import ai_client, dataset_picker, page_header, require_data

page_header(
    "AI assistant",
    "Ask questions about your data in plain English — what's in it, what stands out, "
    "what to do next.",
    eyebrow="Step 5 · Share",
)
require_data()

if not state.api_key():
    with st.container(border=True):
        st.markdown("#### Add a Gemini API key to chat with your data")
        st.markdown(
            "1. Create a free key at [Google AI Studio](https://aistudio.google.com/apikey).\n"
            "2. Paste it under **AI settings** in the sidebar.\n\n"
            "Only summary statistics and the first few rows are sent to Google — never the "
            "whole file."
        )
    st.stop()

settings = get_settings()
_, df = dataset_picker("chat_version")
context = dataset_context(df)
use_agent = False
if settings.enable_code_agent:
    if agent_available():
        use_agent = st.toggle(
            "Let the AI run Python on the data",
            help="Enabled by the server operator (ENABLE_CODE_AGENT). The AI writes and runs "
            "pandas code to compute exact answers. Only use with data you trust.",
        )
    else:
        st.caption("ENABLE_CODE_AGENT is set but the optional `agent` extra is not installed.")

top_left, top_right = st.columns([4, 1])
with top_left, st.expander("What the AI sees", icon=":material/visibility:"):
    st.text(context)
if top_right.button("Clear chat", icon=":material/delete:", width="stretch"):
    state.clear_chat()
    st.rerun()

history = state.chat_history()
if not history:
    st.caption("Try asking:")
    suggestions = [
        "Summarise this dataset in three bullet points.",
        "Which columns have data quality problems?",
        "What would be a good column to predict, and why?",
    ]
    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions, strict=True):
        if col.button(suggestion, width="stretch"):
            st.session_state.pending_question = suggestion
            st.rerun()

for turn in history:
    with st.chat_message(turn.role):
        st.markdown(turn.content)

question = st.chat_input("Ask about your data…") or st.session_state.pop("pending_question", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Thinking…"):
        try:
            if use_agent:
                answer = run_code_agent(df, question, state.api_key(), settings.gemini_model)
            else:
                client = ai_client()
                if client is None:
                    raise AIError("Could not start the AI client. Check the API key.")
                answer = client.generate(
                    question,
                    system=assistant_system_prompt(context),
                    history=history[-10:],
                )
        except AIError as exc:
            st.error(str(exc), icon=":material/error:")
        else:
            st.markdown(answer)
            state.append_chat(ChatTurn("user", question))
            state.append_chat(ChatTurn("assistant", answer))
