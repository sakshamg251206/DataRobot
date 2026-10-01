"""Opt-in code-executing agent (LangChain pandas agent).

The agent writes and runs Python against the in-memory DataFrame, which
means a crafted dataset or prompt could execute arbitrary code on the
server. It is therefore disabled unless ``ENABLE_CODE_AGENT=true`` *and* the
optional ``agent`` extra is installed. Only enable it for local, single-user use.
"""

from __future__ import annotations

import pandas as pd

from autods.ai.client import AIError
from autods.ai.prompts import ANALYST_SYSTEM


def agent_available() -> bool:
    try:
        import langchain_experimental.agents  # noqa: F401
        import langchain_google_genai  # noqa: F401
    except ImportError:
        return False
    return True


def run_code_agent(df: pd.DataFrame, question: str, api_key: str, model: str) -> str:
    if not agent_available():
        raise AIError("Install the optional extra: pip install -e '.[agent]'")
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        prefix=ANALYST_SYSTEM,
        max_iterations=8,
        verbose=False,
    )
    try:
        response = agent.invoke({"input": question})
    except Exception as exc:
        raise AIError(f"The code agent failed: {str(exc)[:300]}") from exc
    return str(response.get("output", response))
