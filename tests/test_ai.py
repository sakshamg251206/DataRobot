import pandas as pd
import pytest

from autods.ai import client as client_module
from autods.ai.client import AIError, ChatTurn, GeminiClient, _friendly
from autods.ai.prompts import (
    MAX_CONTEXT_CHARS,
    assistant_system_prompt,
    column_insight_prompt,
    dataset_context,
    relationship_prompt,
)


def test_dataset_context_is_compact_and_informative(classification_df):
    wide = pd.concat(
        [classification_df] + [classification_df.add_suffix(f"_{i}") for i in range(30)], axis=1
    )
    context = dataset_context(wide)
    assert len(context) <= MAX_CONTEXT_CHARS + 30
    assert "rows" in context
    assert "label" in dataset_context(classification_df)


def test_prompts_cover_column_types(classification_df):
    assert "Skewness" in column_insight_prompt(classification_df, "x1")
    assert "Categorical" in column_insight_prompt(classification_df, "color")
    assert "correlation" in relationship_prompt(classification_df, "x1", "x2").lower()
    assert "Mean of" in relationship_prompt(classification_df, "x1", "color")
    assert "Cross-tabulation" in relationship_prompt(classification_df, "color", "label")
    assert "cannot run code" in assistant_system_prompt("ctx")


def test_client_requires_key():
    with pytest.raises(AIError, match="API key"):
        GeminiClient(api_key="", model="m")


def test_client_maps_history_and_errors(monkeypatch):
    calls = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls["contents"] = contents
            if contents[-1].parts[0].text == "boom":
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return type("R", (), {"text": " hello "})()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(client_module.genai, "Client", FakeClient)
    client = GeminiClient(api_key="k", model="m")
    history = [ChatTurn("user", "hi"), ChatTurn("assistant", "hey")]
    assert client.generate("question", history=history) == "hello"
    assert [c.role for c in calls["contents"]] == ["user", "model", "user"]
    with pytest.raises(AIError, match="quota"):
        client.generate("boom")


def test_friendly_error_messages():
    assert "rejected" in _friendly(RuntimeError("API key not valid"))
    assert "failed" in _friendly(RuntimeError("something else"))
