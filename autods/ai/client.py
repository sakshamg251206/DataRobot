"""Thin wrapper around the Google Gen AI SDK with user-friendly errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from google import genai
    from google.genai import types

    GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    GENAI_AVAILABLE = False


class AIError(RuntimeError):
    """Raised with a message that is safe to show to the user."""


@dataclass(frozen=True)
class ChatTurn:
    role: str  # "user" or "assistant"
    content: str


def _friendly(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "api key" in lowered or "api_key_invalid" in lowered or "permission" in lowered:
        return "The Gemini API key was rejected. Check that it is correct and enabled."
    if "quota" in lowered or "429" in lowered or "resource_exhausted" in lowered:
        return "The Gemini API quota is exhausted or rate-limited. Wait a moment and try again."
    if "not found" in lowered and "model" in lowered:
        return "The configured Gemini model was not found. Set GEMINI_MODEL to an available model."
    return f"The AI request failed: {text[:300]}"


class GeminiClient:
    """Generate text with a Gemini model. One instance per user session."""

    def __init__(self, api_key: str, model: str) -> None:
        if not GENAI_AVAILABLE:  # pragma: no cover
            raise AIError("Install the `google-genai` package to enable AI features.")
        if not api_key:
            raise AIError("Add a Gemini API key to enable AI features.")
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        history: list[ChatTurn] | None = None,
        temperature: float = 0.3,
    ) -> str:
        contents: list[Any] = [
            types.Content(
                role="model" if turn.role == "assistant" else "user",
                parts=[types.Part(text=turn.content)],
            )
            for turn in history or []
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        config = types.GenerateContentConfig(temperature=temperature, system_instruction=system)
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as exc:
            raise AIError(_friendly(exc)) from exc
        text = (response.text or "").strip()
        if not text:
            raise AIError("The model returned an empty response. Try rephrasing the question.")
        return text
