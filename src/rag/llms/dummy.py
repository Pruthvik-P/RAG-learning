"""
Offline placeholder LLM.

Lets you exercise the whole RAG pipeline with NO API key and NO network.  It
cannot actually reason, but it does something pedagogically useful: it returns
the retrieved context verbatim, so you can SEE exactly what a real LLM would
have been given.  That is often enough to tell whether retrieval is working.
"""

from __future__ import annotations

from typing import Dict, List

from .base import BaseLLM


class DummyLLM(BaseLLM):
    """A deterministic fake LLM for offline runs and tests."""

    def __init__(self, note: str | None = None):
        self.note = note or (
            "OFFLINE MODE: no LLM was called. Set DEEPSEEK_API_KEY to get a "
            "real answer. Below is the raw context that would have been sent."
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Find the last user message and echo a trimmed version of it.
        user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        preview = user if len(user) <= 1200 else user[:1200] + "\n...[truncated]"
        return f"{self.note}\n\n--- prompt sent to the model ---\n{preview}"
