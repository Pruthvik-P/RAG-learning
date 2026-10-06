"""
LLMs package.

`get_llm()` returns a DeepSeek client when a key is configured.  If no key is
present and offline fallback is allowed, it returns a DummyLLM so examples and
tests still run end-to-end.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseLLM
from .deepseek import DeepSeekLLM
from .dummy import DummyLLM

__all__ = ["BaseLLM", "DummyLLM", "DeepSeekLLM", "get_llm"]


def get_llm(
    provider: str = "deepseek",
    api_key: Optional[str] = None,
    allow_offline: bool = True,
    **kwargs,
) -> BaseLLM:
    """
    Build an LLM client.

    Args:
        provider: "deepseek" | "dummy".
        api_key: Explicit key; falls back to DEEPSEEK_API_KEY.
        allow_offline: Return DummyLLM instead of raising when no key exists.
    """
    provider = (provider or "deepseek").lower()

    if provider == "dummy":
        return DummyLLM()

    if provider == "deepseek":
        import os

        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            if allow_offline:
                return DummyLLM()
            raise ValueError(
                "No DeepSeek API key found and allow_offline=False."
            )
        # Imported lazily so the dummy path never imports the HTTP client.
        from .deepseek import DeepSeekLLM

        return DeepSeekLLM(api_key=key, **kwargs)

    raise ValueError(f"Unknown LLM provider: {provider!r}")
