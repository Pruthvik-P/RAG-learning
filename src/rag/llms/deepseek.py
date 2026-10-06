"""
DeepSeek chat LLM (OpenAI-compatible HTTP API).

DeepSeek exposes the same shape as OpenAI's Chat Completions API:

    POST {base_url}/chat/completions
    Authorization: Bearer <DEEPSEEK_API_KEY>
    {
      "model": "deepseek-chat",
      "messages": [{"role": "user", "content": "..."}],
      "temperature": 0.2,
      "max_tokens": 1024
    }

We call it with urllib (standard library) so the `openai` package is optional.
Models: "deepseek-chat" (V3, general) and "deepseek-reasoner" (R1, reasoning).
Get a key at https://platform.deepseek.com.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import BaseLLM


class DeepSeekLLM(BaseLLM):
    """
    Chat with DeepSeek.

    Args:
        api_key: Your key.  If None, read from DEEPSEEK_API_KEY.
        base_url: API root, default "https://api.deepseek.com/v1".
        model: "deepseek-chat" or "deepseek-reasoner".
        temperature: 0 = focused/deterministic, 1 = creative. RAG likes low.
        max_tokens: Cap on the reply length.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 60,
    ):
        import os

        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key missing. Pass api_key=... or create a .env "
                "with DEEPSEEK_API_KEY=... (see .env.example)."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages to DeepSeek and return the assistant text."""
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }

        request = Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # Surface the API's error message instead of a bare "HTTP 401".
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"DeepSeek API error {exc.code}: {detail or exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach DeepSeek at {self.base_url}: {exc.reason}"
            ) from exc

        # Standard OpenAI-style response shape.
        return data["choices"][0]["message"]["content"]
