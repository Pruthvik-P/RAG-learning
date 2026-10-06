"""
OpenAI-compatible embedding provider (HTTP, optional).

DeepSeek does NOT currently offer an embeddings endpoint, so if you want a
hosted embedder you would point this class at OpenAI, Together, Voyage, or a
local server that mimics the OpenAI `/embeddings` API.

Uses urllib from the standard library, so the `openai` package is NOT required.
"""

from __future__ import annotations

import json
from typing import List, Optional
from urllib.request import Request, urlopen

from .base import BaseEmbedding


class OpenAICompatibleEmbedding(BaseEmbedding):
    """
    Call any OpenAI-style `POST /embeddings` endpoint.

    Args:
        api_key: Bearer token.  Falls back to the OPENAI_API_KEY env var.
        base_url: e.g. "https://api.openai.com/v1".
        model: embedding model name, e.g. "text-embedding-3-small".
        dimensions: set to request a smaller output dimension (some models
            support truncation via the `dimensions` request field).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        timeout: int = 60,
    ):
        import os

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Pass api_key=... or set OPENAI_API_KEY."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimensions = dimensions
        self.timeout = timeout

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": texts}

        request = Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        # The API may return items out of order; sort by `index` to be safe.
        items = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]
