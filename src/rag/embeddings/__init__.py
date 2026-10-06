"""
Embeddings package.

`get_embedding()` is a small factory that reads the configured provider name
and returns the matching embedder.  This is the one place the rest of the
pipeline needs to know about, so switching providers is a one-line change.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseEmbedding
from .hashing import HashingEmbedding

__all__ = [
    "BaseEmbedding",
    "HashingEmbedding",
    "get_embedding",
]


def get_embedding(
    provider: str = "hashing",
    dimensions: int = 1024,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs,
) -> BaseEmbedding:
    """
    Factory that builds an embedding model by name.

    Args:
        provider: "hashing" | "sentence_transformer" | "openai".

    Extra keyword args are forwarded to the chosen class.
    """
    provider = (provider or "hashing").lower()

    if provider in {"hashing", "hash", "offline"}:
        return HashingEmbedding(dimensions=dimensions, **kwargs)

    if provider in {"sentence_transformer", "sentence-transformers", "st"}:
        # Imported lazily so the default path never needs the heavy package.
        from .sentence_transformer import SentenceTransformerEmbedding

        return SentenceTransformerEmbedding(model_name=model, **kwargs)

    if provider in {"openai", "openai_compatible"}:
        from .openai_compatible import OpenAICompatibleEmbedding

        return OpenAICompatibleEmbedding(model=model, dimensions=dimensions, **kwargs)

    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        "Choose from: hashing, sentence_transformer, openai."
    )
