"""
Retrievers package.

`get_retriever()` picks a retrieval strategy by name so the pipeline only ever
asks for "similarity" or "mmr".
"""

from __future__ import annotations

from ..vectorstores.base import BaseVectorStore
from .base import BaseRetriever
from .mmr import MMRRetriever
from .similarity import SimilarityRetriever

__all__ = [
    "BaseRetriever",
    "SimilarityRetriever",
    "MMRRetriever",
    "get_retriever",
]


def get_retriever(
    vector_store: BaseVectorStore,
    mode: str = "similarity",
    lambda_mult: float = 0.5,
) -> BaseRetriever:
    """Factory: build a retriever from a strategy name."""
    mode = (mode or "similarity").lower()

    if mode in {"similarity", "sim", "vector"}:
        return SimilarityRetriever(vector_store)

    if mode in {"mmr", "diverse"}:
        return MMRRetriever(vector_store, lambda_mult=lambda_mult)

    raise ValueError(
        f"Unknown retrieval mode: {mode!r}. Choose 'similarity' or 'mmr'."
    )
