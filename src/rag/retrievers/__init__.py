"""
Retrievers package.

`get_retriever()` picks a retrieval strategy by name so the pipeline only ever
asks for "similarity", "mmr", "keyword", or "hybrid".

    SimilarityRetriever -> dense vector search (top-k nearest neighbours)
    MMRRetriever        -> relevant AND diverse chunks
    KeywordRetriever    -> BM25 lexical / exact-term search
    HybridRetriever     -> reciprocal-rank fusion of dense + keyword
"""

from __future__ import annotations

from ..vectorstores.base import BaseVectorStore
from .base import BaseRetriever
from .hybrid import HybridRetriever
from .keyword import KeywordRetriever
from .mmr import MMRRetriever
from .similarity import SimilarityRetriever

__all__ = [
    "BaseRetriever",
    "SimilarityRetriever",
    "MMRRetriever",
    "KeywordRetriever",
    "HybridRetriever",
    "get_retriever",
]


def get_retriever(
    vector_store: BaseVectorStore,
    mode: str = "similarity",
    lambda_mult: float = 0.5,
) -> BaseRetriever:
    """Factory: build a retriever from a strategy name."""
    mode = (mode or "similarity").lower()

    if mode in {"similarity", "sim", "vector", "dense"}:
        return SimilarityRetriever(vector_store)

    if mode in {"mmr", "diverse"}:
        return MMRRetriever(vector_store, lambda_mult=lambda_mult)

    if mode in {"keyword", "bm25", "lexical", "sparse"}:
        return KeywordRetriever(vector_store)

    if mode in {"hybrid", "hybrid_search", "rrf"}:
        return HybridRetriever(vector_store)

    raise ValueError(
        f"Unknown retrieval mode: {mode!r}. Choose 'similarity', 'mmr', "
        "'keyword', or 'hybrid'."
    )
