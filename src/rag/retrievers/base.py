"""
Base class for retrievers.

A vector store can already answer "nearest neighbours".  A RETRIEVER is the
layer that decides the retrieval STRATEGY:

    * SimilarityRetriever -> plain top-k nearest neighbours
    * MMRRetriever        -> top-k that are both relevant AND diverse

Keeping retrieval separate from storage lets you swap strategies (and later add
hybrid keyword+vector search) without touching the store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..vectorstores.base import ScoredDocument


class BaseRetriever(ABC):
    """Abstract retriever: given a query, return scored documents."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> List[ScoredDocument]:
        """
        Return up to `k` (Document, score) pairs for `query`, best first.
        """
        raise NotImplementedError
