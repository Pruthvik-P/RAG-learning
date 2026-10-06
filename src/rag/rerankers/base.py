"""
Base class for rerankers.

A reranker takes the ORDERED candidates produced by a retriever and returns a
new ORDER, usually scored by a more expensive relevance function.  Keeping the
interface this small means any reranker (lexical, neural, LLM) can be dropped
into the RAG pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..vectorstores.base import ScoredDocument


class BaseReranker(ABC):
    """Abstract reranker: reorder (and trim) a list of scored candidates."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        """
        Return the candidates reordered by relevance, best first.

        Args:
            query: The user's question.
            documents: (Document, retrieval_score) candidates from a retriever.
            k: Keep only the top `k` after reranking.  None keeps all.
        """
        raise NotImplementedError
