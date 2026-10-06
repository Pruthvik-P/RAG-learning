"""
Similarity retriever — the baseline strategy.

Embeds the query and asks the vector store for the k closest chunks.  This is
what "retrieval" means 90% of the time.  Everything else is a refinement.
"""

from __future__ import annotations

from typing import List

from ..vectorstores.base import BaseVectorStore, ScoredDocument
from .base import BaseRetriever


class SimilarityRetriever(BaseRetriever):
    """Return the top-k most similar chunks."""

    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 4) -> List[ScoredDocument]:
        return self.vector_store.similarity_search(query, k=k)
