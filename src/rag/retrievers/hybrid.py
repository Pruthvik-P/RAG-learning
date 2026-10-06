"""
Hybrid retriever — fuse dense (vector) and sparse (keyword) search.

THE IDEA
--------
Dense embeddings capture meaning but can miss exact terms.  BM25 captures exact
terms but misses paraphrases.  Running BOTH and combining their rankings covers
each other's blind spots.

HOW TO COMBINE RANKINGS?
------------------------
Scores from the two retrievers live on completely different scales (cosine is
0..1; BM25 is unbounded), so adding raw scores is meaningless.  Instead we use
RECIPROCAL RANK FUSION (RRF), which only looks at each document's RANK:

    RRF(d) = sum over retrievers of  weight / (rrf_k + rank(d))

RRF is simple, scale-free, and famously hard to beat.  `rrf_k` (default 60,
from the original paper) damps the influence of the very top ranks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..loaders.base import Document
from ..vectorstores.base import BaseVectorStore, ScoredDocument
from .base import BaseRetriever
from .keyword import KeywordRetriever
from .similarity import SimilarityRetriever


class HybridRetriever(BaseRetriever):
    """
    Fuse a dense retriever and a keyword retriever with RRF.

    Args:
        vector_store: Store backing both the vector and keyword searches.
        dense_weight: RRF weight for the vector retriever.
        sparse_weight: RRF weight for the keyword (BM25) retriever.
        rrf_k: Rank-fusion constant; larger = flatter contribution.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.dense = SimilarityRetriever(vector_store)
        self.sparse = KeywordRetriever(vector_store)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 4) -> List[ScoredDocument]:
        fetch_k = max(k * 4, k)
        dense_results = self.dense.retrieve(query, k=fetch_k)
        sparse_results = self.sparse.retrieve(query, k=fetch_k)

        fused_scores: Dict[str, float] = {}
        documents: Dict[str, Document] = {}

        def fuse(results: List[ScoredDocument], weight: float) -> None:
            for rank, (document, _) in enumerate(results, start=1):
                documents[document.id] = document
                fused_scores[document.id] = (
                    fused_scores.get(document.id, 0.0)
                    + weight / (self.rrf_k + rank)
                )

        fuse(dense_results, self.dense_weight)
        fuse(sparse_results, self.sparse_weight)

        ranked: List[Tuple[str, float]] = sorted(
            fused_scores.items(), key=lambda item: item[1], reverse=True
        )
        return [(documents[doc_id], score) for doc_id, score in ranked[:k]]
