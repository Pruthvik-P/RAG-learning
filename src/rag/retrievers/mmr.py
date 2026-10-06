"""
Maximal Marginal Relevance (MMR) retriever.

THE PROBLEM
-----------
Plain top-k similarity often returns several near-duplicate chunks (e.g. five
almost-identical paragraphs).  The LLM then sees the same fact five times and
misses other useful context.

THE FIX (MMR)
-------------
Pick chunks one at a time, balancing two goals:

    MMR = lambda * similarity(chunk, query)
          - (1 - lambda) * max_similarity(chunk, already_selected)

  * First term  rewards RELEVANCE to the question.
  * Second term penalises REDUNDANCY with chunks we already picked.

`lambda` (0..1) controls the trade-off:
  * lambda = 1.0 -> pure relevance (same as plain similarity)
  * lambda = 0.0 -> pure diversity
  * lambda = 0.5 -> balanced (a common default)
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from ..vectorstores.base import BaseVectorStore, ScoredDocument
from .base import BaseRetriever


class MMRRetriever(BaseRetriever):
    """
    Retrieve a relevant yet diverse set of chunks.

    Args:
        vector_store: Store that can return candidate chunks + their vectors.
        lambda_mult: 1.0 = relevance only, 0.0 = diversity only.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        lambda_mult: float = 0.5,
    ):
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be between 0.0 and 1.0")
        self.vector_store = vector_store
        self.lambda_mult = lambda_mult

    def retrieve(self, query: str, k: int = 4) -> List[ScoredDocument]:
        # 1. Fetch a wider candidate pool than we ultimately need, so MMR has
        #    something to choose from.
        fetch_k = max(k * 4, k)
        candidates = self.vector_store.similarity_search(query, k=fetch_k)
        if not candidates:
            return []

        # 2. Grab the actual vectors for those candidates (needed for the
        #    redundancy term).  Falls back gracefully if the store cannot
        #    provide vectors.
        candidate_vectors = self._vectors_for(candidates)
        query_vector = self.vector_store.embedding.embed_query(query)

        selected: List[int] = []
        remaining: List[int] = list(range(len(candidates)))

        # 3. Iteratively select the candidate with the best MMR score.
        while remaining and len(selected) < k:
            best_index: Optional[int] = None
            best_score = float("-inf")

            for i in remaining:
                relevance = self._cosine(query_vector, candidate_vectors[i])

                if selected:
                    redundancy = max(
                        self._cosine(candidate_vectors[i], candidate_vectors[j])
                        for j in selected
                    )
                else:
                    redundancy = 0.0

                score = (
                    self.lambda_mult * relevance
                    - (1.0 - self.lambda_mult) * redundancy
                )
                if score > best_score:
                    best_score = score
                    best_index = i

            selected.append(best_index)
            remaining.remove(best_index)

        return [candidates[i] for i in selected]

    # ------------------------------------------------------------------ #
    def _vectors_for(self, candidates: Sequence[ScoredDocument]) -> List[List[float]]:
        """Get vectors for candidate documents, re-embedding if necessary."""
        vectors: List[Optional[List[float]]] = []
        for document, _ in candidates:
            vectors.append(self.vector_store.get_vector(document.id))
        if any(v is None for v in vectors):
            # Fallback: re-embed using the store's embedder.
            texts = [doc.page_content for doc, _ in candidates]
            return self.vector_store.embedding.embed_documents(texts)
        return [v for v in vectors if v is not None]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
