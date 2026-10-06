"""
Lexical reranker — a zero-dependency stand-in for a cross-encoder.

HOW IT SCORES
-------------
For each candidate we measure how well its text covers the query, using three
cheap, explainable signals:

  1. coverage   : fraction of DISTINCT query terms that appear in the document.
  2. saturation : a term-frequency score that grows with repeats but flattens
                  out (so spamming a word cannot dominate).
  3. phrase     : a bonus when the whole query appears verbatim.

We then blend that lexical score with the retrieval score (min-max normalised)
so we keep some of the retriever's opinion rather than trusting keywords alone.
This is exactly the shape of a real cross-encoder, just with a much weaker
"model" — which makes it perfect for learning the flow.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..vectorstores.base import ScoredDocument
from .base import BaseReranker


class LexicalReranker(BaseReranker):
    """
    Rerank candidates by query-term overlap blended with the retrieval score.

    Args:
        lexical_weight: Weight on the lexical relevance score (0..1).
        original_weight: Weight on the normalised retrieval score (0..1).
        k1: Term-frequency saturation constant for the saturation signal.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

    def __init__(
        self,
        lexical_weight: float = 0.7,
        original_weight: float = 0.3,
        k1: float = 1.5,
    ):
        self.lexical_weight = lexical_weight
        self.original_weight = original_weight
        self.k1 = k1

    # ------------------------------------------------------------------ #
    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        if not documents:
            return []

        query_terms = set(self._tokenize(query))
        normalised_scores = self._min_max([score for _, score in documents])

        reranked: List[ScoredDocument] = []
        for (document, original_score), original_norm in zip(
            documents, normalised_scores
        ):
            lexical = self._lexical_score(query, query_terms, document.page_content)
            final = (
                self.lexical_weight * lexical
                + self.original_weight * original_norm
            )
            reranked.append((document, final))

        reranked.sort(key=lambda pair: pair[1], reverse=True)
        return reranked[:k] if k is not None else reranked

    # ------------------------------------------------------------------ #
    def _lexical_score(self, query: str, query_terms: set, text: str) -> float:
        """Combine the three signals into a single 0..1 relevance score."""
        if not query_terms:
            return 0.0

        tokens = self._tokenize(text)
        token_counts: dict = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        present = query_terms & set(token_counts)
        coverage = len(present) / len(query_terms)

        matches = sum(token_counts.get(term, 0) for term in query_terms)
        saturation = matches / (matches + self.k1) if matches else 0.0

        phrase = 1.0 if query.strip().lower() in text.lower() else 0.0

        return 0.6 * coverage + 0.3 * saturation + 0.1 * phrase

    def _tokenize(self, text: str) -> List[str]:
        return self._TOKEN_RE.findall(text.lower())

    @staticmethod
    def _min_max(scores: List[float]) -> List[float]:
        """Scale scores to 0..1; if all equal, everyone gets 1.0."""
        if not scores:
            return []
        low, high = min(scores), max(scores)
        if high - low == 0:
            return [1.0] * len(scores)
        return [(score - low) / (high - low) for score in scores]
