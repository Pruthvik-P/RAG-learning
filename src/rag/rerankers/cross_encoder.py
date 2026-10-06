"""
Cross-encoder reranker — the "real" neural reranker (OPTIONAL).

WHAT IS A CROSS-ENCODER?
------------------------
A bi-encoder embeds the query and each document SEPARATELY, then compares the
vectors.  It is fast (documents can be embedded ahead of time) but the query
never "sees" the document during encoding.

A cross-encoder feeds (query, document) TOGETHER through one transformer and
outputs a single relevance score.  The query can attend to every document
token, so accuracy is much higher — but it cannot be precomputed, so it is only
affordable on a small candidate list.  Hence: retrieve first, cross-encode
second.

This module needs the optional dependencies:

    pip install sentence-transformers

Everything else in the project still runs without them.
"""

from __future__ import annotations

from typing import List, Optional

from ..vectorstores.base import ScoredDocument
from .base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """
    Rerank with a sentence-transformers `CrossEncoder`.

    Args:
        model_name: Any HuggingFace cross-encoder.  The default is a small,
            fast MS MARCO model suitable for CPU use.
        device: Optional torch device string ("cpu", "cuda").
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CrossEncoderReranker needs the optional 'sentence-transformers' "
                "package. Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        if not documents:
            return []

        pairs = [(query, document.page_content) for document, _ in documents]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        reranked = [(document, float(score)) for (document, _), score in ranked]
        return reranked[:k] if k is not None else reranked
