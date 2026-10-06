"""
Rerankers package.

RETRIEVE vs RERANK
------------------
Retrieval is optimized for SPEED: it must scan millions of chunks, so it uses
cheap similarity (a single dot product).  That is why it often returns a few
"almost right" chunks at the top.

Reranking is a second, SLOWER pass over a small candidate list (say the top 20)
using a much more accurate relevance model.  You retrieve broadly, then rerank
narrowly (a "retrieve-and-rerank" pipeline):

    query -> [retriever] -> 20 candidates -> [reranker] -> best 4 -> LLM

Rerankers here:
    LexicalReranker      -> zero-dependency term-overlap scoring
    CrossEncoderReranker -> optional neural model (needs sentence-transformers)
"""

from __future__ import annotations

from .base import BaseReranker
from .lexical import LexicalReranker

__all__ = [
    "BaseReranker",
    "LexicalReranker",
    "get_reranker",
]


def get_reranker(kind: str = "lexical", **kwargs) -> BaseReranker:
    """
    Factory: build a reranker by name.

    Args:
        kind: "lexical" (default, offline) or "cross_encoder"/"neural"
            (requires the optional `sentence-transformers` package).
    """
    kind = (kind or "lexical").lower()

    if kind in {"lexical", "keyword", "overlap", "offline"}:
        return LexicalReranker(**kwargs)

    if kind in {"cross_encoder", "cross-encoder", "crossencoder", "neural"}:
        # Imported lazily so the default path stays dependency-free.
        from .cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(**kwargs)

    raise ValueError(
        f"Unknown reranker: {kind!r}. Choose 'lexical' or 'cross_encoder'."
    )
