"""
Keyword retriever using BM25 — sparse, lexical search.

WHY A KEYWORD RETRIEVER?
------------------------
Vector search matches *meaning* but can miss exact strings: names, error codes,
acronyms, rare terms.  A classic keyword/lexical retriever handles those well
because it scores literal term overlap.  Combining the two is called HYBRID
search (see hybrid.py).

BM25 is the standard ranking function behind Elasticsearch/Lucene.  Intuitively
it rewards a document when:
  * the query term appears often in it (term frequency), with diminishing
    returns (so 100 repeats are not 100x better), and
  * the term is rare across the whole corpus (inverse document frequency),
  * while penalising very long documents (length normalisation).

This implementation is pure Python and reads the documents straight from the
vector store, so it needs no separate index server.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

from ..vectorstores.base import BaseVectorStore, ScoredDocument
from .base import BaseRetriever


class KeywordRetriever(BaseRetriever):
    """
    Rank stored documents with BM25 over a shared tokenizer.

    Args:
        vector_store: Store holding the documents to search.
        k1: Term-frequency saturation (1.2–2.0 typical).  Higher lets repeated
            terms keep adding score for longer.
        b: Length-normalisation strength (0 = ignore length, 1 = full).
        tokenizer: Callable text -> list of tokens.  Defaults to lowercased
            word tokens.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

    def __init__(
        self,
        vector_store: BaseVectorStore,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.vector_store = vector_store
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        return self._TOKEN_RE.findall(text.lower())

    def retrieve(self, query: str, k: int = 4) -> List[ScoredDocument]:
        documents = self.vector_store.documents
        if not documents:
            return []

        # --- 1. Build the corpus statistics once per query. ------------- #
        tokenized = [self._tokenize(doc.page_content) for doc in documents]
        lengths = [len(tokens) for tokens in tokenized]
        avg_length = (sum(lengths) / len(lengths)) if lengths else 0.0
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized:
            doc_freq.update(set(tokens))
        n_docs = len(documents)

        # --- 2. Score every document against the query terms. ----------- #
        query_terms = self._tokenize(query)
        scored: List[ScoredDocument] = []

        for document, tokens, length in zip(documents, tokenized, lengths):
            counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * (length / avg_length if avg_length else 0.0)
                )
                score += idf * (freq * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append((document, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
