"""
Offline "hashing trick" embedding — zero dependencies, no downloads.

WHY THIS EXISTS
---------------
Real embedding models are neural networks (hundreds of MB).  That's great for
quality but bad for a first lesson (slow, needs internet, needs big packages).
This embedding lets you run the ENTIRE RAG pipeline with just the standard
library so you can focus on the pipeline, then swap in a real model later.

HOW IT WORKS (the "hashing trick")
----------------------------------
1. Tokenize the text into words.
2. For each word (and word n-grams), hash it to an index in [0, dimensions).
3. Add a signed count at that index (the sign reduces collision bias).
4. L2-normalize the vector so cosine similarity is meaningful.

It is basically a bag-of-words/TF model mapped into a fixed-size vector.
It captures exact keyword overlap (lexical similarity), NOT deep semantics —
but the geometry and the pipeline are identical to a real embedder.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List, Sequence, Tuple

from .base import BaseEmbedding


class HashingEmbedding(BaseEmbedding):
    """
    Deterministic hashing embedder.

    Args:
        dimensions: Vector size.  Larger = fewer hash collisions = better.
        ngram_range: Word n-grams to include, e.g. (1, 2) uses unigrams AND
            bigrams, which captures a little more word-order information.
        lowercase: Fold case before hashing.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

    def __init__(
        self,
        dimensions: int = 1024,
        ngram_range: Tuple[int, int] = (1, 2),
        lowercase: bool = True,
    ):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        self.ngram_range = ngram_range
        self.lowercase = lowercase

    @property
    def dimensions(self) -> int:
        return self._dimensions

    # ------------------------------------------------------------------ #
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    # ------------------------------------------------------------------ #
    def _tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return self._TOKEN_RE.findall(text)

    def _ngrams(self, tokens: Sequence[str]) -> Iterable[str]:
        """Yield word n-grams joined by spaces across the requested range."""
        start, end = self.ngram_range
        for n in range(start, end + 1):
            if n == 1:
                yield from tokens
            else:
                for i in range(len(tokens) - n + 1):
                    yield " ".join(tokens[i : i + n])

    def _hash(self, token: str) -> int:
        """
        Stable hash of a token.

        IMPORTANT: we use md5 (from hashlib) instead of Python's built-in
        hash(), because built-in hashing is randomized per process.  That would
        make vectors differ between runs and break any saved index!
        """
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self._dimensions

        for token in self._ngrams(self._tokenize(text)):
            hashed = self._hash(token)
            index = hashed % self._dimensions
            # Use one bit of the hash to choose a sign (+1/-1).  This makes
            # colliding terms cancel on average instead of always adding up.
            sign = 1.0 if (hashed // self._dimensions) % 2 == 0 else -1.0
            vector[index] += sign

        return self._normalize(vector)

    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        """Scale the vector to unit length (so dot product == cosine sim)."""
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
