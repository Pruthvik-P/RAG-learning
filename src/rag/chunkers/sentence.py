"""
Sentence-aware chunker.

Groups complete sentences into chunks that stay under `chunk_size`.  Because
sentences are never split, each chunk is grammatically self-contained — which
usually produces cleaner embeddings than a raw character cut.

We use a lightweight regex sentence splitter so no NLP library is required.
"""

from __future__ import annotations

import re
from typing import List

from .base import BaseChunker


class SentenceChunker(BaseChunker):
    """
    Pack whole sentences together until the size budget is reached.

    Args:
        sentence_overlap: How many trailing sentences of the previous chunk to
            repeat at the start of the next chunk (context continuity).
    """

    # Split after . ! ? while keeping common abbreviations like "e.g." intact
    # is hard in pure regex; for learning this simple version is good enough.
    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        sentence_overlap: int = 1,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.sentence_overlap = sentence_overlap

    def split_text(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []

        for sentence in sentences:
            # Would adding this sentence exceed the budget?  Then flush.
            candidate_len = sum(len(s) + 1 for s in current) + len(sentence)
            if current and candidate_len > self.chunk_size:
                chunks.append(" ".join(current).strip())
                # Carry over the last N sentences for overlap.
                current = current[-self.sentence_overlap :] if self.sentence_overlap else []

            current.append(sentence.strip())

        if current:
            chunks.append(" ".join(current).strip())

        # Drop empties that can appear from leading whitespace.
        return [c for c in chunks if c]

    def _split_sentences(self, text: str) -> List[str]:
        """Normalise whitespace and split text into sentence strings."""
        text = re.sub(r"\s+", " ", text).strip()
        return [s for s in self._SENTENCE_RE.split(text) if s.strip()]
