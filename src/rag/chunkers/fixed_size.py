"""
Fixed-size character chunker.

The simplest strategy: walk through the text in windows of `chunk_size`
characters, moving forward by `chunk_size - chunk_overlap` each time.

Pros:  predictable chunk sizes, trivially fast, easy to reason about.
Cons:  can cut words/sentences in half at boundaries.

Great first chunker to understand the mechanics before moving to smarter ones.
"""

from __future__ import annotations

from typing import List

from .base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed-length character windows with overlap.

    Example (size=10, overlap=3):
        "ABCDEFGHIJKLMNOPQRST"
        -> "ABCDEFGHIJ"
        -> "HIJKLMNOPQ"   (last 3 chars "HIJ" repeated for continuity)
        -> "QRST"
    """

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        # Step size is how far the window moves each iteration.  Because the
        # window is `chunk_size` wide, the overlap equals chunk_size - step.
        step = self.chunk_size - self.chunk_overlap
        chunks: List[str] = []

        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            if chunk.strip():  # skip whitespace-only chunks
                chunks.append(chunk)

            # Stop once the window reaches the end of the string.
            if start + self.chunk_size >= len(text):
                break

        return chunks
