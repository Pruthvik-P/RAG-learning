"""
Recursive character chunker — the workhorse used by most RAG systems.

IDEA
----
Instead of blindly cutting every N characters, we try to split on "natural"
boundaries in priority order:

    paragraph  ->  line  ->  sentence  ->  word  ->  character

We first split on paragraphs.  If a piece is still too big, we split THAT
piece on the next separator, and so on.  This keeps paragraphs and sentences
intact whenever possible while still respecting the size limit.

This mirrors LangChain's `RecursiveCharacterTextSplitter`.
"""

from __future__ import annotations

from typing import List

from .base import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Split text by trying separators from most to least semantic.

    Args:
        separators: Ordered list of boundary strings.  The algorithm prefers
            earlier entries.  The final "" means "hard cut as a last resort".
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: List[str] | None = None,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    # ------------------------------------------------------------------ #
    # Recursive core
    # ------------------------------------------------------------------ #
    def _split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split `text` using the current separator priority list."""
        # Choose the highest-priority separator that actually occurs.
        separator = separators[-1]
        remaining = []
        for index, candidate in enumerate(separators):
            if candidate == "":
                separator = candidate
                break
            if candidate in text:
                separator = candidate
                # Remember the lower-priority separators for the next level.
                remaining = separators[index + 1 :]
                break

        # Perform the split.  An empty separator means split into characters.
        if separator:
            pieces = text.split(separator)
            # Re-attach the separator except at the very start, so we lose no
            # characters (important for markdown list markers, etc.).
            pieces = [
                piece if i == 0 else separator + piece
                for i, piece in enumerate(pieces)
            ]
        else:
            pieces = list(text)

        # Keep small pieces; recursively split any piece that is too large.
        chunks: List[str] = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                chunks.append(piece)
            elif remaining:
                chunks.extend(self._split(piece, remaining))
            else:
                # No separators left: force a hard character cut.
                chunks.extend(self._hard_split(piece))

        # Merge tiny neighbours back together (up to chunk_size) so we don't
        # produce dozens of one-word chunks.
        return self._merge(chunks)

    def _hard_split(self, text: str) -> List[str]:
        """Last resort: cut a too-long, separator-less string by size."""
        step = self.chunk_size - self.chunk_overlap
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), step)
            if text[i : i + self.chunk_size].strip()
        ]

    def _merge(self, pieces: List[str]) -> List[str]:
        """Greedily combine adjacent small pieces without exceeding chunk_size."""
        merged: List[str] = []
        current = ""

        for piece in pieces:
            if not piece:
                continue
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                if current.strip():
                    merged.append(current)
                current = piece

        if current.strip():
            merged.append(current)

        return merged
