"""
Token (word) chunker.

WHY COUNT TOKENS INSTEAD OF CHARACTERS?
---------------------------------------
Character chunkers are easy but the *real* limits that matter are the model's
context window and the embedding model's input length — both measured in
TOKENS (roughly words or sub-words), not characters.  This chunker splits on a
token budget so every chunk is a predictable size for the model.

We use a whitespace tokenizer by default (zero dependencies); pass your own
`tokenizer` to plug in a real BPE tokenizer from `tiktoken` / `transformers`.
"""

from __future__ import annotations

import re
from typing import Callable, List

from .base import BaseChunker


class TokenChunker(BaseChunker):
    """
    Split text into windows of `chunk_size` TOKENS (not characters!).

    Args:
        chunk_size: Number of tokens per chunk.
        chunk_overlap: Number of trailing tokens repeated in the next chunk.
        tokenizer: Callable splitting text into a list of token strings.
            Defaults to a whitespace/word tokenizer.
        joiner: String used to stitch tokens back into text (" " default).
    """

    _DEFAULT_TOKEN_RE = re.compile(r"\S+")

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 20,
        tokenizer: Callable[[str], List[str]] | None = None,
        joiner: str = " ",
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.tokenizer = tokenizer or (lambda text: self._DEFAULT_TOKEN_RE.findall(text))
        self.joiner = joiner

    def split_text(self, text: str) -> List[str]:
        tokens = self.tokenizer(text)
        if not tokens:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: List[str] = []

        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.chunk_size]
            if window:
                chunks.append(self.joiner.join(window))
            # Stop once the window reaches the final token.
            if start + self.chunk_size >= len(tokens):
                break

        return chunks
