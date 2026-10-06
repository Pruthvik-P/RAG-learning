"""
Base classes for text chunking.

WHY CHUNK?
----------
Embedding models and LLMs have limited context windows, and a 50-page PDF's
meaning is diluted if you embed it as ONE vector.  Chunking splits long text
into smaller, semantically meaningful pieces so that:

  1. Retrieval is precise (we fetch only the relevant paragraph).
  2. The prompt we send to the LLM stays small and cheap.

The trade-off:
  * Chunks too small -> lose context ("it" refers to what?).
  * Chunks too big   -> retrieve noise, waste tokens.

Two universal knobs:
  * chunk_size    : target length of each chunk
  * chunk_overlap : how many characters neighboring chunks share, so an idea
                    split across a boundary still appears in one chunk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from ..loaders.base import Document


class BaseChunker(ABC):
    """
    Abstract chunker.

    Subclasses implement `split_text` (split one string) and get
    `split_documents` for free (apply to a list of Documents, preserving and
    amending metadata).
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size "
                f"(got overlap={chunk_overlap}, size={chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ----------------------------- to implement ---------------------------- #
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split a single string into a list of chunk strings."""
        raise NotImplementedError

    # --------------------------- shared behavior --------------------------- #
    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """
        Split many Documents into many smaller Documents.

        For every chunk we copy the parent's metadata and add:
          * chunk_index : 0-based position of the chunk inside its parent
          * chunk_total : total number of chunks produced from the parent

        Preserving `source` is critical: it is what lets the final answer cite
        which file (and line range) a fact came from.
        """
        chunks: List[Document] = []

        for document in documents:
            piece_texts = self.split_text(document.page_content)

            for index, piece in enumerate(piece_texts):
                # Start from a copy so we never mutate the original document.
                metadata = dict(document.metadata)
                metadata["chunk_index"] = index
                metadata["chunk_total"] = len(piece_texts)

                chunks.append(
                    Document(page_content=piece, metadata=metadata)
                )

        return chunks
