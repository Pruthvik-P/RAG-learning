"""
Base class for embedding models.

WHAT IS AN EMBEDDING?
---------------------
An embedding maps text to a fixed-length list of numbers (a vector) such that
texts with similar MEANING end up close together.  "distance" between vectors
(usually cosine similarity) becomes a proxy for semantic similarity.

RAG relies on this: we embed every chunk, embed the user's question, and fetch
the chunks whose vectors are closest to the question vector.

Two methods matter:
  * embed_documents(texts) -> vectors for many chunks (bulk ingestion)
  * embed_query(text)      -> vector for one question  (search time)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """Abstract interface implemented by every embedding provider."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Length of the vectors this model produces."""
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of chunk texts. Returns one vector per input text."""
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        Default implementation just reuses `embed_documents`.  Some providers
        (e.g. certain OpenAI models) use DIFFERENT instructions for queries vs
        documents and override this method to reflect that.
        """
        return self.embed_documents([text])[0]

    # Convenience so `len(embedding_model)` reflects vector size.
    def __len__(self) -> int:
        return self.dimensions
