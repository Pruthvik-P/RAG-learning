"""
Base class for vector stores.

WHAT IS A VECTOR STORE?
-----------------------
A searchable database of (vector, document) pairs.  Given a query vector it
returns the stored documents whose vectors are most similar.

Tiny stores fit in a Python list (this project's InMemoryVectorStore).  Large
ones use dedicated engines (FAISS, Chroma, Qdrant, pgvector...) that add
approximate-nearest-neighbour indexes so search stays fast at millions of
vectors.  The INTERFACE below stays the same, which is the point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..loaders.base import Document

# A search result pairs a Document with its similarity score.
ScoredDocument = Tuple[Document, float]


class BaseVectorStore(ABC):
    """Abstract interface for all vector stores."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Embed and store documents. Returns their ids."""
        raise NotImplementedError

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        """Embed and store raw strings. Returns their ids."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[ScoredDocument]:
        """Return the `k` documents most similar to the query string."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search_by_vector(
        self, query_vector: List[float], k: int = 4
    ) -> List[ScoredDocument]:
        """Return the `k` documents most similar to a pre-computed vector."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Remove documents by id."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of stored documents."""
        raise NotImplementedError
