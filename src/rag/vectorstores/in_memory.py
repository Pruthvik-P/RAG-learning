"""
In-memory vector store using cosine similarity.

This is the clearest way to internalise what a vector store actually does:

    for every stored vector:
        score = cosine_similarity(query_vector, stored_vector)
    return the top-k by score

No index magic.  It is O(n) per query, which is fine for learning and for
small/medium knowledge bases.  For millions of vectors you'd swap in FAISS,
Chroma, etc. — the public methods here are designed to match those tools.

You can also persist the store to a JSON file with `save()` / `load()`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from ..embeddings.base import BaseEmbedding
from ..loaders.base import Document
from .base import BaseVectorStore, ScoredDocument


class InMemoryVectorStore(BaseVectorStore):
    """
    A `list`-backed vector store.

    Args:
        embedding: Any BaseEmbedding.  Used to embed both stored texts and
            incoming queries so they live in the same vector space.
    """

    def __init__(self, embedding: BaseEmbedding):
        self.embedding = embedding
        # Parallel lists: index i describes the same document everywhere.
        self._documents: List[Document] = []
        self._vectors: List[List[float]] = []
        self._ids: List[str] = []

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Embed every document's text and store it."""
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        self._vectors.extend(self.embedding.embed_documents(texts))
        self._documents.extend(documents)
        self._ids.extend(doc.id for doc in documents)
        return [doc.id for doc in documents]

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        """Wrap raw strings in Documents, embed, and store them."""
        metadatas = metadatas or [{} for _ in texts]
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        return self.add_documents(documents)

    # ------------------------------------------------------------------ #
    # Reading / search
    # ------------------------------------------------------------------ #
    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[ScoredDocument]:
        """Embed the query then delegate to the vector search."""
        query_vector = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(query_vector, k=k)

    def similarity_search_by_vector(
        self, query_vector: List[float], k: int = 4
    ) -> List[ScoredDocument]:
        """Brute-force cosine similarity over every stored vector."""
        if not self._vectors:
            return []

        scored = [
            (document, self._cosine(query_vector, vector))
            for document, vector in zip(self._documents, self._vectors)
        ]

        # Highest similarity first; JSON serialisable float scores.
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """
        Cosine similarity = dot(a, b) / (|a| * |b|).

        Ranges from -1 (opposite) to 1 (identical direction).  For embeddings
        this measures how aligned the two meanings are, ignoring length.
        """
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #
    def get_vector(self, doc_id: str) -> Optional[List[float]]:
        """
        Return the stored vector for a document id, or None if absent.

        Used by strategies like MMR that need the raw vectors of candidates.
        """
        try:
            index = self._ids.index(doc_id)
        except ValueError:
            return None
        return self._vectors[index]

    def delete(self, ids: List[str]) -> None:
        """Remove entries by id, keeping the three parallel lists in sync."""
        drop = set(ids)
        keep = [
            i for i, doc_id in enumerate(self._ids) if doc_id not in drop
        ]
        self._documents = [self._documents[i] for i in keep]
        self._vectors = [self._vectors[i] for i in keep]
        self._ids = [self._ids[i] for i in keep]

    @property
    def documents(self) -> List[Document]:
        """Read-only-ish access to the stored documents."""
        return list(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

    # ------------------------------------------------------------------ #
    # Persistence (save the whole index to disk as JSON)
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Serialise documents, vectors, and ids to a JSON file."""
        payload = {
            "dimensions": self.embedding.dimensions,
            "ids": self._ids,
            "vectors": self._vectors,
            "documents": [
                {"page_content": doc.page_content, "metadata": doc.metadata, "id": doc.id}
                for doc in self._documents
            ],
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(
        cls, path: str, embedding: BaseEmbedding
    ) -> "InMemoryVectorStore":
        """Recreate a store from a file written by `save()`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(embedding=embedding)
        store._ids = data["ids"]
        store._vectors = [[float(x) for x in row] for row in data["vectors"]]
        store._documents = [
            Document(
                page_content=item["page_content"],
                metadata=item.get("metadata", {}),
                id=item.get("id"),
            )
            for item in data["documents"]
        ]
        return store
