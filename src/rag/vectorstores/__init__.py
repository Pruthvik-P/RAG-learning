"""
Vector stores package.

Currently a single, dependency-free in-memory implementation.  The abstract
`BaseVectorStore` defines the interface so a FAISS/Chroma/pgvector backend can
be dropped in without touching the retrievers or pipeline.
"""

from .base import BaseVectorStore, ScoredDocument
from .in_memory import InMemoryVectorStore

__all__ = [
    "BaseVectorStore",
    "ScoredDocument",
    "InMemoryVectorStore",
]
