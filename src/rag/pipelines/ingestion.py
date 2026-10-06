"""
Ingestion pipeline: the "write side" of RAG.

This is the offline job you run once (or whenever documents change).  It walks
the four stages of preparing a knowledge base:

        LOAD  ->  CHUNK  ->  EMBED  ->  STORE
        (files)   (pieces)   (vectors)  (searchable index)

After ingestion you have a vector store ready to answer queries.  The "read
side" (retrieval + generation) lives in rag_pipeline.py.

Usage:
    pipeline = IngestionPipeline()
    store = pipeline.ingest("data/sample_docs")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..chunkers.base import BaseChunker
from ..chunkers.recursive import RecursiveChunker
from ..config import Settings, settings as default_settings
from ..embeddings.base import BaseEmbedding
from ..embeddings import get_embedding
from ..loaders.base import BaseLoader, Document
from ..loaders.directory_loader import DirectoryLoader
from ..vectorstores.base import BaseVectorStore
from ..vectorstores.in_memory import InMemoryVectorStore


@dataclass
class IngestionStats:
    """Small report so learners can see what each stage produced."""

    documents_loaded: int = 0
    chunks_created: int = 0
    chunks_stored: int = 0
    sources: List[str] = field(default_factory=list)


class IngestionPipeline:
    """
    Composable ingestion pipeline.

    Every stage is injectable: pass your own loader/chunker/embedding/store, or
    let the constructor build sensible defaults from `Settings`.
    """

    def __init__(
        self,
        loader: Optional[BaseLoader] = None,
        chunker: Optional[BaseChunker] = None,
        embedding: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
        settings: Settings = default_settings,
    ):
        self.settings = settings
        self.loader = loader or DirectoryLoader()
        self.chunker = chunker or RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embedding = embedding or get_embedding(
            provider=settings.embedding_provider,
            dimensions=settings.embedding_dimensions,
            model=settings.embedding_model,
        )
        self.vector_store = vector_store or InMemoryVectorStore(self.embedding)

    # ------------------------------------------------------------------ #
    def ingest(self, source: str, verbose: bool = True) -> BaseVectorStore:
        """
        Run LOAD -> CHUNK -> EMBED -> STORE for everything under `source`.

        Returns the populated vector store (also available as self.vector_store).
        """
        stats = IngestionStats()

        # 1) LOAD: turn files into Documents.
        documents = self.loader.load(source)
        stats.documents_loaded = len(documents)
        stats.sources = sorted({d.metadata.get("source", "?") for d in documents})
        if verbose:
            print(f"[ingest] loaded {len(documents)} document(s)")

        # 2) CHUNK: split long Documents into retrievable pieces.
        chunks = self.chunker.split_documents(documents)
        stats.chunks_created = len(chunks)
        if verbose:
            print(f"[ingest] created {len(chunks)} chunk(s)")

        # 3) + 4) EMBED & STORE (the store embeds internally).
        self.vector_store.add_documents(chunks)
        stats.chunks_stored = len(self.vector_store)
        if verbose:
            print(
                f"[ingest] stored {stats.chunks_stored} chunk(s) "
                f"in the vector store"
            )

        self.stats = stats
        return self.vector_store

    def ingest_documents(self, documents: List[Document]) -> BaseVectorStore:
        """Ingest in-memory Documents (skip the loader stage)."""
        chunks = self.chunker.split_documents(documents)
        self.vector_store.add_documents(chunks)
        return self.vector_store
