"""
LESSON 4 — Building and searching a vector store.

A vector store is just a searchable collection of (vector, document) pairs.
We load + chunk + embed the sample docs, run a few searches, then persist the
index to disk and reload it.

Run:
    python examples/04_vector_store.py
"""

import _bootstrap  # noqa: F401

from pathlib import Path

from rag.config import settings
from rag.embeddings import HashingEmbedding
from rag.loaders import DirectoryLoader
from rag.chunkers import RecursiveChunker
from rag.vectorstores import InMemoryVectorStore

SAMPLE_DIR = settings.data_dir / "sample_docs"
INDEX_FILE = settings.project_root / "artifacts" / "sample.index.json"


def build_store() -> InMemoryVectorStore:
    """LOAD -> CHUNK -> EMBED -> STORE, spelled out one step at a time."""
    documents = DirectoryLoader().load(str(SAMPLE_DIR))
    chunks = RecursiveChunker(chunk_size=400, chunk_overlap=80).split_documents(documents)

    store = InMemoryVectorStore(HashingEmbedding(dimensions=1024))
    store.add_documents(chunks)
    return store


def main() -> None:
    print("=" * 70)
    print("LESSON 4: Vector store")
    print("=" * 70)

    store = build_store()
    print(f"\nIndexed {len(store)} chunks from {SAMPLE_DIR.name}")

    # ---- Search ---------------------------------------------------------- #
    query = "How does cosine similarity work?"
    print(f"\nSearch: {query!r}")
    for rank, (doc, score) in enumerate(store.similarity_search(query, k=3), 1):
        source = Path(doc.metadata["source"]).name
        snippet = doc.page_content.replace("\n", " ")[:80]
        print(f"  {rank}. score={score:.3f} [{source}] {snippet!r} ...")

    # ---- Add more text at runtime --------------------------------------- #
    store.add_texts(
        ["RAG stands for Retrieval-Augmented Generation."],
        [{"source": "inline-note"}],
    )
    print(f"\nAfter adding one inline text, store now has {len(store)} chunks.")

    # ---- Persist and reload --------------------------------------------- #
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    store.save(str(INDEX_FILE))
    print(f"Saved index to {INDEX_FILE.relative_to(settings.project_root)}")

    reloaded = InMemoryVectorStore.load(str(INDEX_FILE), HashingEmbedding(dimensions=1024))
    print(f"Reloaded index has {len(reloaded)} chunks.")


if __name__ == "__main__":
    main()
