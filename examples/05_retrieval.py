"""
LESSON 5 — Retrievers: similarity vs MMR.

The retriever is the strategy layer on top of the vector store.  We compare:

  * SimilarityRetriever — the top-k closest chunks (may be repetitive).
  * MMRRetriever        — relevant AND diverse chunks (less redundancy).

Run:
    python examples/05_retrieval.py
"""

import _bootstrap  # noqa: F401

from pathlib import Path

from rag.config import settings
from rag.embeddings import HashingEmbedding
from rag.loaders import DirectoryLoader
from rag.chunkers import RecursiveChunker
from rag.vectorstores import InMemoryVectorStore
from rag.retrievers import SimilarityRetriever, MMRRetriever

SAMPLE_DIR = settings.data_dir / "sample_docs"


def show(title: str, results) -> None:
    print(f"\n{title}")
    print("-" * 70)
    for rank, (doc, score) in enumerate(results, 1):
        source = Path(doc.metadata["source"]).name
        snippet = doc.page_content.replace("\n", " ")[:70]
        print(f"  {rank}. score={score:.3f} [{source}] {snippet!r} ...")


def main() -> None:
    documents = DirectoryLoader().load(str(SAMPLE_DIR))
    chunks = RecursiveChunker(chunk_size=400, chunk_overlap=80).split_documents(documents)
    store = InMemoryVectorStore(HashingEmbedding(dimensions=1024))
    store.add_documents(chunks)

    print("=" * 70)
    print("LESSON 5: Retrieval strategies")
    print("=" * 70)
    print(f"Store size: {len(store)} chunks")

    query = "What is chunking and why is chunk overlap useful?"

    similarity = SimilarityRetriever(store).retrieve(query, k=3)
    show("SimilarityRetriever (pure relevance)", similarity)

    mmr = MMRRetriever(store, lambda_mult=0.5).retrieve(query, k=3)
    show("MMRRetriever (relevance + diversity, lambda=0.5)", mmr)

    mmr_diverse = MMRRetriever(store, lambda_mult=0.0).retrieve(query, k=3)
    show("MMRRetriever (pure diversity, lambda=0.0)", mmr_diverse)

    print("\nTakeaway: lambda controls the relevance/diversity trade-off.")
    print("Higher lambda -> more like plain similarity.")
    print("Lower lambda  -> results pulled from more varied sources.")


if __name__ == "__main__":
    main()
