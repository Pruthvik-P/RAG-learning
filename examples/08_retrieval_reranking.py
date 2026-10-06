"""
LESSON 8 — Visualizing retrieval and reranking.

Two ideas are shown:

1. RETRIEVAL STRATEGIES.  Four retrievers answer the same query and we compare
   who surfaced what:
       similarity (dense) | mmr (diverse) | keyword/BM25 (lexical) | hybrid (RRF)

2. RERANKING.  We retrieve a wide candidate pool, then reorder it with a
   reranker, and draw a before/after diff showing how ranks moved.

Run:
    python examples/08_retrieval_reranking.py
"""

import _bootstrap  # noqa: F401

from rag.config import settings
from rag.chunkers import RecursiveChunker
from rag.embeddings import HashingEmbedding
from rag.loaders import DirectoryLoader
from rag.rerankers import LexicalReranker, get_reranker
from rag.retrievers import (
    SimilarityRetriever,
    MMRRetriever,
    KeywordRetriever,
    HybridRetriever,
)
from rag.vectorstores import InMemoryVectorStore
from rag.visualization import (
    render_rerank,
    render_results,
    render_retriever_comparison,
)

SAMPLE_DIR = settings.data_dir / "sample_docs"


def build_store() -> InMemoryVectorStore:
    documents = DirectoryLoader().load(str(SAMPLE_DIR))
    chunks = RecursiveChunker(chunk_size=400, chunk_overlap=80).split_documents(documents)
    store = InMemoryVectorStore(HashingEmbedding(dimensions=1024))
    store.add_documents(chunks)
    return store


def main() -> None:
    store = build_store()
    query = "What is chunking and why is chunk overlap useful?"

    retrievers = {
        "similarity": SimilarityRetriever(store),
        "mmr": MMRRetriever(store, lambda_mult=0.5),
        "keyword": KeywordRetriever(store),
        "hybrid": HybridRetriever(store),
    }

    # ---- 1. One retriever in detail ------------------------------------ #
    similarity_results = retrievers["similarity"].retrieve(query, k=4)
    print(render_results(query, similarity_results, name="SimilarityRetriever"))

    # ---- 2. All four, side by side ------------------------------------- #
    print()
    print(render_retriever_comparison(query, retrievers, k=3, snippet_width=38))

    print("\nDense search finds paraphrases; keyword search finds exact terms;")
    print("MMR avoids duplicates; hybrid combines dense + lexical rankings.")

    # ---- 3. Reranking the candidate pool -------------------------------- #
    # Retrieve BROADLY (top 8), then rerank NARROWLY (keep top 4).
    candidates = retrievers["similarity"].retrieve(query, k=8)
    reranker = LexicalReranker(lexical_weight=0.7, original_weight=0.3)
    reranked = reranker.rerank(query, candidates, k=4)

    print()
    print(render_rerank(query, candidates, reranked, name=type(reranker).__name__))

    # The factory is what the RAGPipeline uses under the hood.
    assert isinstance(get_reranker("lexical"), LexicalReranker)

    print("\nTakeaway")
    print("--------")
    print("Retrieval is cheap and cast a wide net; reranking is smarter but is")
    print("only run on the small candidate set, so you get accuracy cheaply.")
    print("Enable it in the full pipeline with:")
    print('    RAG_USE_RERANKING=true  (and optionally RAG_RERANK_KIND=cross_encoder)')


if __name__ == "__main__":
    main()
