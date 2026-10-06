"""
LESSON 3 — Embeddings.

An embedding turns text into a vector.  This script shows the key property RAG
depends on: texts with similar MEANING get similar vectors (high cosine
similarity), even when they do not share every word.

Run:
    python examples/03_embeddings.py
"""

import _bootstrap  # noqa: F401

import math

from rag.embeddings import HashingEmbedding


def cosine(a, b):
    """Cosine similarity: 1.0 = identical direction, 0.0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    # The default embedder is pure-Python and offline.  It captures lexical
    # (keyword) similarity.  Swap in SentenceTransformerEmbedding for deeper
    # semantic matching once you install sentence-transformers.
    embedder = HashingEmbedding(dimensions=1024, ngram_range=(1, 2))

    print("=" * 70)
    print("LESSON 3: Embeddings")
    print("=" * 70)

    query = "What are vector databases used for?"
    candidates = [
        "A vector database stores embeddings and finds similar vectors.",  # close
        "Vector stores power semantic search over documents.",            # close-ish
        "The weather today is sunny with a light breeze.",                # unrelated
        "Chunking splits long documents into smaller passages.",          # related topic
    ]

    print(f"\nDimensions per vector: {embedder.dimensions}")
    print(f"Query: {query!r}\n")

    query_vector = embedder.embed_query(query)
    print(f"{'similarity':>10}  candidate")
    print("-" * 70)
    scored = []
    for text in candidates:
        score = cosine(query_vector, embedder.embed_documents([text])[0])
        scored.append((score, text))
    for score, text in sorted(scored, reverse=True):
        print(f"{score:>10.3f}  {text}")

    print("\nThe most relevant sentence scores highest, the unrelated one lowest.")
    print("In a vector store we simply sort by this score and return the top-k.")
    print("\nNote: the hashing embedder matches on WORDS. Neural embedders")
    print("(e.g. all-MiniLM-L6-v2) also match on MEANING, so synonyms and")
    print("paraphrases score higher. Same idea, better vectors.")


if __name__ == "__main__":
    main()
