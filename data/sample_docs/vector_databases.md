# Vector Databases and Similarity Search

## What is a vector database?

A vector database stores *embeddings* — fixed-length lists of floating point
numbers — and can quickly find the vectors most similar to a query vector.
Unlike a normal database, which matches exact values, a vector database matches
by **semantic closeness**.

## Distance metrics

- **Cosine similarity** measures the angle between two vectors. It ignores
  magnitude and is the most common metric for text embeddings. Values range
  from -1 (opposite) to 1 (identical).
- **Euclidean (L2) distance** measures straight-line distance. Smaller is
  more similar. If vectors are normalized to unit length, L2 and cosine give
  the same ranking.
- **Dot product** is like cosine but also rewards larger magnitudes. Some
  models are trained specifically for dot-product search.

## Exact vs approximate search

- **Exact (brute-force) search** compares the query against every vector.
  It is perfectly accurate but scales linearly: O(n) per query. An in-memory
  list works fine for thousands of vectors.
- **Approximate Nearest Neighbour (ANN)** indexes trade a tiny bit of recall
  for enormous speed. Popular algorithms include HNSW (a graph), IVF
  (clustering), and product quantization (compression).

## Popular options

- **FAISS** — a fast local library from Meta; no server required.
- **Chroma** — developer-friendly, easy to persist and embed.
- **Qdrant** — production-grade, written in Rust, rich filtering.
- **pgvector** — adds vector search to PostgreSQL, great if you already use it.
- **Pinecone / Weaviate** — managed, cloud-hosted services.

## Choosing a store

For learning and small projects, an in-memory or FAISS index is plenty. Reach
for a dedicated server when you need persistence, filtering, horizontal
scaling, or millions of vectors.
