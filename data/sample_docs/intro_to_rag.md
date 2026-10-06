# Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines a
**retrieval system** with a **large language model (LLM)**. Instead of asking
the LLM to answer from its internal (and possibly outdated or hallucinated)
memory, RAG first *retrieves* relevant passages from an external knowledge base
and then *augments* the prompt with those passages before the model *generates*
an answer.

The acronym spells out its three stages:

1. **Retrieve** — find the most relevant pieces of text for the question.
2. **Augment** — insert those pieces into the prompt as context.
3. **Generate** — let the LLM write an answer grounded in that context.

## Why RAG matters

- **Reduces hallucinations.** The model is told to answer only from the
  provided context, so it is far less likely to invent facts.
- **Fresh knowledge.** Update the knowledge base and the model instantly
  "knows" new information; no retraining required.
- **Source citations.** You can point to the exact document a fact came from.
- **Cheaper than fine-tuning.** No gradient updates or GPUs needed to add data.
- **Private data.** Your documents never need to be baked into model weights.

## The offline (ingestion) pipeline

Before you can retrieve anything you must build an index:

1. **Load** documents from files (PDF, Markdown, text, ...).
2. **Chunk** them into smaller passages.
3. **Embed** each chunk into a numeric vector.
4. **Store** the vectors in a vector database.

## The online (query) pipeline

At question time you run:

1. **Embed the question** with the same embedding model used for chunks.
2. **Search** the vector store for the closest chunk vectors.
3. **Build a prompt** containing the retrieved chunks plus the question.
4. **Call the LLM** to produce a grounded answer.

## Key trade-offs

- **Chunk size:** too small loses context; too large adds noise. A few hundred
  characters with some overlap is a common starting point.
- **Top-k:** retrieving too few chunks may miss the answer; retrieving too many
  floods the prompt with irrelevant text.
- **Embedding quality:** better embeddings mean better retrieval. Neural
  embeddings usually beat keyword-based ones.
