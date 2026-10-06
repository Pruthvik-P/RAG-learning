# RAG Learning — Build Retrieval-Augmented Generation From Scratch

A readable, heavily-commented RAG implementation you can learn from. The **core
pipeline runs on the Python standard library only** (no heavy installs), and it
uses **DeepSeek** for generation. Every stage of RAG lives in its own folder and
has a matching hands-on lesson under `examples/`.

```
LOAD  ->  CHUNK  ->  EMBED  ->  STORE      (ingestion, run once)
RETRIEVE  ->  AUGMENT  ->  GENERATE        (query, run per question)
```

---

## 1. What is RAG? (the 60-second version)

An LLM answers from its internal memory, which can be outdated or wrong. **RAG**
first *retrieves* relevant passages from your own documents, *augments* the
prompt with them, then lets the LLM *generate* an answer grounded in that
context. Benefits: fewer hallucinations, fresh knowledge, source citations, and
no fine-tuning.

Read `data/sample_docs/` — the knowledge base is about RAG itself, so you can
literally ask the system "What is RAG?".

---

## 2. Quick start

```powershell
# 1. (optional) real answers: copy env template and add your DeepSeek key
Copy-Item .env.example .env
#    then edit .env -> DEEPSEEK_API_KEY=sk-...

# 2. Run the lessons (no installation required)
python examples/01_loading.py
python examples/02_chunking.py
python examples/03_embeddings.py
python examples/04_vector_store.py
python examples/05_retrieval.py
python examples/06_full_rag.py "What is retrieval-augmented generation?"
python examples/07_chunking_visualization.py
python examples/08_retrieval_reranking.py

# 3. Run the tests
python -m unittest discover -s tests -v
```

> No API key? Everything still runs — the pipeline falls back to a `DummyLLM`
> that shows you the exact prompt that would have been sent to DeepSeek.

---

## 3. Project layout

```
RAG-learing/
├── .env.example              # config template (API keys, chunk size, top_k...)
├── pyproject.toml            # package metadata (optional editable install)
├── requirements.txt          # optional extras (all commented out)
├── data/
│   └── sample_docs/          # sample knowledge base (.md / .txt) about RAG
├── examples/                 # numbered, runnable lessons 01..06
├── tests/                    # stdlib unittest suite
└── src/rag/
    ├── config.py             # Settings + tiny .env loader
    ├── loaders/              # files        -> Document
    │   ├── base.py           # Document + BaseLoader (the shared contract)
    │   ├── text_loader.py
    │   ├── markdown_loader.py
    │   ├── pdf_loader.py          # optional (needs pypdf)
    │   └── directory_loader.py    # dispatch by file extension
    ├── chunkers/             # Document     -> smaller Documents
    │   ├── fixed_size.py
    │   ├── token.py          # split every N words/tokens
    │   ├── recursive.py      # the sensible default
    │   ├── sentence.py
    │   └── markdown_header.py # follow the '#' heading structure
    ├── embeddings/           # text         -> vector
    │   ├── hashing.py        # offline, dependency-free default
    │   ├── sentence_transformer.py  # optional, real neural embeddings
    │   └── openai_compatible.py     # optional, hosted embeddings
    ├── vectorstores/         # vector + Document -> searchable index
    │   └── in_memory.py      # cosine-similarity search + save/load
    ├── retrievers/           # query        -> relevant Documents
    │   ├── similarity.py     # dense vector search
    │   ├── mmr.py            # relevance + diversity
    │   ├── keyword.py        # BM25 lexical / exact-term search
    │   └── hybrid.py         # reciprocal-rank fusion of dense + keyword
    ├── rerankers/            # candidates   -> reordered candidates
    │   ├── lexical.py        # offline term-overlap reranker (default)
    │   └── cross_encoder.py  # optional neural reranker
    ├── visualization/        # terminal ASCII views of every stage
    │   ├── bars.py           # shared bar/span drawing helpers
    │   ├── chunking.py       # chunk summaries + span/overlap maps
    │   └── retrieval.py      # result charts, comparisons, rerank diffs
    ├── llms/                 # prompt       -> answer
    │   ├── deepseek.py       # DeepSeek chat (OpenAI-compatible HTTP)
    │   └── dummy.py          # offline placeholder
    └── pipelines/
        ├── ingestion.py      # LOAD -> CHUNK -> EMBED -> STORE
        └── rag_pipeline.py   # RETRIEVE -> (RERANK) -> AUGMENT -> GENERATE
```

---

## 4. The learning path

| Lesson | File | Concept |
|-------:|------|---------|
| 1 | `examples/01_loading.py` | Load files into `Document` (text + metadata) |
| 2 | `examples/02_chunking.py` | Split documents; compare chunking strategies |
| 3 | `examples/03_embeddings.py` | Text -> vectors; cosine similarity |
| 4 | `examples/04_vector_store.py` | Index, search, save/load a vector store |
| 5 | `examples/05_retrieval.py` | Similarity vs MMR retrieval |
| 6 | `examples/06_full_rag.py` | Full pipeline with DeepSeek + citations |
| 7 | `examples/07_chunking_visualization.py` | Visualize 5 chunkers (summaries + span maps) |
| 8 | `examples/08_retrieval_reranking.py` | Compare 4 retrievers and visualize reranking |

---

## 5. Using the library

```python
from rag import RAGPipeline

# Ingest a folder and get a query-ready pipeline in one call.
rag = RAGPipeline.from_directory("data/sample_docs")

result = rag.answer("How does cosine similarity work?")
print(result.answer)

for doc, score in result.sources:
    print(score, doc.metadata["source"])
```

Compose your own stages when experimenting:

```python
from rag import (
    DirectoryLoader, RecursiveChunker, HashingEmbedding,
    InMemoryVectorStore, SimilarityRetriever, DeepSeekLLM, RAGPipeline,
)

documents = DirectoryLoader().load("data/sample_docs")
chunks = RecursiveChunker(chunk_size=400, chunk_overlap=80).split_documents(documents)
store = InMemoryVectorStore(HashingEmbedding(dimensions=1024))
store.add_documents(chunks)

rag = RAGPipeline(store, llm=DeepSeekLLM(model="deepseek-chat"))
print(rag.ask("What are the stages of RAG?").answer)
```

---

## 6. Chunking, retrieval & reranking (and how to see them)

Every stage is pluggable, and each strategy can be compared side by side with
terminal visualizations (no plotting library needed).

**Chunking** — five strategies in `rag.chunkers`: `FixedSizeChunker`,
`TokenChunker`, `RecursiveChunker`, `SentenceChunker`, `MarkdownHeaderChunker`.

```python
from rag import MarkdownHeaderChunker, RecursiveChunker
from rag.visualization import render_chunking_report

report = render_chunking_report(document, {
    "recursive": RecursiveChunker(chunk_size=300, chunk_overlap=60),
    "by-heading": MarkdownHeaderChunker(chunk_size=300, chunk_overlap=60),
})
print(report)   # summary table + a span map showing each chunk & its overlap
```

**Retrieval** — four strategies in `rag.retrievers`: `SimilarityRetriever`
(dense), `MMRRetriever` (diverse), `KeywordRetriever` (BM25), `HybridRetriever`
(RRF fusion of dense + keyword).

```python
from rag.visualization import render_retriever_comparison
print(render_retriever_comparison(query, {
    "similarity": SimilarityRetriever(store),
    "keyword": KeywordRetriever(store),
    "hybrid": HybridRetriever(store),
}))
```

**Reranking** — retrieve broadly, then rerank narrowly
(`rag.rerankers.LexicalReranker` offline, or `CrossEncoderReranker` with the
optional `sentence-transformers` package).

```python
from rag.rerankers import get_reranker
from rag.visualization import render_rerank

candidates = SimilarityRetriever(store).retrieve(query, k=8)
reranked = get_reranker("lexical").rerank(query, candidates, k=4)
print(render_rerank(query, candidates, reranked))
```

Enable reranking inside the full pipeline with settings (or pass your own
`reranker`):

```python
from rag import RAGPipeline, Settings, LexicalReranker

settings = Settings(use_reranking=True, rerank_kind="lexical", top_k=4)
rag = RAGPipeline.from_directory("data/sample_docs", settings=settings)
```

---

## 7. Configuration

Settings come from environment variables or a `.env` file (see `.env.example`).
Key knobs:

| Variable | Default | Meaning |
|----------|---------|---------|
| `DEEPSEEK_API_KEY` | – | Your DeepSeek key |
| `DEEPSEEK_MODEL` | `deepseek-chat` | `deepseek-chat` or `deepseek-reasoner` |
| `RAG_CHUNK_SIZE` | `800` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `120` | Shared characters between chunks |
| `RAG_TOP_K` | `4` | Chunks retrieved per question |
| `RAG_RETRIEVAL_MODE` | `similarity` | `similarity`, `mmr`, `keyword`, or `hybrid` |
| `RAG_USE_RERANKING` | `false` | Enable a reranking pass over the candidates |
| `RAG_RERANK_KIND` | `lexical` | `lexical` or `cross_encoder` |
| `RAG_RERANK_CANDIDATES` | `20` | Candidates fetched before reranking |
| `RAG_EMBEDDING_PROVIDER` | `hashing` | `hashing`, `sentence_transformer`, `openai` |

---

## 8. Upgrading to real embeddings (optional)

The default `HashingEmbedding` matches on words so the project runs with zero
dependencies. For **semantic** matching, install and switch:

```powershell
pip install sentence-transformers
$env:RAG_EMBEDDING_PROVIDER = "sentence_transformer"
python examples/06_full_rag.py
```

Everything else stays the same — that is the point of the interfaces.

---

## 9. Extending the system

- **New file format:** subclass `BaseLoader`, add it to `DirectoryLoader.loader_map`.
- **New chunker:** subclass `BaseChunker`, implement `split_text`.
- **New embedding:** subclass `BaseEmbedding`, implement `embed_documents`.
- **New vector DB:** subclass `BaseVectorStore` (e.g. wrap FAISS or Chroma).
- **New retrieval strategy:** subclass `BaseRetriever` (e.g. hybrid BM25 + vector).
- **New reranker:** subclass `BaseReranker`, implement `rerank` (e.g. an LLM).
- **New visualization:** add a renderer under `rag.visualization`.
- **New LLM:** subclass `BaseLLM`, implement `chat` (any OpenAI-compatible API).
