"""
RAG learning package.

A from-scratch, dependency-free (core) Retrieval-Augmented Generation toolkit
built to be READ.  The pipeline stages map onto folders:

    loaders/       files  -> Document
    chunkers/      Document -> smaller Documents
    embeddings/    text   -> vector
    vectorstores/  vector + Document -> searchable index
    retrievers/    query  -> relevant Documents
    llms/          prompt -> answer   (DeepSeek)
    pipelines/     ties it all together

Quick start:
    from rag import RAGPipeline
    rag = RAGPipeline.from_directory("data/sample_docs")
    print(rag.answer("What is retrieval-augmented generation?").answer)
"""

from .config import Settings, settings
from .loaders import Document, TextLoader, MarkdownLoader, PDFLoader, DirectoryLoader
from .chunkers import FixedSizeChunker, RecursiveChunker, SentenceChunker
from .embeddings import HashingEmbedding, get_embedding
from .vectorstores import InMemoryVectorStore
from .retrievers import SimilarityRetriever, MMRRetriever, get_retriever
from .llms import DeepSeekLLM, DummyLLM, get_llm
from .pipelines import IngestionPipeline, RAGPipeline, RAGResponse

__version__ = "0.1.0"

__all__ = [
    # config
    "Settings",
    "settings",
    # loaders
    "Document",
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "DirectoryLoader",
    # chunkers
    "FixedSizeChunker",
    "RecursiveChunker",
    "SentenceChunker",
    # embeddings
    "HashingEmbedding",
    "get_embedding",
    # vector stores
    "InMemoryVectorStore",
    # retrievers
    "SimilarityRetriever",
    "MMRRetriever",
    "get_retriever",
    # llms
    "DeepSeekLLM",
    "DummyLLM",
    "get_llm",
    # pipelines
    "IngestionPipeline",
    "RAGPipeline",
    "RAGResponse",
]
