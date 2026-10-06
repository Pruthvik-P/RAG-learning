"""
Pipelines package.

* IngestionPipeline : write side  (LOAD -> CHUNK -> EMBED -> STORE)
* RAGPipeline       : read side   (RETRIEVE -> AUGMENT -> GENERATE)
"""

from .ingestion import IngestionPipeline, IngestionStats
from .rag_pipeline import RAGPipeline, RAGResponse

__all__ = [
    "IngestionPipeline",
    "IngestionStats",
    "RAGPipeline",
    "RAGResponse",
]
