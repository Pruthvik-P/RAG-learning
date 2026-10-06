"""
Chunkers package.

A chunker turns long Documents into many smaller Documents.  We provide three
strategies with increasing "semantic awareness":

    FixedSizeChunker  -> cut every N characters (simple, predictable)
    RecursiveChunker  -> prefer paragraph/sentence/word boundaries (default)
    SentenceChunker   -> never break a sentence (clean, grammar-aware)
"""

from .base import BaseChunker
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker
from .sentence import SentenceChunker

__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SentenceChunker",
]
