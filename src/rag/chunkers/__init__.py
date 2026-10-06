"""
Chunkers package.

A chunker turns long Documents into many smaller Documents.  We provide five
strategies with increasing "semantic awareness":

    FixedSizeChunker      -> cut every N characters (simple, predictable)
    TokenChunker          -> cut every N tokens/words (matches model limits)
    RecursiveChunker      -> prefer paragraph/sentence/word boundaries (default)
    SentenceChunker       -> never break a sentence (clean, grammar-aware)
    MarkdownHeaderChunker -> follow the document's own '#' headings
"""

from .base import BaseChunker
from .fixed_size import FixedSizeChunker
from .markdown_header import MarkdownHeaderChunker
from .recursive import RecursiveChunker
from .sentence import SentenceChunker
from .token import TokenChunker

__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "TokenChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "MarkdownHeaderChunker",
]
