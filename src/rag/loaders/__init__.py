"""
Document Loaders Package

This package contains various document loaders for RAG systems.
Document loaders are responsible for reading different file formats
and converting them into a standardized format for processing.

Key Concepts:
- Document: A standardized representation of loaded content with text and metadata
- Loader: A class that knows how to read a specific file format
- Metadata: Additional information about the document (source, page number, etc.)
"""

from .base import Document, BaseLoader, create_documents_from_texts
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .markdown_loader import MarkdownLoader
from .directory_loader import DirectoryLoader, get_default_loader_map

__all__ = [
    "Document",
    "BaseLoader",
    "create_documents_from_texts",
    "TextLoader",
    "PDFLoader",
    "MarkdownLoader",
    "DirectoryLoader",
    "get_default_loader_map",
]