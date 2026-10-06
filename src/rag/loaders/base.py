"""
Base Classes for Document Loading

This module defines the fundamental data structures and abstract base class
that all document loaders must implement.

Key Concepts:
- Document: A container for text content and associated metadata
- BaseLoader: Abstract base class defining the interface for all loaders
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class Document:
    """
    Represents a loaded document with its content and metadata.
    
    This is the standard data structure used throughout the RAG pipeline.
    All loaders return a list of Document objects.
    
    Attributes:
        page_content: The actual text content of the document
        metadata: Dictionary containing additional information about the document
                  Common keys: source, page, author, created_at, etc.
        id: Unique identifier for the document (auto-generated if not provided)
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if self.id is None:
            import hashlib
            # Create a deterministic ID based on content and source
            content_hash = hashlib.md5(
                f"{self.page_content}{self.metadata.get('source', '')}".encode()
            ).hexdigest()[:12]
            self.id = f"doc_{content_hash}"
    
    def __len__(self) -> int:
        """Return the length of the document content."""
        return len(self.page_content)
    
    def __str__(self) -> str:
        """Human-readable representation."""
        source = self.metadata.get('source', 'unknown')
        preview = self.page_content[:100] + "..." if len(self.page_content) > 100 else self.page_content
        return f"Document(source={source}, length={len(self)}, preview='{preview}')"


class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.
    
    All document loaders must inherit from this class and implement
    the load() method. This ensures a consistent interface across
    different file formats.
    
    Usage:
        class MyCustomLoader(BaseLoader):
            def load(self, source: str) -> List[Document]:
                # Implementation here
                pass
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the loader with optional configuration.
        
        Args:
            **kwargs: Configuration options specific to each loader
        """
        self.config = kwargs
    
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """
        Load documents from the given source.
        
        This is the main method that must be implemented by all subclasses.
        
        Args:
            source: Path to the file or directory to load
            
        Returns:
            List of Document objects containing the loaded content
            
        Raises:
            FileNotFoundError: If the source doesn't exist
            ValueError: If the source format is not supported
        """
        pass
    
    def load_from_path(self, path: Path) -> List[Document]:
        """
        Convenience method to load from a Path object.
        
        Args:
            path: Path object pointing to the file/directory
            
        Returns:
            List of Document objects
        """
        return self.load(str(path))
    
    def validate_source(self, source: str) -> bool:
        """
        Validate that the source exists and is readable.
        
        Args:
            source: Path to validate
            
        Returns:
            True if valid, False otherwise
        """
        path = Path(source)
        return path.exists() and path.is_file()


# Utility function to create documents from raw text
def create_documents_from_texts(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None
) -> List[Document]:
    """
    Helper function to create Document objects from raw text strings.
    
    Useful for testing or when you already have text in memory.
    
    Args:
        texts: List of text strings
        metadatas: Optional list of metadata dicts (one per text)
        
    Returns:
        List of Document objects
    """
    if metadatas is None:
        metadatas = [{} for _ in texts]
    
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadatas)
    ]