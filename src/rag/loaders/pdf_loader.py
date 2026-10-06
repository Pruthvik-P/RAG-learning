"""
PDF loader (optional dependency).

PDFs need a parsing library.  To keep the core project dependency-free we
import `pypdf` LAZILY (inside the method) and raise a clear, actionable error
if it is not installed.

Install it with:
    pip install pypdf

Learning note: a good RAG PDF loader returns one Document PER PAGE.  Page-level
metadata lets you cite "page 7" in the final answer and enables filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .base import BaseLoader, Document


class PDFLoader(BaseLoader):
    """
    Load a PDF file, one Document per page.

    Args:
        password: Optional password for encrypted PDFs.
        extract_metadata: Copy PDF metadata (title, author...) into the
            Document metadata when available.
    """

    def __init__(
        self,
        password: Optional[str] = None,
        extract_metadata: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.password = password
        self.extract_metadata = extract_metadata

    def load(self, source: str) -> List[Document]:
        """Read a PDF and return a list of Documents, one per page."""
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")

        try:
            from pypdf import PdfReader  # imported here on purpose (optional dep)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "PDFLoader requires the optional 'pypdf' package.\n"
                "Install it with:  pip install pypdf"
            ) from exc

        reader = PdfReader(str(path))
        if self.password:
            reader.decrypt(self.password)

        # Pull document-level metadata once.
        pdf_meta = {}
        if self.extract_metadata and reader.metadata:
            for key in ("title", "author", "subject", "creator"):
                value = getattr(reader.metadata, key, None)
                if value:
                    pdf_meta[key] = str(value)

        documents: List[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                # Skip blank pages so we don't embed noise.
                continue
            metadata = {
                "source": str(path),
                "file_name": path.name,
                "file_type": "pdf",
                "page": page_number,
                "total_pages": len(reader.pages),
                **pdf_meta,
            }
            documents.append(Document(page_content=text, metadata=metadata))

        return documents
