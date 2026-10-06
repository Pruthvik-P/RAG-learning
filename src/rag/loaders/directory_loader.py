"""
Directory loader.

Real knowledge bases are usually a FOLDER of mixed files, not a single file.
This loader walks a directory, picks the right loader for each file extension,
and concatenates all resulting Documents.

This is the "dispatcher" pattern: one entry point, many format-specific loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type

from .base import BaseLoader, Document
from .markdown_loader import MarkdownLoader
from .pdf_loader import PDFLoader
from .text_loader import TextLoader


class DirectoryLoader(BaseLoader):
    """
    Recursively load every supported file inside a directory.

    Args:
        glob: File pattern to match (default: everything).
        recursive: Walk sub-directories too.
        loader_map: Which loader to use per extension.  Override this to plug
            in your own loaders, e.g. {".csv": CSVLoader()}.
        silent_errors: If True, a broken file is skipped instead of crashing
            the whole batch (useful for large, messy folders).
    """

    def __init__(
        self,
        glob: str = "**/*",
        recursive: bool = True,
        loader_map: Optional[Dict[str, BaseLoader]] = None,
        silent_errors: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.glob = glob
        self.recursive = recursive
        self.silent_errors = silent_errors

        # Default extension -> loader wiring.  Add entries here to support
        # more formats (e.g. ".csv": CSVLoader()).
        self.loader_map: Dict[str, BaseLoader] = loader_map or {
            ".txt": TextLoader(),
            ".md": MarkdownLoader(),
            ".markdown": MarkdownLoader(),
            ".pdf": PDFLoader(),
        }

    def load(self, source: str) -> List[Document]:
        """Load every supported file under `source`."""
        root = Path(source)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {source}")
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {source}")

        pattern = self.glob if self.recursive else "*"
        documents: List[Document] = []

        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue

            extension = path.suffix.lower()
            loader = self.loader_map.get(extension)
            if loader is None:
                # Unknown extension -> ignore politely.
                continue

            try:
                documents.extend(loader.load(str(path)))
            except Exception as exc:  # noqa: BLE001 - deliberate broad catch
                if self.silent_errors:
                    print(f"[DirectoryLoader] skipped {path}: {exc}")
                else:
                    raise

        return documents


def get_default_loader_map() -> Dict[str, BaseLoader]:
    """Expose the default map for callers that want to extend it."""
    return {
        ".txt": TextLoader(),
        ".md": MarkdownLoader(),
        ".markdown": MarkdownLoader(),
        ".pdf": PDFLoader(),
    }
