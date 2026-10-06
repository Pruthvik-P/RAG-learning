"""
Plain text (.txt) document loader.

This is the simplest possible loader and a good place to start learning.
A loader's only job is: "read a file -> return a list of Document objects".

Every loader in this project follows the same contract so the rest of the
pipeline can treat all documents identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .base import BaseLoader, Document


class TextLoader(BaseLoader):
    """
    Load a UTF-8 text file into a single Document.

    Example:
        loader = TextLoader(encoding="utf-8")
        docs = loader.load("data/sample_docs/notes.txt")
    """

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding

    def load(self, source: str) -> List[Document]:
        """
        Read the file at `source` and wrap its contents in a Document.

        Args:
            source: Path to a .txt file.

        Returns:
            A list containing exactly one Document.

        Raises:
            FileNotFoundError: if the path does not exist.
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {source}")

        # `errors="ignore"` makes the loader robust against odd bytes instead
        # of crashing the whole ingestion job.
        text = path.read_text(encoding=self.encoding, errors="ignore")

        # Metadata travels WITH the text through the whole pipeline.  Keeping
        # the source path lets us cite where an answer came from later.
        metadata = {
            "source": str(path),
            "file_name": path.name,
            "file_type": "text",
        }

        return [Document(page_content=text, metadata=metadata)]
