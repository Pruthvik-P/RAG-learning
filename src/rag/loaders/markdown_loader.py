"""
Markdown (.md / .markdown) document loader.

Markdown is often the best format for a RAG knowledge base because the
heading structure carries semantic meaning.  This loader reads the file and
optionally records the heading hierarchy in metadata, which can later be used
to build better prompts or smarter filtering.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .base import BaseLoader, Document


class MarkdownLoader(BaseLoader):
    """
    Load a Markdown file into a single Document.

    Args:
        strip_front_matter: Remove a leading YAML front-matter block
            (the `--- ... ---` section) if present.
        keep_headings: When True, headings stay in the text (they are useful
            context for the LLM).  Set to False to remove them.
    """

    # Matches a leading YAML front-matter block at the very start of the file.
    _FRONT_MATTER = re.compile(r"^\s*---\s*\n.*?\n---\s*\n", re.DOTALL)
    # Matches a single markdown heading line, capturing its level and text.
    _HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

    def __init__(
        self,
        strip_front_matter: bool = True,
        keep_headings: bool = True,
        encoding: str = "utf-8",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strip_front_matter = strip_front_matter
        self.keep_headings = keep_headings
        self.encoding = encoding

    def load(self, source: str) -> List[Document]:
        """Read a Markdown file and return a single Document."""
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {source}")

        text = path.read_text(encoding=self.encoding, errors="ignore")

        # Optionally drop the metadata block that many static-site generators
        # put at the top of markdown files.
        if self.strip_front_matter:
            text = self._FRONT_MATTER.sub("", text)

        headings = self._extract_headings(text)

        # If the caller does not want headings in the body, remove the `#`.
        if not self.keep_headings:
            lines = []
            for line in text.splitlines():
                match = self._HEADING.match(line)
                lines.append(match.group(2) if match else line)
            text = "\n".join(lines)

        metadata = {
            "source": str(path),
            "file_name": path.name,
            "file_type": "markdown",
            # A flat list of headings gives the LLM a table of contents.
            "headings": headings,
        }

        return [Document(page_content=text, metadata=metadata)]

    def _extract_headings(self, text: str) -> List[str]:
        """Return all heading lines (without the leading '#') in order."""
        headings = []
        for line in text.splitlines():
            match = self._HEADING.match(line.strip())
            if match:
                headings.append(match.group(2).strip())
        return headings
