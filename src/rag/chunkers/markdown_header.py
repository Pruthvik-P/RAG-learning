"""
Markdown-header chunker — split along the document's own structure.

WHY USE HEADINGS?
-----------------
Markdown already encodes structure: `#`, `##`, `###` mark sections that mean
something to a human.  Splitting on those headings gives chunks that line up
with the author's own topics, and each chunk can carry a "breadcrumb" of the
headings above it so it still makes sense in isolation:

    [Key trade-offs > Chunk size]
    Chunk size: too small loses context; too large adds noise...

This is usually a *first* split.  A section can still be longer than
`chunk_size`, so we hand it to a recursive splitter as a fallback.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .base import BaseChunker
from .recursive import RecursiveChunker


class MarkdownHeaderChunker(BaseChunker):
    """
    Split markdown text into one chunk per heading section.

    Args:
        chunk_size: Max characters per chunk.  Larger sections are delegated
            to `fallback`.
        chunk_overlap: Passed to the fallback splitter.
        max_level: Ignore headings deeper than this level (default: 6, i.e.
            use every heading).
        fallback: Chunker used when a single section is too long.  Defaults to
            `RecursiveChunker` with the same size/overlap.
    """

    # A markdown ATX heading: one or more '#', a space, then the title.
    _HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        max_level: int = 6,
        fallback: BaseChunker | None = None,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.max_level = max_level
        self.fallback = fallback or RecursiveChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        sections = self._collect_sections(text)

        chunks: List[str] = []
        for path, body in sections:
            # Build a human-readable breadcrumb of the headings above this
            # section, e.g. "Key trade-offs > Chunk size".
            breadcrumb = " > ".join(path)
            header = f"[{breadcrumb}]\n" if breadcrumb else ""
            full = header + body

            if len(full) <= self.chunk_size:
                chunks.append(full)
            else:
                # Too long -> recursive fallback keeps paragraph/sentence
                # boundaries intact.
                chunks.extend(self.fallback.split_text(full))

        return [chunk for chunk in chunks if chunk.strip()]

    # ------------------------------------------------------------------ #
    def _collect_sections(self, text: str) -> List[Tuple[List[str], str]]:
        """
        Walk the text line by line and return (heading_path, body) pairs.

        The heading path is the stack of currently open headings, from the most
        general to the most specific.
        """
        sections: List[Tuple[List[str], str]] = []
        # level -> title for the currently "open" headings.
        open_headings: Dict[int, str] = {}
        buffer: List[str] = []

        def flush() -> None:
            """Emit the buffered body under the current heading path."""
            body = "\n".join(buffer).strip()
            if body:
                path = [open_headings[level] for level in sorted(open_headings)]
                sections.append((path, body))
            buffer.clear()

        for line in text.splitlines():
            match = self._HEADING.match(line.strip())
            if match:
                # A new heading ends the previous section.
                flush()
                level = len(match.group(1))
                title = match.group(2).strip()

                # Deeper-or-equal headings close out when we go up a level.
                for open_level in [lv for lv in open_headings if lv >= level]:
                    del open_headings[open_level]

                if level <= self.max_level:
                    open_headings[level] = title
            else:
                buffer.append(line)

        flush()
        return sections
