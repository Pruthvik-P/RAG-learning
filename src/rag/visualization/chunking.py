"""
Visualize how different chunkers cut up a document.

Two complementary views are provided:

1. A SUMMARY TABLE — how many chunks each strategy produced, their sizes, and
   how much text was emitted (overlap makes this exceed the document length).

2. A SPAN MAP — a fixed-width timeline per chunk showing where it sits in the
   document.  Overlapping chunks visibly overlap on the map, which is the
   quickest way to *see* what `chunk_overlap` does.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..chunkers.base import BaseChunker
from ..loaders.base import Document
from .bars import bar, rule, span_bar, title, truncate

# A span is (start_char, end_char); None means "could not locate in the text".
Span = Optional[Tuple[int, int]]

# MarkdownHeaderChunker prepends a "[Heading > Subheading]" breadcrumb line.
# Strip it before searching so the body still maps onto the source text.
_BREADCRUMB = re.compile(r"^\[[^\]]*\]\n")


def _document_label(document: Document) -> str:
    """A short human name for a document (file name or source)."""
    source = document.metadata.get("source") or document.metadata.get("file_name")
    return Path(source).name if source else "<memory>"


def summarize_chunks(document: Document, chunks: Sequence[Document]) -> Dict[str, float]:
    """Compute headline numbers for one chunking run."""
    sizes = [len(chunk) for chunk in chunks]
    document_len = max(len(document), 1)
    total = sum(sizes)
    return {
        "chunks": len(chunks),
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
        "avg": (total / len(sizes)) if sizes else 0.0,
        "emitted": total,
        # Can exceed 100% because overlapping text is counted in two chunks.
        "coverage": total / document_len,
    }


def _locate_all(text: str, chunk_texts: Sequence[str]) -> Tuple[List[Span], int]:
    """
    Find the character span of every chunk inside the document.

    Fixed/recursive chunks are exact substrings, so we search the raw text.
    Chunkers that normalise whitespace (sentence/token) produce chunks that only
    match a whitespace-collapsed version, so we fall back to that for ALL
    chunks to keep every offset in one coordinate space.
    """
    # Drop any breadcrumb prefix before locating the body.
    bodies = [_BREADCRUMB.sub("", chunk, count=1) for chunk in chunk_texts]

    def all_findable(haystack: str, needles: Sequence[str]) -> bool:
        return all(haystack.find(needle.strip()) != -1 for needle in needles if needle.strip())

    if all_findable(text, bodies):
        source = text
        needles = bodies
    else:
        source = re.sub(r"\s+", " ", text)
        needles = [re.sub(r"\s+", " ", body).strip() for body in bodies]

    spans: List[Span] = []
    cursor = 0
    for raw_needle in needles:
        needle = raw_needle.strip()
        if not needle:
            spans.append(None)
            continue
        index = source.find(needle, cursor)
        if index == -1:
            index = source.find(needle)
        if index == -1:
            spans.append(None)
        else:
            spans.append((index, index + len(needle)))
            # Allow the next chunk to start inside this one (overlap).
            cursor = index
    return spans, len(source)


def render_chunk_spans(
    document: Document,
    chunks: Sequence[Document],
    width: int = 64,
    name: str = "chunks",
) -> str:
    """Render the timeline view for a single chunker's output."""
    chunk_texts = [chunk.page_content for chunk in chunks]
    spans, source_len = _locate_all(document.page_content, chunk_texts)
    scale = source_len / width if width else 1.0

    lines = [
        f"{name}  ({len(chunks)} chunks, {source_len} chars, 1 col ~ {scale:.1f} chars)",
        "   #   chars              size  |" + "-" * width + "|",
    ]

    for index, (chunk, span) in enumerate(zip(chunks, spans)):
        if span is None:
            location = f"{'?':^12}"
            drawing = "." * width
        else:
            location = f"{span[0]:>5}..{span[1]:>5}"
            drawing = span_bar(span[0], span[1], source_len, width)
        lines.append(
            f"  {index:>2}  [{location}] {len(chunk):>5}  |{drawing}|"
        )

    return "\n".join(lines)


def render_chunking_report(
    document: Document,
    chunkers: Dict[str, BaseChunker],
    width: int = 64,
) -> str:
    """
    Full side-by-side report: summary table + a span map per strategy.

    Args:
        document: The source Document (its text is what gets split).
        chunkers: Mapping of strategy name -> chunker instance.
        width: Timeline columns used by each span map.
    """
    lines: List[str] = [
        title(
            "CHUNKING STRATEGIES  "
            f"(document: {_document_label(document)}, {len(document)} chars, "
            f"{len(chunkers)} strategies)"
        )
    ]

    header = f"{'strategy':<24}{'chunks':>7}{'min':>7}{'max':>7}{'avg':>8}{'emitted':>9}{'coverage':>10}"
    lines.append(header)
    lines.append(rule(len(header)))

    summaries: Dict[str, Dict[str, float]] = {}
    for name, chunker in chunkers.items():
        chunks = chunker.split_documents([document])
        summary = summarize_chunks(document, chunks)
        summaries[name] = summary
        lines.append(
            f"{name:<24}{int(summary['chunks']):>7}{int(summary['min']):>7}"
            f"{int(summary['max']):>7}{summary['avg']:>8.0f}"
            f"{int(summary['emitted']):>9}{summary['coverage'] * 100:>9.0f}%"
        )

    lines.append("")
    lines.append(
        "coverage > 100% means overlap: the same characters appear in 2+ chunks."
    )

    for name, chunker in chunkers.items():
        lines.append("")
        lines.append(rule(78))
        chunks = chunker.split_documents([document])
        lines.append(render_chunk_spans(document, chunks, width=width, name=name))

    return "\n".join(lines)


def preview_chunks(
    chunks: Sequence[Document],
    count: int = 3,
    snippet_width: int = 72,
) -> str:
    """A quick numbered preview of the first few chunk texts."""
    lines: List[str] = []
    for index, chunk in enumerate(chunks[:count]):
        location = chunk.metadata.get("source")
        label = Path(location).name if location else "?"
        lines.append(
            f"[{index}] ({label}, {len(chunk)} chars) "
            f"{truncate(chunk.page_content, snippet_width)}"
        )
    if len(chunks) > count:
        lines.append(f"... and {len(chunks) - count} more chunk(s)")
    return "\n".join(lines)


def render_size_distribution(
    document: Document,
    chunks: Sequence[Document],
    width: int = 40,
) -> str:
    """A bar per chunk showing its size — handy for spotting tiny/huge chunks."""
    sizes = [len(chunk) for chunk in chunks]
    largest = max(sizes) if sizes else 1
    lines = [title(f"Chunk sizes (max {largest})", width=width + 20)]
    for index, size in enumerate(sizes):
        lines.append(f"{index:>3} {size:>6} |{bar(size, largest, width)}|")
    return "\n".join(lines)
