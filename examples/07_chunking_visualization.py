"""
LESSON 7 — Visualizing chunking strategies.

Chunking is invisible in most pipelines, which makes it hard to build intuition.
This lesson runs FIVE chunkers over the same document and draws the result:

  * a summary table (chunk count, sizes, overlap/coverage), and
  * a span map per strategy, where overlapping chunks visibly overlap.

Run:
    python examples/07_chunking_visualization.py
"""

import _bootstrap  # noqa: F401

from rag.config import settings
from rag.chunkers import (
    FixedSizeChunker,
    TokenChunker,
    RecursiveChunker,
    SentenceChunker,
    MarkdownHeaderChunker,
)
from rag.loaders import MarkdownLoader
from rag.visualization import (
    preview_chunks,
    render_chunking_report,
    render_size_distribution,
)

SAMPLE = settings.data_dir / "sample_docs" / "intro_to_rag.md"


def main() -> None:
    document = MarkdownLoader().load(str(SAMPLE))[0]

    # Five strategies, all roughly targeting the same budget.  Note that the
    # TokenChunker's `chunk_size` counts WORDS, not characters.
    chunkers = {
        "FixedSizeChunker": FixedSizeChunker(chunk_size=300, chunk_overlap=60),
        "TokenChunker": TokenChunker(chunk_size=50, chunk_overlap=10),
        "RecursiveChunker": RecursiveChunker(chunk_size=300, chunk_overlap=60),
        "SentenceChunker": SentenceChunker(chunk_size=300, chunk_overlap=60),
        "MarkdownHeaderChunker": MarkdownHeaderChunker(chunk_size=300, chunk_overlap=60),
    }

    print(render_chunking_report(document, chunkers, width=64))

    # Zoom in on one strategy to see individual chunk sizes and text.
    print("\n" + "=" * 78)
    print("Zoom-in: RecursiveChunker")
    print("=" * 78)
    recursive = RecursiveChunker(chunk_size=300, chunk_overlap=60)
    chunks = recursive.split_documents([document])
    print(render_size_distribution(document, chunks, width=44))
    print("\nFirst chunks:")
    print(preview_chunks(chunks, count=4))

    print("\nHow to read the span maps")
    print("-------------------------")
    print("- Overlapping bars show shared text between neighbouring chunks.")
    print("- MarkdownHeaderChunker prefixes each chunk with a heading breadcrumb,")
    print("  so it may emit more/different text than the others.")
    print("- coverage > 100% is expected whenever chunk_overlap > 0.")


if __name__ == "__main__":
    main()
