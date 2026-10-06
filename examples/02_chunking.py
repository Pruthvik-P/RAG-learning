"""
LESSON 2 — Chunking.

Long documents are split into smaller passages so retrieval can be precise and
prompts stay small.  This script compares three strategies side by side.

Run:
    python examples/02_chunking.py
"""

import _bootstrap  # noqa: F401

from rag.config import settings
from rag.loaders import MarkdownLoader
from rag.chunkers import FixedSizeChunker, RecursiveChunker, SentenceChunker

SAMPLE = settings.data_dir / "sample_docs" / "intro_to_rag.md"


def main() -> None:
    document = MarkdownLoader().load(str(SAMPLE))[0]

    print("=" * 70)
    print("LESSON 2: Chunking strategies")
    print("=" * 70)
    print(f"Source document: {len(document)} characters\n")

    chunkers = {
        "FixedSizeChunker": FixedSizeChunker(chunk_size=400, chunk_overlap=80),
        "RecursiveChunker": RecursiveChunker(chunk_size=400, chunk_overlap=80),
        "SentenceChunker": SentenceChunker(chunk_size=400, chunk_overlap=80),
    }

    for name, chunker in chunkers.items():
        chunks = chunker.split_documents([document])
        sizes = [len(c) for c in chunks]
        print(f"{name}")
        print(f"  chunks       : {len(chunks)}")
        print(f"  size range   : {min(sizes)} .. {max(sizes)} chars")
        print(f"  metadata keys: {sorted(chunks[0].metadata.keys())}")
        print(f"  chunk 0      : {chunks[0].page_content[:90].strip()!r} ...")
        print()

    # Show how overlap + metadata help you trace a chunk back to its origin.
    print("Notice each chunk keeps its source metadata plus chunk_index/total.")
    print("That is how an answer can later say 'this came from intro_to_rag.md'.")

    print("\nWhich to use?")
    print("  - RecursiveChunker is the best general-purpose default.")
    print("  - SentenceChunker is clean for prose.")
    print("  - FixedSizeChunker is simplest when you need exact sizes.")


if __name__ == "__main__":
    main()
