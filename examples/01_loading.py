"""
LESSON 1 — Loading documents.

The first stage of RAG: turn files on disk into `Document` objects (text +
metadata).  Metadata survives the whole pipeline and is what lets us cite
sources later.

Run:
    python examples/01_loading.py
"""

import _bootstrap  # noqa: F401  (sets up imports; must come first)

from pathlib import Path

from rag.loaders import TextLoader, MarkdownLoader, DirectoryLoader
from rag.config import settings

SAMPLE_DIR = settings.data_dir / "sample_docs"


def main() -> None:
    print("=" * 70)
    print("LESSON 1: Loading documents")
    print("=" * 70)

    # ---- 1a. Load ONE markdown file ------------------------------------- #
    md_path = SAMPLE_DIR / "intro_to_rag.md"
    markdown_docs = MarkdownLoader().load(str(md_path))
    doc = markdown_docs[0]

    print(f"\nLoaded {len(markdown_docs)} document(s) from {md_path.name}")
    print(f"  type        : {doc.metadata['file_type']}")
    print(f"  characters  : {len(doc)}")
    print(f"  id          : {doc.id}")
    print(f"  headings    : {doc.metadata['headings'][:4]} ...")
    print(f"  preview     : {doc.page_content[:120].strip()!r} ...")

    # ---- 1b. Load ONE text file ----------------------------------------- #
    txt_path = SAMPLE_DIR / "chunking_and_prompts.txt"
    text_docs = TextLoader().load(str(txt_path))
    print(f"\nLoaded {len(text_docs)} document(s) from {txt_path.name} "
          f"({len(text_docs[0])} characters)")

    # ---- 1c. Load a WHOLE directory at once ----------------------------- #
    # This is what the ingestion pipeline uses under the hood.
    all_docs = DirectoryLoader().load(str(SAMPLE_DIR))
    print(f"\nDirectoryLoader read {len(all_docs)} document(s) from {SAMPLE_DIR.name}:")
    for d in all_docs:
        print(f"  - {d.metadata['file_name']:<28} {len(d):>5} chars")

    print("\nKey idea: every loader returns the SAME Document shape,")
    print("so the rest of the pipeline never cares about file formats.")


if __name__ == "__main__":
    main()
