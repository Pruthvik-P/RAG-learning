"""
LESSON 6 — Full RAG end to end with DeepSeek.

This is the payoff: ask a natural-language question and get a grounded answer
with citations.

Setup:
    1. Copy ".env.example" to ".env".
    2. Put your DeepSeek key in DEEPSEEK_API_KEY.
    (Without a key the script still runs using the offline DummyLLM so you can
    see the exact prompt that would be sent.)

Run:
    python examples/06_full_rag.py
    python examples/06_full_rag.py "What is MMR?"   # ask your own question
"""

import _bootstrap  # noqa: F401

import sys
from pathlib import Path

from rag.config import settings
from rag.pipelines import RAGPipeline

SAMPLE_DIR = settings.data_dir / "sample_docs"


def print_sources(sources) -> None:
    print("\nSources used:")
    for i, (doc, score) in enumerate(sources, 1):
        path = Path(doc.metadata.get("source", "?")).name
        idx = doc.metadata.get("chunk_index", "?")
        print(f"  [{i}] {path} (chunk {idx}, score {score:.3f})")


def main() -> None:
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is retrieval-augmented generation and why is it useful?"
    )

    print("=" * 70)
    print("LESSON 6: Full RAG pipeline (DeepSeek)")
    print("=" * 70)

    # Build once: ingest the folder, then wrap it in the query pipeline.
    rag = RAGPipeline.from_directory(str(SAMPLE_DIR), settings=settings)

    llm_name = type(rag.llm).__name__
    print(f"\nLLM: {llm_name}  |  mode: {settings.retrieval_mode}  |  top_k: {settings.top_k}")

    # Show the THREE stages explicitly so the flow is visible.
    print(f"\nQuestion: {question}")

    print("\n[1/3] RETRIEVE ...")
    sources = rag.retrieve(question)
    print_sources(sources)

    print("\n[2/3] AUGMENT (building the prompt) ...")
    prompt = rag.build_prompt(question, sources)
    print(f"  prompt length: {len(prompt)} characters")

    print("\n[3/3] GENERATE ...")
    answer = rag.generate(prompt)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    if llm_name == "DummyLLM":
        print("\n(Tip: set DEEPSEEK_API_KEY in .env for real generated answers.)")


if __name__ == "__main__":
    main()
