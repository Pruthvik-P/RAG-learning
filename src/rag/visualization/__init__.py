"""
Visualization package — see what each RAG stage actually does.

Terminal-first, zero dependencies.  Import the report you need:

    from rag.visualization import render_chunking_report, render_retriever_comparison, render_rerank

* chunking : compare chunkers and see the exact spans/overlap on a timeline.
* retrieval: bar charts, retriever side-by-side comparison, and rerank diffs.
"""

from .bars import bar, min_max_bars, rule, span_bar, title, truncate
from .chunking import (
    preview_chunks,
    render_chunk_spans,
    render_chunking_report,
    render_size_distribution,
    summarize_chunks,
)
from .retrieval import (
    render_rerank,
    render_results,
    render_retriever_comparison,
)

__all__ = [
    "bar",
    "min_max_bars",
    "rule",
    "span_bar",
    "title",
    "truncate",
    "summarize_chunks",
    "render_chunk_spans",
    "render_chunking_report",
    "render_size_distribution",
    "preview_chunks",
    "render_results",
    "render_retriever_comparison",
    "render_rerank",
]
