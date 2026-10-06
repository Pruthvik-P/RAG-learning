"""
Visualize retrieval and reranking.

Three views:

* render_results           — one retriever's ranked list as a bar chart.
* render_retriever_comparison — several retrievers side by side; cells show the
  rank a document got in each strategy, so you can see where they disagree.
* render_rerank            — the retriever's list BEFORE vs the reranker's list
  AFTER, with arrows showing how far each chunk moved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from ..retrievers.base import BaseRetriever
from ..vectorstores.base import ScoredDocument
from .bars import min_max_bars, rule, title, truncate


def _source(document) -> str:
    location = document.metadata.get("source") or document.metadata.get("file_name")
    return Path(location).name if location else "<memory>"


def _label(document, snippet_width: int) -> str:
    return f"[{_source(document)}] {truncate(document.page_content, snippet_width)}"


def render_results(
    query: str,
    results: Sequence[ScoredDocument],
    width: int = 44,
    name: str = "results",
    snippet_width: int = 60,
) -> str:
    """Render a ranked result list with proportional score bars."""
    lines = [title(f"{name}  —  query: {truncate(query, 60)}")]
    if not results:
        lines.append("(no results)")
        return "\n".join(lines)

    bars = min_max_bars([score for _, score in results], width=width)
    for rank, ((document, score), drawing) in enumerate(zip(results, bars), start=1):
        lines.append(f"{rank:>2}. {score:+.3f} |{drawing}| {_source(document)}")
        lines.append(f"      {truncate(document.page_content, snippet_width)}")
    return "\n".join(lines)


def render_retriever_comparison(
    query: str,
    retrievers: Dict[str, BaseRetriever],
    k: int = 4,
    snippet_width: int = 40,
) -> str:
    """
    Table comparing which documents each retriever surfaced, and at what rank.

    Rows are documents; columns are retrievers; a cell is the 1-based rank or
    '-'.  Rows are sorted by the best (smallest) rank any retriever gave them.
    """
    names = list(retrievers)
    results: Dict[str, List[ScoredDocument]] = {
        name: retriever.retrieve(query, k=k) for name, retriever in retrievers.items()
    }

    # Collect every document once, in first-seen order.
    documents: List = []
    seen = set()
    for name in names:
        for document, _ in results[name]:
            if document.id not in seen:
                seen.add(document.id)
                documents.append(document)

    rank_maps: Dict[str, Dict[str, int]] = {
        name: {document.id: rank for rank, (document, _) in enumerate(results[name], 1)}
        for name in names
    }

    def best_rank(document) -> int:
        return min(rank_maps[name].get(document.id, 10**9) for name in names)

    documents.sort(key=best_rank)

    col = max(max(len(name) for name in names), 8) + 2
    header = f"{'document':<{snippet_width}}" + "".join(
        f"{name:>{col}}" for name in names
    )
    lines = [title(f"RETRIEVER COMPARISON  —  query: {truncate(query, 56)}"), header, rule(len(header))]

    for document in documents:
        row = f"{_label(document, snippet_width - 1):<{snippet_width}}"
        for name in names:
            rank = rank_maps[name].get(document.id)
            row += f"{(str(rank) if rank else '-'):>{col}}"
        lines.append(row)

    return "\n".join(lines)


def render_rerank(
    query: str,
    before: Sequence[ScoredDocument],
    after: Sequence[ScoredDocument],
    width: int = 36,
    name: str = "reranker",
    snippet_width: int = 52,
) -> str:
    """
    Before/after view of reranking.

    `before` is the retriever output; `after` is the reranker output.  Each line
    is a chunk in its NEW position, annotated with where it used to rank.
    """
    lines = [title(f"RERANK  ({name})  —  query: {truncate(query, 52)}")]
    if not after:
        lines.append("(no results)")
        return "\n".join(lines)

    before_rank: Dict[str, int] = {
        document.id: rank for rank, (document, _) in enumerate(before, 1)
    }
    after_bars = min_max_bars([score for _, score in after], width=width)

    lines.append(f"  {'new':>3} {'was':>5} {'move':>6}  {'score':>7}  chunk")
    lines.append(rule(len(lines[-1]) + snippet_width))
    for new_rank, ((document, score), drawing) in enumerate(zip(after, after_bars), 1):
        was = before_rank.get(document.id)
        if was is None:
            was_text, move = "new", "++"
        else:
            was_text = str(was)
            delta = was - new_rank
            move = "^" + str(delta) if delta > 0 else ("v" + str(-delta) if delta < 0 else "=")
        lines.append(
            f"  {new_rank:>3} {was_text:>5} {move:>6}  {score:>+7.3f}  "
            f"{_label(document, snippet_width)}"
        )
        lines.append(f"  {'':>23}|{drawing}|")

    dropped = [
        document
        for document, _ in before
        if document.id not in {doc.id for doc, _ in after}
    ]
    if dropped:
        lines.append("")
        lines.append("Dropped from the top-k after reranking:")
        for document in dropped:
            lines.append(f"  - {_label(document, snippet_width)}")

    return "\n".join(lines)
