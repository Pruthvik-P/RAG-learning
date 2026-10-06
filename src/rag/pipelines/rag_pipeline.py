"""
The RAG pipeline: the "read side" of RAG.

Given a question it runs the classic three steps:

        RETRIEVE  ->  AUGMENT  ->  GENERATE
        (find k       (build a     (LLM writes
         chunks)       prompt)      the answer)

        question
           |
           v
   [embed query] --> [vector search] --> top-k chunks
           |
           v
   [prompt template: context + question]
           |
           v
   [DeepSeek LLM] --> grounded answer + cited sources

Usage:
    from rag.pipelines.rag_pipeline import RAGPipeline
    rag = RAGPipeline.from_directory("data/sample_docs")
    result = rag.answer("What is RAG?")
    print(result.answer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..config import Settings, settings as default_settings
from ..llms.base import BaseLLM
from ..llms import get_llm
from ..retrievers.base import BaseRetriever
from ..retrievers import get_retriever
from ..vectorstores.base import BaseVectorStore, ScoredDocument
from .ingestion import IngestionPipeline


# --------------------------------------------------------------------------- #
# Prompt template — the heart of "augmentation"
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. Follow these rules:\n"
    "1. Base every statement on the context. Do not invent facts.\n"
    "2. If the answer is not in the context, say you don't know.\n"
    "3. Cite the sources you used using their [number] tags.\n"
    "4. Be concise and direct."
)

PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
Each context block is labelled with a source number.

{context}

---
Question: {question}

Answer (cite the [number]s you used):"""


@dataclass
class RAGResponse:
    """Everything the pipeline produced, useful for debugging + citations."""

    answer: str
    question: str
    sources: List[ScoredDocument] = field(default_factory=list)
    prompt: str = ""

    def __str__(self) -> str:  # nicer printing in examples
        return self.answer


class RAGPipeline:
    """
    End-to-end retrieval-augmented generation.

    Args:
        vector_store: A populated store (from IngestionPipeline).
        llm: Any BaseLLM.  Defaults to DeepSeek (or DummyLLM offline).
        retriever: Any BaseRetriever.  Defaults to similarity search.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        llm: Optional[BaseLLM] = None,
        retriever: Optional[BaseRetriever] = None,
        settings: Settings = default_settings,
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.retriever = retriever or get_retriever(
            vector_store,
            mode=settings.retrieval_mode,
            lambda_mult=settings.mmr_lambda,
        )
        self.llm = llm or get_llm(
            provider="deepseek",
            api_key=settings.deepseek_api_key,
            allow_offline=settings.allow_offline_fallback,
            base_url=settings.deepseek_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )

    # ------------------------------------------------------------------ #
    # Convenience constructor: ingest a folder then be ready to answer.
    # ------------------------------------------------------------------ #
    @classmethod
    def from_directory(
        cls,
        source: str,
        settings: Settings = default_settings,
        llm: Optional[BaseLLM] = None,
        verbose: bool = True,
    ) -> "RAGPipeline":
        """Build an ingestion pipeline, ingest `source`, and wrap the store."""
        ingestion = IngestionPipeline(settings=settings)
        store = ingestion.ingest(source, verbose=verbose)
        return cls(vector_store=store, llm=llm, settings=settings)

    # ------------------------------------------------------------------ #
    # The three steps
    # ------------------------------------------------------------------ #
    def retrieve(self, question: str, k: Optional[int] = None) -> List[ScoredDocument]:
        """STEP 1: fetch the most relevant chunks for the question."""
        top_k = k or self.settings.top_k
        return self.retriever.retrieve(question, k=top_k)

    def build_prompt(self, question: str, sources: List[ScoredDocument]) -> str:
        """STEP 2 (AUGMENT): format retrieved chunks into the prompt."""
        blocks = []
        for number, (document, score) in enumerate(sources, start=1):
            source = document.metadata.get("source", "unknown")
            blocks.append(
                f"[{number}] (source: {source})\n{document.page_content}"
            )
        context = "\n\n".join(blocks) if blocks else "(no relevant context found)"
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(self, prompt: str) -> str:
        """STEP 3 (GENERATE): ask the LLM to answer from the context."""
        return self.llm.generate(prompt, system=SYSTEM_PROMPT)

    # ------------------------------------------------------------------ #
    def answer(self, question: str, k: Optional[int] = None) -> RAGResponse:
        """Run retrieve -> augment -> generate and return the full response."""
        sources = self.retrieve(question, k=k)
        prompt = self.build_prompt(question, sources)
        text = self.generate(prompt)
        return RAGResponse(
            answer=text, question=question, sources=sources, prompt=prompt
        )

    # `ask` is a friendly alias.
    ask = answer

    def stream_answer(self, question: str, k: Optional[int] = None):
        """Yield (stage, payload) tuples so callers can narrate the process."""
        yield "retrieve", None
        sources = self.retrieve(question, k=k)
        yield "sources", sources

        yield "augment", None
        prompt = self.build_prompt(question, sources)
        yield "prompt", prompt

        yield "generate", None
        yield "answer", self.generate(prompt)
