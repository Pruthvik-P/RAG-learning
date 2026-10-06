"""
Central configuration for the RAG learning project.

Why a config module?
--------------------
A RAG system has a LOT of "knobs": chunk size, how many chunks to retrieve,
which embedding model to use, which LLM to call, API keys, etc.  Instead of
scattering magic numbers across the codebase we keep them in one place.

We read settings from environment variables (and an optional `.env` file) so
that secrets like the DeepSeek API key are never hard-coded in source files.

Learning notes
--------------
* `DEEPSEEK_API_KEY`  -> your key from https://platform.deepseek.com
* `DEEPSEEK_BASE_URL` -> DeepSeek exposes an OpenAI-compatible API, so the
  base URL looks like an OpenAI base URL ("https://api.deepseek.com/v1").
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Tiny .env loader (no external dependency like python-dotenv required)
# --------------------------------------------------------------------------- #
def load_dotenv(dotenv_path: Optional[Path] = None) -> None:
    """
    Load key=value pairs from a `.env` file into os.environ.

    This keeps the project runnable with ZERO third-party packages.  If a key
    is already present in the real environment we do NOT overwrite it, so a
    real shell export always wins over the file.
    """
    if dotenv_path is None:
        # config.py -> rag -> src -> project root
        dotenv_path = Path(__file__).resolve().parents[2] / ".env"

    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # Skip blanks and comments.
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# Load .env as soon as this module is imported.
load_dotenv()


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to a default."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to a default."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """
    All tunable settings in one object.

    You can create a custom instance and pass it into the pipeline, which makes
    experimenting (e.g. chunk_size=256 vs 1024) easy:

        settings = Settings(chunk_size=256, top_k=3)
        pipeline = RAGPipeline(settings=settings)
    """

    # ------------------------ paths ------------------------ #
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    # ------------------------ chunking --------------------- #
    # How many characters per chunk, and how much adjacent chunks overlap.
    # Overlap helps avoid cutting a sentence's meaning in half at a boundary.
    chunk_size: int = _env_int("RAG_CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("RAG_CHUNK_OVERLAP", 120)

    # ------------------------ embeddings ------------------- #
    # The "hashing" embedder is pure-Python (no downloads) and is the default
    # so the project works offline.  Swap in SentenceTransformerEmbedding for
    # much better semantic quality once you install `sentence-transformers`.
    embedding_provider: str = os.environ.get("RAG_EMBEDDING_PROVIDER", "hashing")
    embedding_dimensions: int = _env_int("RAG_EMBEDDING_DIM", 1024)
    embedding_model: str = os.environ.get(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ------------------------ retrieval -------------------- #
    top_k: int = _env_int("RAG_TOP_K", 4)          # chunks to retrieve
    retrieval_mode: str = os.environ.get("RAG_RETRIEVAL_MODE", "similarity")
    mmr_lambda: float = _env_float("RAG_MMR_LAMBDA", 0.5)  # 1.0=relevance, 0=diversity

    # ------------------------ reranking -------------------- #
    # Optional second pass that reorders the retrieved candidates.  The retriever
    # fetches `rerank_candidates` chunks first, then the reranker keeps top_k.
    use_reranking: bool = os.environ.get("RAG_USE_RERANKING", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    rerank_kind: str = os.environ.get("RAG_RERANK_KIND", "lexical")
    rerank_candidates: int = _env_int("RAG_RERANK_CANDIDATES", 20)

    # ------------------------ DeepSeek LLM ----------------- #
    deepseek_api_key: Optional[str] = os.environ.get("DEEPSEEK_API_KEY")
    deepseek_base_url: str = os.environ.get(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
    )
    llm_model: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    llm_temperature: float = _env_float("DEEPSEEK_TEMPERATURE", 0.2)
    llm_max_tokens: int = _env_int("DEEPSEEK_MAX_TOKENS", 1024)
    llm_timeout: int = _env_int("DEEPSEEK_TIMEOUT", 60)

    # ------------------------ misc ------------------------- #
    # If True, the pipeline quietly falls back to an offline "DummyLLM" when
    # no API key is configured.  Handy for learning without spending money.
    allow_offline_fallback: bool = os.environ.get(
        "RAG_ALLOW_OFFLINE", "true"
    ).lower() in {"1", "true", "yes"}

    @property
    def data_dir(self) -> Path:
        """Default folder where sample documents live."""
        return self.project_root / "data"


# A ready-to-use default settings instance.  Import it directly:
#     from rag.config import settings
settings = Settings()
