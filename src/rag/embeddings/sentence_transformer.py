"""
Sentence-Transformers embedding (optional, high quality).

This wraps a real neural embedding model.  It is OPTIONAL because it needs:

    pip install sentence-transformers

(which also pulls in PyTorch — a large download).  When you are ready for
noticeably better retrieval quality, set:

    RAG_EMBEDDING_PROVIDER=sentence_transformer

The model is loaded lazily so merely importing this module costs nothing.
"""

from __future__ import annotations

from typing import List, Optional

from .base import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):
    """
    Embed text with a `sentence-transformers` model.

    Args:
        model_name: Any model on the HuggingFace hub.  The default
            "all-MiniLM-L6-v2" is small (~80 MB), fast, and a great default.
        device: "cpu", "cuda", or None to auto-detect.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None            # loaded on first use
        self._dimensions: Optional[int] = None

    def _load(self):
        """Import and instantiate the model exactly once."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SentenceTransformerEmbedding requires 'sentence-transformers'.\n"
                "Install it with:  pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(self.model_name, device=self.device)
        # Ask the model how wide its vectors are.
        self._dimensions = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        self._load()
        return int(self._dimensions)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._load()
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        # Convert numpy rows to plain Python lists so the rest of the pipeline
        # stays dependency-agnostic.
        return [list(map(float, row)) for row in vectors]
