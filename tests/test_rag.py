"""
Unit tests for the RAG learning package.

Uses the standard-library `unittest` runner so no installation is required:

    python -m unittest discover -s tests -v

The final test exercises the WHOLE pipeline offline with the DummyLLM, which
proves every stage is wired together correctly.
"""

import sys
import unittest
from pathlib import Path

# Make `import rag` work when tests run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.chunkers import FixedSizeChunker, RecursiveChunker, SentenceChunker
from rag.embeddings import HashingEmbedding
from rag.loaders import Document, TextLoader, MarkdownLoader, DirectoryLoader
from rag.llms import DummyLLM
from rag.pipelines import IngestionPipeline, RAGPipeline
from rag.retrievers import SimilarityRetriever, MMRRetriever
from rag.vectorstores import InMemoryVectorStore

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_docs"


class TestLoaders(unittest.TestCase):
    def test_text_loader(self):
        docs = TextLoader().load(str(SAMPLE_DIR / "chunking_and_prompts.txt"))
        self.assertEqual(len(docs), 1)
        self.assertGreater(len(docs[0]), 0)
        self.assertEqual(docs[0].metadata["file_type"], "text")

    def test_markdown_loader_extracts_headings(self):
        docs = MarkdownLoader().load(str(SAMPLE_DIR / "intro_to_rag.md"))
        self.assertEqual(len(docs), 1)
        self.assertTrue(docs[0].metadata["headings"])
        self.assertNotIn("---", docs[0].page_content[:10])

    def test_directory_loader(self):
        docs = DirectoryLoader().load(str(SAMPLE_DIR))
        self.assertGreaterEqual(len(docs), 3)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            TextLoader().load("does/not/exist.txt")


class TestDocument(unittest.TestCase):
    def test_deterministic_id(self):
        a = Document(page_content="hello", metadata={"source": "x"})
        b = Document(page_content="hello", metadata={"source": "x"})
        self.assertEqual(a.id, b.id)

    def test_length(self):
        self.assertEqual(len(Document(page_content="abcd")), 4)


class TestChunkers(unittest.TestCase):
    def _doc(self):
        return Document(page_content="A" * 1000, metadata={"source": "s"})

    def test_fixed_size_respects_size(self):
        chunks = FixedSizeChunker(chunk_size=100, chunk_overlap=10).split_documents([self._doc()])
        self.assertTrue(all(len(c.page_content) <= 100 for c in chunks))
        self.assertGreater(len(chunks), 1)

    def test_recursive_preserves_metadata(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split_documents([self._doc()])
        self.assertEqual(chunks[0].metadata["source"], "s")
        self.assertEqual(chunks[0].metadata["chunk_total"], len(chunks))

    def test_sentence_chunker_splits_sentences(self):
        text = "One. Two. Three. Four. Five."
        chunks = SentenceChunker(chunk_size=12, chunk_overlap=0).split_documents(
            [Document(page_content=text)]
        )
        self.assertGreaterEqual(len(chunks), 2)

    def test_overlap_must_be_smaller(self):
        with self.assertRaises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)


class TestEmbeddings(unittest.TestCase):
    def test_dimensions_and_normalization(self):
        emb = HashingEmbedding(dimensions=256)
        vector = emb.embed_query("hello world hello world")
        self.assertEqual(len(vector), 256)
        # L2 norm should be ~1.0 for non-empty text.
        norm = sum(x * x for x in vector) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_deterministic(self):
        emb = HashingEmbedding(dimensions=256)
        self.assertEqual(emb.embed_query("same text"), emb.embed_query("same text"))

    def test_empty_text(self):
        emb = HashingEmbedding(dimensions=64)
        self.assertEqual(emb.embed_query(""), [0.0] * 64)


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryVectorStore(HashingEmbedding(dimensions=512))
        self.store.add_texts(
            [
                "Vector databases store embeddings for similarity search.",
                "Chunking splits documents into smaller passages.",
                "The cat sat on the mat.",
            ]
        )

    def test_add_and_len(self):
        self.assertEqual(len(self.store), 3)

    def test_search_ranks_relevant_first(self):
        results = self.store.similarity_search("similarity search embeddings", k=1)
        self.assertIn("Vector databases", results[0][0].page_content)

    def test_top_k_limit(self):
        self.assertEqual(len(self.store.similarity_search("anything", k=2)), 2)

    def test_delete(self):
        doc_id = self.store.documents[0].id
        self.store.delete([doc_id])
        self.assertEqual(len(self.store), 2)

    def test_save_and_load(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "index.json")
            self.store.save(path)
            reloaded = InMemoryVectorStore.load(path, HashingEmbedding(dimensions=512))
            self.assertEqual(len(reloaded), len(self.store))


class TestRetrievers(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryVectorStore(HashingEmbedding(dimensions=512))
        self.store.add_texts([f"Document number {i} about topic {i}" for i in range(10)])

    def test_similarity_returns_k(self):
        results = SimilarityRetriever(self.store).retrieve("topic 3", k=3)
        self.assertEqual(len(results), 3)

    def test_mmr_returns_unique(self):
        results = MMRRetriever(self.store, lambda_mult=0.5).retrieve("topic", k=4)
        ids = [doc.id for doc, _ in results]
        self.assertEqual(len(ids), len(set(ids)))


class TestPipelineEndToEnd(unittest.TestCase):
    def test_full_pipeline_offline(self):
        # DummyLLM lets everything run with no API key / network.
        ingestion = IngestionPipeline()
        store = ingestion.ingest(str(SAMPLE_DIR), verbose=False)
        self.assertGreater(len(store), 0)

        rag = RAGPipeline(vector_store=store, llm=DummyLLM())
        response = rag.answer("What is RAG?")

        self.assertTrue(response.answer)
        self.assertTrue(response.sources)
        self.assertIn("What is RAG?", response.prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
