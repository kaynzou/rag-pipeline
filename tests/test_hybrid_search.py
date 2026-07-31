import pytest
import numpy as np
from src.chunking import TextSplitter, Chunk
from src.bm25 import BM25Index
from src.vector_store import VectorStore
from src.hybrid_search import HybridSearch, HybridResult


@pytest.fixture(scope="module")
def setup():
    chunks = [
        Chunk(text="BM25 is a keyword scoring algorithm for retrieval", chunk_id=0, source="a.txt", start_token=0, end_token=10),
        Chunk(text="RAG combines retrieval with generation", chunk_id=1, source="b.txt", start_token=0, end_token=10),
        Chunk(text="Vector embeddings capture semantic meaning", chunk_id=2, source="c.txt", start_token=0, end_token=10),
        Chunk(text="Evaluation metrics measure RAG performance", chunk_id=3, source="d.txt", start_token=0, end_token=10),
        Chunk(text="The cat sat on the mat", chunk_id=4, source="e.txt", start_token=0, end_token=10),
    ]
    encoder = __import__("src.embedding", fromlist=["LocalEmbeddingModel"]).LocalEmbeddingModel()
    pipeline = __import__("src.embedding", fromlist=["EmbeddingPipeline"]).EmbeddingPipeline(model=encoder)
    embedded = pipeline.embed_chunks(chunks)

    bm25 = BM25Index()
    bm25.build(chunks)

    vector_store = VectorStore(dimension=encoder.dimension, model_name=encoder.model_name)
    vector_store.add(embedded)

    hybrid = HybridSearch(
        bm25_index=bm25,
        vector_store=vector_store,
        encoder=encoder,
        top_k=3,
        candidates_per_method=10,
    )

    return hybrid, chunks


class TestHybridSearch:
    def test_search_returns_results(self, setup):
        hybrid, _ = setup
        results = hybrid.search("BM25 scoring")
        assert len(results) > 0
        assert all(isinstance(r, HybridResult) for r in results)

    def test_result_has_all_fields(self, setup):
        hybrid, _ = setup
        results = hybrid.search("BM25")
        assert len(results) > 0
        r = results[0]
        assert hasattr(r, "chunk_id")
        assert hasattr(r, "text")
        assert hasattr(r, "bm25_score")
        assert hasattr(r, "dense_score")
        assert hasattr(r, "rrf_score")

    def test_scores_descending(self, setup):
        hybrid, _ = setup
        results = hybrid.search("BM25 scoring")
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_term_boosted(self, setup):
        hybrid, _ = setup
        results = hybrid.search("BM25")
        assert results[0].chunk_id == 0

    def test_top_k_respected(self, setup):
        hybrid, _ = setup
        results = hybrid.search("BM25 scoring")
        assert len(results) <= 3

    def test_dense_score_present(self, setup):
        hybrid, _ = setup
        results = hybrid.search("vector embeddings")
        assert any(r.dense_score > 0 for r in results)

    def test_no_results_for_nonexistent(self, setup):
        hybrid, _ = setup
        results = hybrid.search("xyznonexistentterm12345")
        assert all(r.bm25_score == 0.0 for r in results)
