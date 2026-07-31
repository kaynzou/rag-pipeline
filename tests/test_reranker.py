import pytest
from src.chunking import TextSplitter, Chunk
from src.bm25 import BM25Index
from src.embedding import LocalEmbeddingModel, EmbeddingPipeline
from src.vector_store import VectorStore
from src.hybrid_search import HybridSearch, HybridResult
from src.reranker import CrossEncoderReranker, RerankResult


@pytest.fixture(scope="module")
def setup():
    chunks = [
        Chunk(text="BM25 is a keyword scoring algorithm for retrieval", chunk_id=0, source="a.txt", start_token=0, end_token=10),
        Chunk(text="RAG combines retrieval with generation", chunk_id=1, source="b.txt", start_token=0, end_token=10),
        Chunk(text="Vector embeddings capture semantic meaning", chunk_id=2, source="c.txt", start_token=0, end_token=10),
        Chunk(text="Evaluation metrics measure RAG performance", chunk_id=3, source="d.txt", start_token=0, end_token=10),
        Chunk(text="The cat sat on the mat", chunk_id=4, source="e.txt", start_token=0, end_token=10),
    ]
    encoder = LocalEmbeddingModel()
    pipeline = EmbeddingPipeline(model=encoder)
    embedded = pipeline.embed_chunks(chunks)

    bm25 = BM25Index()
    bm25.build(chunks)

    vector_store = VectorStore(dimension=encoder.dimension, model_name=encoder.model_name)
    vector_store.add(embedded)

    hybrid = HybridSearch(
        bm25_index=bm25,
        vector_store=vector_store,
        encoder=encoder,
        top_k=5,
        candidates_per_method=10,
    )

    reranker = CrossEncoderReranker()

    return hybrid, reranker, chunks


class TestCrossEncoderReranker:
    def test_rerank_returns_results(self, setup):
        hybrid, reranker, _ = setup
        results = hybrid.search("BM25 scoring")
        reranked = reranker.rerank("BM25 scoring", results, top_k=3)
        assert len(reranked) > 0
        assert all(isinstance(r, RerankResult) for r in reranked)

    def test_rerank_respects_top_k(self, setup):
        hybrid, reranker, _ = setup
        results = hybrid.search("BM25 scoring")
        reranked = reranker.rerank("BM25 scoring", results, top_k=2)
        assert len(reranked) <= 2

    def test_rerank_preserves_scores(self, setup):
        hybrid, reranker, _ = setup
        results = hybrid.search("BM25 scoring")
        reranked = reranker.rerank("BM25 scoring", results, top_k=5)
        assert all(hasattr(r, "rerank_score") for r in reranked)
        assert all(hasattr(r, "bm25_score") for r in reranked)
        assert all(hasattr(r, "dense_score") for r in reranked)

    def test_rerank_scores_descending(self, setup):
        hybrid, reranker, _ = setup
        results = hybrid.search("BM25 scoring")
        reranked = reranker.rerank("BM25 scoring", results, top_k=5)
        scores = [r.rerank_score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty_input(self, setup):
        _, reranker, _ = setup
        reranked = reranker.rerank("test", [], top_k=5)
        assert reranked == []

    def test_exact_match_reranked_high(self, setup):
        hybrid, reranker, _ = setup
        results = hybrid.search("BM25 keyword scoring")
        reranked = reranker.rerank("BM25 keyword scoring", results, top_k=1)
        assert reranked[0].chunk_id == 0