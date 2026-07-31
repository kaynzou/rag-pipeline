import pytest
from src.chunking import TextSplitter, Chunk
from src.bm25 import BM25Index, BM25Result, reciprocal_rank_fusion


class TestBM25Index:
    def setup_method(self):
        self.chunks = [
            Chunk(text="RAG combines retrieval with generation", chunk_id=0, source="a.txt", start_token=0, end_token=10),
            Chunk(text="BM25 is a keyword scoring algorithm", chunk_id=1, source="b.txt", start_token=0, end_token=10),
            Chunk(text="Vector embeddings capture semantic meaning", chunk_id=2, source="c.txt", start_token=0, end_token=10),
            Chunk(text="The cat sat on the mat", chunk_id=3, source="d.txt", start_token=0, end_token=10),
        ]
        self.index = BM25Index()
        self.index.build(self.chunks)

    def test_build_sets_avgdl(self):
        assert self.index._avgdl > 0

    def test_search_returns_results(self):
        results = self.index.search("BM25 scoring", top_k=2)
        assert len(results) > 0
        assert all(isinstance(r, BM25Result) for r in results)

    def test_scores_descending(self):
        results = self.index.search("BM25 scoring", top_k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_term_match_ranked_high(self):
        results = self.index.search("BM25", top_k=1)
        assert results[0].chunk_id == 1

    def test_no_match_returns_empty(self):
        results = self.index.search("xyz_nonexistent_term_abc", top_k=5)
        assert len(results) == 0

    def test_top_k_limits_results(self):
        results = self.index.search("retrieval", top_k=1)
        assert len(results) == 1

    def test_idf_for_common_term_is_low(self):
        common = "the"
        rare = "BM25"
        if common in self.index._idf and rare in self.index._idf:
            assert self.index._idf[rare] > self.index._idf[common]

    def test_score_for_nonexistent_term(self):
        score = self.index.score(["nonexistent"], 0)
        assert score == 0.0

    def test_unbuilt_index_raises(self):
        idx = BM25Index()
        with pytest.raises(RuntimeError):
            idx.search("test")

    def test_empty_corpus(self):
        idx = BM25Index()
        idx.build([])
        results = idx.search("test")
        assert len(results) == 0


class TestReciprocalRankFusion:
    def test_fusion_ranks_by_combined_score(self):
        bm25_results = [
            BM25Result(chunk_id=1, text="", source="", score=1.0, start_token=0, end_token=10),
            BM25Result(chunk_id=2, text="", source="", score=0.9, start_token=0, end_token=10),
            BM25Result(chunk_id=3, text="", source="", score=0.8, start_token=0, end_token=10),
        ]
        dense_results = [
            type("Result", (), {"chunk_id": 2})(),
            type("Result", (), {"chunk_id": 1})(),
            type("Result", (), {"chunk_id": 3})(),
        ]
        fused = reciprocal_rank_fusion(bm25_results, dense_results, k=60)
        assert fused[0]["chunk_id"] in (1, 2)
        assert len(fused) == 3

    def test_fusion_empty_lists(self):
        fused = reciprocal_rank_fusion([], [], k=60)
        assert fused == []

    def test_fusion_single_list(self):
        bm25_results = [
            BM25Result(chunk_id=1, text="", source="", score=1.0, start_token=0, end_token=10),
        ]
        fused = reciprocal_rank_fusion(bm25_results, [], k=60)
        assert len(fused) == 1
        assert fused[0]["chunk_id"] == 1