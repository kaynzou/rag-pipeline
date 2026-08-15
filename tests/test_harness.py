from unittest.mock import Mock
import pytest
from src.harness import LatencyTracker, RAGHarness, LatencyRecord
from src.pipeline import RAGPipeline, RAGResponse
from src.generator import Source
from src.reranker import RerankResult


class TestLatencyTracker:
    def test_empty_tracker(self):
        tracker = LatencyTracker()
        assert tracker.p50 == 0.0
        assert tracker.p70 == 0.0
        assert tracker.p100 == 0.0
        assert tracker.count == 0

    def test_percentiles(self):
        tracker = LatencyTracker()
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            tracker.record(LatencyRecord(total_ms=ms))
        assert tracker.p50 == 55.0
        assert tracker.p70 == 73.0
        assert tracker.p100 == 100.0
        assert tracker.count == 10


class TestRAGHarness:
    def test_harness_with_mock_pipeline(self):
        pipeline = Mock(spec=RAGPipeline)
        pipeline.indexed = True
        reranked = [
            RerankResult(chunk_id=0, text="BM25 is a keyword scoring algorithm", source="a.txt", rerank_score=0.9,
                         bm25_score=1.0, dense_score=0.8, rrf_score=0.5, start_token=0, end_token=8),
        ]
        pipeline.query.return_value = RAGResponse(
            answer="BM25 is a keyword scoring algorithm [0].",
            sources=[Source(chunk_id=0, source_file="a.txt", text_preview="BM25...", rerank_score=0.9)],
            chunks_retrieved=2,
            chunks_used=1,
            reranked_chunks=reranked,
        )

        harness = RAGHarness(pipeline=pipeline, max_retries=1)
        result = harness.query("What is BM25?", top_k=2)

        assert result.answer is not None
        assert result.guardrail_passed is True
        assert result.latency.total_ms > 0
        assert harness.latency.count == 1

    def test_harness_input_guardrail_blocks_empty(self):
        pipeline = Mock(spec=RAGPipeline)
        harness = RAGHarness(pipeline=pipeline, max_retries=1)
        result = harness.query("", top_k=2)
        assert result.guardrail_passed is False
        assert result.fallback is True
        assert result.guardrail_category == "invalid"

    def test_harness_retry_on_failure(self):
        pipeline = Mock(spec=RAGPipeline)
        pipeline.indexed = True
        reranked = [
            RerankResult(chunk_id=0, text="ok", source="a.txt", rerank_score=0.9,
                         bm25_score=1.0, dense_score=0.8, rrf_score=0.5, start_token=0, end_token=1),
        ]
        pipeline.query.side_effect = [RuntimeError("fail"), RAGResponse(
            answer="ok", sources=[], chunks_retrieved=0, chunks_used=0, reranked_chunks=reranked
        )]

        harness = RAGHarness(pipeline=pipeline, max_retries=2)
        result = harness.query("test", top_k=2)
        assert result.answer == "ok"
        assert pipeline.query.call_count == 2
