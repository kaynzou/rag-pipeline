import json
from unittest.mock import MagicMock
import pytest
from pathlib import Path
from src.chunking import TextSplitter, Chunk
from src.bm25 import BM25Index
from src.embedding import LocalEmbeddingModel, EmbeddingPipeline
from src.vector_store import VectorStore
from src.hybrid_search import HybridSearch
from src.reranker import CrossEncoderReranker
from src.generator import Generator, RAGResponse, Source
from src.pipeline import RAGPipeline
from src.eval import EvaluationHarness, LabeledExample, EvaluationReport, RetrievalMetrics, GenerationMetrics


def _make_mock_generator():
    mock_gen = MagicMock()
    mock_gen.generate.return_value = RAGResponse(
        answer="BM25 is a keyword scoring algorithm [0].",
        sources=[Source(chunk_id=0, source_file="a.txt", text_preview="BM25...", rerank_score=0.9)],
        chunks_retrieved=2,
        chunks_used=2,
        reranked_chunks=[],
    )
    return mock_gen


@pytest.fixture(scope="module")
def indexed_pipeline(tmp_path_factory):
    chunks = [
        Chunk(text="BM25 is a keyword scoring algorithm for retrieval", chunk_id=0, source="a.txt", start_token=0, end_token=10),
        Chunk(text="RAG combines retrieval with generation", chunk_id=1, source="b.txt", start_token=0, end_token=10),
        Chunk(text="Vector embeddings capture semantic meaning", chunk_id=2, source="c.txt", start_token=0, end_token=10),
        Chunk(text="Evaluation metrics measure RAG performance", chunk_id=3, source="d.txt", start_token=0, end_token=10),
        Chunk(text="The cat sat on the mat", chunk_id=4, source="e.txt", start_token=0, end_token=10),
    ]

    persist_dir = str(tmp_path_factory.mktemp("index"))
    pipeline = RAGPipeline(persist_dir=persist_dir)
    full_text = "\n\n".join(c.text for c in chunks)
    pipeline.index(full_text, source="test_corpus.txt")
    pipeline._generator = _make_mock_generator()

    return pipeline, chunks, persist_dir


class TestRAGPipeline:
    def test_index_creates_components(self, indexed_pipeline):
        pipeline, _, _ = indexed_pipeline
        assert pipeline.indexed

    def test_query_returns_response(self, indexed_pipeline):
        pipeline, _, _ = indexed_pipeline
        response = pipeline.query("BM25 scoring", top_k=2)
        assert response.answer is not None
        assert len(response.sources) <= 2

    def test_query_raises_before_index(self):
        pipeline = RAGPipeline()
        with pytest.raises(RuntimeError):
            pipeline.query("test")

    def test_save_and_load(self, indexed_pipeline, tmp_path):
        pipeline, _, persist_dir = indexed_pipeline
        new_pipeline = RAGPipeline(persist_dir=str(tmp_path))
        new_pipeline.load(persist_dir)
        new_pipeline._generator = _make_mock_generator()
        assert new_pipeline.indexed
        response = new_pipeline.query("BM25 scoring", top_k=2)
        assert response.answer is not None


class TestEvaluationHarness:
    def test_evaluate_returns_report(self, indexed_pipeline):
        pipeline, chunks, _ = indexed_pipeline

        examples = [
            LabeledExample(
                question="BM25 keyword scoring",
                answer="BM25 is a keyword scoring algorithm.",
                relevant_chunk_ids=[0],
                required_facts=["keyword scoring algorithm"],
            ),
            LabeledExample(
                question="RAG evaluation metrics",
                answer="Evaluation metrics measure RAG performance.",
                relevant_chunk_ids=[3],
                required_facts=["evaluation metrics", "RAG performance"],
            ),
        ]

        harness = EvaluationHarness(pipeline=pipeline, labeled_examples=examples, top_k=2)
        report = harness.evaluate()

        assert isinstance(report, EvaluationReport)
        assert 0.0 <= report.retrieval.precision_at_k <= 1.0
        assert 0.0 <= report.retrieval.recall_at_k <= 1.0
        assert report.retrieval.num_queries == 2

    def test_faithfulness_computation(self):
        harness = EvaluationHarness.__new__(EvaluationHarness)
        harness._top_k = 5

        score = harness._compute_faithfulness("BM25 is a keyword scoring algorithm for retrieval", ["keyword scoring algorithm", "BM25"])
        assert score == 1.0

        score = harness._compute_faithfulness("The cat sat on the mat", ["BM25"])
        assert score == 0.0

    def test_groundedness_computation(self, indexed_pipeline):
        pipeline, chunks, _ = indexed_pipeline
        response = pipeline.query("BM25", top_k=2)
        harness = EvaluationHarness.__new__(EvaluationHarness)
        harness._top_k = 5
        score = harness._compute_groundedness(response, {0, 1})
        assert 0.0 <= score <= 1.0

    def test_report_save_and_load(self, indexed_pipeline, tmp_path):
        pipeline, chunks, _ = indexed_pipeline
        examples = [LabeledExample(question="BM25", answer="BM25 is a keyword algorithm.", relevant_chunk_ids=[0])]
        harness = EvaluationHarness(pipeline=pipeline, labeled_examples=examples, top_k=2)
        report = harness.evaluate()

        save_path = str(tmp_path / "report.json")
        harness.save_report(report, save_path)
        assert Path(save_path).exists()

        with open(save_path) as f:
            data = json.load(f)
        assert "retrieval" in data
        assert "generation" in data
        assert "detailed" in data
