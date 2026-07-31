from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.pipeline import RAGPipeline, RAGResponse
from src.generator import Source
from src.server import app


@pytest.fixture()
def mock_pipeline(monkeypatch):
    pipeline = MagicMock(spec=RAGPipeline)
    pipeline.indexed = True
    pipeline.query.return_value = RAGResponse(
        answer="BM25 is a keyword scoring algorithm [0].",
        sources=[
            Source(chunk_id=0, source_file="a.txt", text_preview="BM25...", rerank_score=0.9),
            Source(chunk_id=1, source_file="b.txt", text_preview="RAG...", rerank_score=0.7),
        ],
        chunks_retrieved=10,
        chunks_used=5,
        reranked_chunks=[],
    )

    import src.server as server_mod
    monkeypatch.setattr(server_mod, "pipeline", pipeline)
    return pipeline


class TestServer:
    def test_health(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_when_indexed(self, mock_pipeline):
        client = TestClient(app)
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_ready_when_not_indexed(self, monkeypatch):
        import src.server as server_mod
        monkeypatch.setattr(server_mod, "pipeline", None)
        client = TestClient(app)
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is False

    def test_query_returns_answer(self, mock_pipeline):
        client = TestClient(app)
        response = client.post("/query", json={"question": "What is BM25?", "top_k": 2})
        assert response.status_code == 200
        data = response.json()
        assert "BM25" in data["answer"]
        assert len(data["sources"]) == 2
        assert data["latency_ms"] > 0

    def test_query_validates_input(self):
        client = TestClient(app)
        response = client.post("/query", json={"question": "", "top_k": 2})
        assert response.status_code == 422

    def test_query_not_ready(self, monkeypatch):
        import src.server as server_mod
        monkeypatch.setattr(server_mod, "pipeline", None)
        client = TestClient(app)
        response = client.post("/query", json={"question": "What is BM25?", "top_k": 2})
        assert response.status_code == 503

    def test_query_stream_returns_event(self, mock_pipeline):
        client = TestClient(app)
        response = client.post("/query/stream", json={"question": "What is BM25?", "top_k": 2})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
