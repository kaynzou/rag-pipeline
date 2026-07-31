import pytest
from unittest.mock import MagicMock, patch
from src.chunking import TextSplitter, Chunk
from src.bm25 import BM25Index
from src.embedding import LocalEmbeddingModel, EmbeddingPipeline
from src.vector_store import VectorStore
from src.hybrid_search import HybridSearch, HybridResult
from src.reranker import CrossEncoderReranker, RerankResult
from src.generator import Generator, RAGResponse, Source


@pytest.fixture(scope="module")
def setup():
    chunks = [
        Chunk(text="RAG evaluation uses precision and recall metrics", chunk_id=0, source="eval.txt", start_token=0, end_token=10),
        Chunk(text="BM25 is a keyword scoring algorithm", chunk_id=1, source="bm25.txt", start_token=0, end_token=10),
        Chunk(text="Vector embeddings capture semantic meaning", chunk_id=2, source="emb.txt", start_token=0, end_token=10),
        Chunk(text="The cat sat on the mat", chunk_id=3, source="cat.txt", start_token=0, end_token=10),
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
        top_k=3,
        candidates_per_method=10,
    )

    reranker = CrossEncoderReranker()

    return hybrid, reranker


class TestGenerator:
    def test_generate_returns_response(self, setup):
        hybrid, reranker = setup
        query = "RAG evaluation"
        results = hybrid.search(query)
        reranked = reranker.rerank(query, results, top_k=2)

        with patch("src.generator.OpenAI") as MockClient:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "RAG evaluation uses precision and recall [0]."
            mock_client.chat.completions.create.return_value = mock_response
            MockClient.return_value = mock_client

            gen = Generator.__new__(Generator)
            gen._client = mock_client
            gen._model = "claude-3-haiku-20240307"
            gen._max_tokens = 1024
            gen._temperature = 0.0

            response = gen.generate(query, reranked)
            assert isinstance(response, RAGResponse)

    def test_generate_empty_chunks(self, setup):
        with patch("src.generator.OpenAI") as MockClient:
            MockClient.return_value = MagicMock()

            gen = Generator.__new__(Generator)
            gen._client = MockClient.return_value
            gen._model = "claude-3-haiku-20240307"
            gen._max_tokens = 1024
            gen._temperature = 0.0

            response = gen.generate("test", [])
            assert "I don't have enough information" in response.answer
            assert response.chunks_used == 0

    def test_generate_has_sources(self, setup):
        hybrid, reranker = setup
        query = "RAG evaluation"
        results = hybrid.search(query)
        reranked = reranker.rerank(query, results, top_k=2)

        with patch("src.generator.OpenAI") as MockClient:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "RAG evaluation uses precision and recall [0]."
            mock_client.chat.completions.create.return_value = mock_response
            MockClient.return_value = mock_client

            gen = Generator.__new__(Generator)
            gen._client = mock_client
            gen._model = "claude-3-haiku-20240307"
            gen._max_tokens = 1024
            gen._temperature = 0.0

            response = gen.generate(query, reranked)
            assert len(response.sources) > 0
            assert all(isinstance(s, Source) for s in response.sources)

    def test_generate_system_prompt_contains_constraints(self, setup):
        gen = Generator.__new__(Generator)
        gen._model = "claude-3-haiku-20240307"
        gen._max_tokens = 1024
        gen._temperature = 0.0

        system_prompt = gen._build_system_prompt()
        assert "ONLY the provided context" in system_prompt
        assert "citation" in system_prompt.lower()
        assert "make up" in system_prompt.lower()

    def test_generate_user_prompt_format(self, setup):
        gen = Generator.__new__(Generator)
        gen._model = "claude-3-haiku-20240307"
        gen._max_tokens = 1024
        gen._temperature = 0.0

        chunks = [
            RerankResult(
                chunk_id=0,
                text="RAG evaluation uses precision and recall",
                source="eval.txt",
                rerank_score=0.9,
                bm25_score=1.0,
                dense_score=0.5,
                rrf_score=0.03,
                start_token=0,
                end_token=10,
            )
        ]

        prompt = gen._build_user_prompt("What is RAG evaluation?", chunks)
        assert "chunk_id: 0" in prompt
        assert "What is RAG evaluation?" in prompt
        assert "RAG evaluation uses precision and recall" in prompt
