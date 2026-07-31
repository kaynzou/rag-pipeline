import numpy as np
import pytest
from src.vector_store import VectorStore, SearchResult
from src.embedding import LocalEmbeddingModel, EmbeddedChunk


@pytest.fixture(scope="module")
def model():
    return LocalEmbeddingModel()


@pytest.fixture(scope="module")
def embedded_chunks(model):
    texts = [
        "RAG combines retrieval with generation for better answers.",
        "Vector search finds similar documents using embeddings.",
        "BM25 is a keyword scoring algorithm for information retrieval.",
        "The cat sat on the mat and looked at the rat.",
        "Quantum computing uses qubits instead of classical bits.",
        "Chunking splits documents into smaller pieces for retrieval.",
    ]
    vectors = model.embed(texts)
    chunks = []
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        chunks.append(
            EmbeddedChunk(
                chunk_id=i,
                text=text,
                source=f"doc_{i}.txt",
                start_token=0,
                end_token=10,
                embedding=vec,
            )
        )
    return chunks


@pytest.fixture()
def store(model, embedded_chunks):
    s = VectorStore(dimension=model.dimension, model_name="test-model")
    s.add(embedded_chunks)
    return s


class TestVectorStore:
    def test_add_chunks(self, model, embedded_chunks):
        store = VectorStore(dimension=model.dimension)
        store.add(embedded_chunks)
        assert store.size == 6

    def test_add_empty_list(self, model):
        store = VectorStore(dimension=model.dimension)
        store.add([])
        assert store.size == 0

    def test_add_single_chunk(self, model):
        store = VectorStore(dimension=model.dimension)
        vec = model.embed(["hello world"])[0]
        chunk = EmbeddedChunk(
            chunk_id=0, text="hello", source="a.txt", start_token=0, end_token=1, embedding=vec
        )
        store.add([chunk])
        assert store.size == 1

    def test_search_returns_results(self, store):
        model = LocalEmbeddingModel()
        query_vec = model.embed(["vector search"])[0]
        results = store.search(query_vec, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_scores_descending(self, store):
        model = LocalEmbeddingModel()
        query_vec = model.embed(["RAG evaluation"])[0]
        results = store.search(query_vec, top_k=6)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k(self, store):
        model = LocalEmbeddingModel()
        query_vec = model.embed(["test"])[0]
        results = store.search(query_vec, top_k=2)
        assert len(results) == 2

    def test_search_top_k_exceeds_size(self, store):
        model = LocalEmbeddingModel()
        query_vec = model.embed(["test"])[0]
        results = store.search(query_vec, top_k=100)
        assert len(results) == store.size

    def test_search_relevant_result_ranked_high(self, store):
        model = LocalEmbeddingModel()
        query_vec = model.embed(["BM25 keyword search"])[0]
        results = store.search(query_vec, top_k=1)
        assert "BM25" in results[0].text

    def test_search_empty_store(self, model):
        store = VectorStore(dimension=model.dimension)
        query_vec = model.embed(["test"])[0]
        with pytest.raises(ValueError):
            store.search(query_vec)

    def test_search_wrong_dimension(self, store):
        wrong_vec = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError):
            store.search(wrong_vec)

    def test_search_zero_norm_query(self, store):
        zero_vec = np.zeros(store._dimension, dtype=np.float32)
        with pytest.raises(ValueError):
            store.search(zero_vec)

    def test_save_and_load(self, store, tmp_path):
        store.save(str(tmp_path))
        loaded = VectorStore.load(str(tmp_path))
        assert loaded.size == store.size
        assert loaded._dimension == store._dimension
        assert loaded._model_name == store._model_name

        model = LocalEmbeddingModel()
        query_vec = model.embed(["vector search"])[0]
        orig_results = store.search(query_vec, top_k=3)
        load_results = loaded.search(query_vec, top_k=3)

        for r1, r2 in zip(orig_results, load_results):
            assert r1.chunk_id == r2.chunk_id
            assert np.isclose(r1.score, r2.score)

    def test_dimension_mismatch_on_add(self, model, embedded_chunks):
        store = VectorStore(dimension=999)
        with pytest.raises(ValueError):
            store.add(embedded_chunks)
    
    def test_search_raises_on_model_mismatch(self, store):
        model = LocalEmbeddingModel()
        query_vec = model.embed(["vector search"])[0]
        with pytest.raises(ValueError):
            store.search(query_vec, top_k=3, model_name="wrong-model")

    def test_search_succeeds_with_matching_model_name(self, store):
        model = LocalEmbeddingModel()
        query_vec = model.embed(["vector search"])[0]
        results = store.search(query_vec, top_k=3, model_name="test-model")
        assert len(results) == 3
