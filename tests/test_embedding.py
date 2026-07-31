import numpy as np
import pytest
from src.embedding import (
    LocalEmbeddingModel,
    cosine_similarity,
    EmbeddedChunk,
    EmbeddingPipeline,
)


class TestLocalEmbeddingModel:
    def test_embed_shape(self):
        model = LocalEmbeddingModel()
        vectors = model.embed(["hello world", "test sentence"])
        assert vectors.shape == (2, model.dimension)

    def test_dimension(self):
        model = LocalEmbeddingModel()
        assert model.dimension == 384

    def test_similar_texts_are_close(self):
        model = LocalEmbeddingModel()
        texts = [
            "The cat sat on the mat.",
            "A feline was resting on a rug.",
            "Quantum computing uses qubits.",
        ]
        vectors = model.embed(texts)
        sims = cosine_similarity(vectors[0:1], vectors[1:2])
        assert sims[0][0] > 0.5

    def test_different_texts_are_far(self):
        model = LocalEmbeddingModel()
        texts = [
            "The cat sat on the mat.",
            "Quantum computing uses qubits.",
        ]
        vectors = model.embed(texts)
        sims = cosine_similarity(vectors[0:1], vectors[1:2])
        assert sims[0][0] < 0.3


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([[1.0, 0.0, 0.0]])
        sim = cosine_similarity(v, v)
        assert np.isclose(sim[0][0], 1.0)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        sim = cosine_similarity(a, b)
        assert np.isclose(sim[0][0], 0.0)

    def test_opposite_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[-1.0, 0.0]])
        sim = cosine_similarity(a, b)
        assert np.isclose(sim[0][0], -1.0)


class TestEmbeddedChunk:
    def test_create(self):
        chunk = EmbeddedChunk(
            chunk_id=1,
            text="hello",
            source="test.txt",
            start_token=0,
            end_token=10,
            embedding=np.array([0.1, 0.2, 0.3]),
        )
        assert chunk.chunk_id == 1
        assert chunk.embedding.shape == (3,)
