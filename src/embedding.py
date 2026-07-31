from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class EmbeddedChunk:
    chunk_id: int
    text: str
    source: str
    start_token: int
    end_token: int
    embedding: np.ndarray
    metadata: dict = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package required. Install with: pip install openai")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension = dimensions

    def embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            dimensions=self._dimension,
        )
        vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        return np.vstack(vectors)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.model


class LocalEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_embedding_dimension()
        self._model_name = model_name

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)

    a_norm = np.maximum(a_norm, 1e-8)
    b_norm = np.maximum(b_norm, 1e-8)

    dot = np.sum(a * b, axis=-1, keepdims=True)
    return dot / (a_norm * b_norm)


class EmbeddingPipeline:
    def __init__(self, model: Optional[EmbeddingModel] = None) -> None:
        if model is None:
            try:
                from src.embedding import LocalEmbeddingModel
                model = LocalEmbeddingModel()
            except ImportError:
                raise ImportError(
                    "No embedding model available. Either install sentence-transformers "
                    "(`pip install -e '.[local]'`) or pass an EmbeddingModel explicitly."
                )
        self.model = model

    def embed_chunks(self, chunks: List) -> List[EmbeddedChunk]:
        texts = [c.text for c in chunks]
        vectors = self.model.embed(texts)

        embedded: List[EmbeddedChunk] = []
        for chunk, vector in zip(chunks, vectors):
            embedded.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    start_token=chunk.start_token,
                    end_token=chunk.end_token,
                    embedding=vector,
                    metadata=chunk.metadata,
                )
            )

        return embedded
