from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.embedding import EmbeddedChunk


@dataclass
class SearchResult:
    chunk_id: int
    text: str
    source: str
    score: float
    start_token: int
    end_token: int
    metadata: dict = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class VectorStore:
    def __init__(self, dimension: int, model_name: str = "unknown") -> None:
        self._dimension = dimension
        self._model_name = model_name
        self._vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._chunks: List[dict] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunks: List[EmbeddedChunk]) -> None:
        if not chunks:
            return

        vectors = np.array([c.embedding for c in chunks], dtype=np.float32)

        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension {vectors.shape[1]} != store dimension {self._dimension}"
            )

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vectors = vectors / norms

        self._vectors = np.vstack([self._vectors, vectors]) if self.size > 0 else vectors

        for chunk in chunks:
            self._chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "start_token": chunk.start_token,
                    "end_token": chunk.end_token,
                    "metadata": chunk.metadata,
                }
            )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        model_name: Optional[str] = None,
    ) -> List[SearchResult]:
        if self.size == 0:
            raise ValueError("Vector store is empty. Add chunks before searching.")

        if model_name is not None and model_name != self._model_name:
            raise ValueError(
                f"Query embedded with '{model_name}' but store was built with "
                f"'{self._model_name}'. Re-embed with the same model."
            )

        query_vector = np.asarray(query_vector, dtype=np.float32).flatten()

        if query_vector.shape[0] != self._dimension:
            raise ValueError(
                f"Query dimension {query_vector.shape[0]} != store dimension {self._dimension}"
            )

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query vector has zero norm — cannot compute similarity.")
        query_normalized = query_vector / query_norm

        scores = self._vectors @ query_normalized

        k = min(top_k, self.size)
        if k == self.size:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: List[SearchResult] = []
        for idx in top_indices:
            chunk_data = self._chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk_data["chunk_id"],
                    text=chunk_data["text"],
                    source=chunk_data["source"],
                    score=float(scores[idx]),
                    start_token=chunk_data["start_token"],
                    end_token=chunk_data["end_token"],
                    metadata=chunk_data["metadata"],
                )
            )

        return results

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "vectors.npy", self._vectors)

        with open(path / "chunks.json", "w") as f:
            json.dump(self._chunks, f, indent=2)

        with open(path / "config.json", "w") as f:
            json.dump({"dimension": self._dimension, "model_name": self._model_name}, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> VectorStore:
        path = Path(directory)

        with open(path / "config.json") as f:
            config = json.load(f)

        store = cls(dimension=config["dimension"], model_name=config["model_name"])
        store._vectors = np.load(path / "vectors.npy").astype(np.float32)

        with open(path / "chunks.json") as f:
            store._chunks = json.load(f)

        return store