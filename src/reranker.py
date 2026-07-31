from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from src.hybrid_search import HybridResult


@dataclass
class RerankResult:
    chunk_id: int
    text: str
    source: str
    rerank_score: float
    bm25_score: float
    dense_score: float
    rrf_score: float
    start_token: int
    end_token: int
    metadata: dict = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: List[HybridResult], top_k: int = 5) -> List[RerankResult]:
        pass


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )
        self._model = CrossEncoder(model_name)
        self._model_name = model_name

    def rerank(self, query: str, results: List[HybridResult], top_k: int = 5) -> List[RerankResult]:
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)

        scored = [(r, float(s)) for r, s in zip(results, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked: List[RerankResult] = []
        for result, score in scored[:top_k]:
            reranked.append(
                RerankResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    source=result.source,
                    rerank_score=score,
                    bm25_score=result.bm25_score,
                    dense_score=result.dense_score,
                    rrf_score=result.rrf_score,
                    start_token=result.start_token,
                    end_token=result.end_token,
                    metadata=result.metadata,
                )
            )

        return reranked


class LLMReranker(Reranker):
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307") -> None:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        self._client = Anthropic(api_key=api_key)
        self._model = model

    def rerank(self, query: str, results: List[HybridResult], top_k: int = 5) -> List[RerankResult]:
        if not results:
            return []

        scored: List[tuple] = []
        for result in results:
            prompt = self._build_prompt(query, result.text)
            score = self._get_llm_score(prompt)
            scored.append((result, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        reranked: List[RerankResult] = []
        for result, score in scored[:top_k]:
            reranked.append(
                RerankResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    source=result.source,
                    rerank_score=score,
                    bm25_score=result.bm25_score,
                    dense_score=result.dense_score,
                    rrf_score=result.rrf_score,
                    start_token=result.start_token,
                    end_token=result.end_token,
                    metadata=result.metadata,
                )
            )

        return reranked

    def _build_prompt(self, query: str, document: str) -> str:
        return (
            f"Rate the relevance of the following document to the query on a scale of 0.0 to 1.0.\n\n"
            f"Query: {query}\n\n"
            f"Document: {document}\n\n"
            f"Relevance score (0.0-1.0):"
        )

    def _get_llm_score(self, prompt: str) -> float:
        try:
            response = self._client.messages.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            text = response.content[0].text.strip()
            return float(text)
        except Exception:
            return 0.0