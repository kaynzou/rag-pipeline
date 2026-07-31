from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.bm25 import BM25Index, BM25Result, reciprocal_rank_fusion
from src.vector_store import VectorStore, SearchResult


@dataclass
class HybridResult:
    chunk_id: int
    text: str
    source: str
    rrf_score: float
    bm25_score: float
    dense_score: float
    start_token: int
    end_token: int
    metadata: dict = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class HybridSearch:
    def __init__(
        self,
        bm25_index: BM25Index,
        vector_store: VectorStore,
        encoder: object,
        top_k: int = 5,
        candidates_per_method: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.encoder = encoder
        self.top_k = top_k
        self.candidates_per_method = candidates_per_method
        self.rrf_k = rrf_k

    def search(self, query: str) -> List[HybridResult]:
        bm25_results = self.bm25_index.search(query, top_k=self.candidates_per_method)

        query_vec = self.encoder.embed([query])[0]
        dense_results = self.vector_store.search( query_vec, top_k=self.candidates_per_method, model_name=getattr(self.encoder, "model_name", None),)

        fused = reciprocal_rank_fusion(bm25_results, dense_results, k=self.rrf_k)

        bm25_map = {r.chunk_id: r for r in bm25_results}
        dense_map = {r.chunk_id: r for r in dense_results}

        results: List[HybridResult] = []
        for item in fused[: self.top_k]:
            cid = item["chunk_id"]

            bm25_result = bm25_map.get(cid)
            dense_result = dense_map.get(cid)

            text = ""
            source = ""
            start_token = 0
            end_token = 0
            metadata = {}

            if bm25_result:
                text = bm25_result.text
                source = bm25_result.source
                start_token = bm25_result.start_token
                end_token = bm25_result.end_token
                metadata = bm25_result.metadata
            elif dense_result:
                text = dense_result.text
                source = dense_result.source
                start_token = dense_result.start_token
                end_token = dense_result.end_token
                metadata = dense_result.metadata

            results.append(
                HybridResult(
                    chunk_id=cid,
                    text=text,
                    source=source,
                    rrf_score=item["rrf_score"],
                    bm25_score=bm25_result.score if bm25_result else 0.0,
                    dense_score=dense_result.score if dense_result else 0.0,
                    start_token=start_token,
                    end_token=end_token,
                    metadata=metadata,
                )
            )

        return results
