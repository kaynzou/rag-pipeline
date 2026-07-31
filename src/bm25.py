from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.chunking import Chunk


@dataclass
class BM25Result:
    chunk_id: int
    text: str
    source: str
    score: float
    start_token: int
    end_token: int
    metadata: dict = field(default_factory=dict)


class BM25Index:
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Optional[object] = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self._doc_tokens: List[List[str]] = []
        self._doc_freqs: List[Dict[str, int]] = []
        self._doc_lengths: List[int] = []
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self._chunks: List[Chunk] = []
        self._built: bool = False

    def _tokenize(self, text: str) -> List[str]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        return self._simple_tokenize(text)

    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        return text.lower().split()

    def build(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        self._doc_tokens = []
        self._doc_freqs = []
        self._doc_lengths = []

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self._doc_tokens.append(tokens)
            self._doc_lengths.append(len(tokens))

            freq: Dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self._doc_freqs.append(freq)

        N = len(chunks)
        if N == 0:
            self._avgdl = 0.0
            self._idf = {}
            self._built = True
            return

        self._avgdl = sum(self._doc_lengths) / N

        doc_freq: Dict[str, int] = {}
        for freq in self._doc_freqs:
            for token in freq:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self._idf = {}
        for token, df in doc_freq.items():
            self._idf[token] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        self._built = True

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        score = 0.0
        doc_len = self._doc_lengths[doc_idx]
        freq = self._doc_freqs[doc_idx]

        for token in query_tokens:
            if token not in self._idf:
                continue
            tf = freq.get(token, 0)
            if tf == 0:
                continue
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
            score += self._idf[token] * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 5) -> List[BM25Result]:
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        query_tokens = self._tokenize(query)
        scores = []

        for i in range(len(self._chunks)):
            s = self.score(query_tokens, i)
            if s > 0:
                scores.append((i, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = min(top_k, len(scores))

        results: List[BM25Result] = []
        for idx, score in scores[:top_k]:
            chunk = self._chunks[idx]
            results.append(
                BM25Result(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    score=score,
                    start_token=chunk.start_token,
                    end_token=chunk.end_token,
                    metadata=chunk.metadata,
                )
            )

        return results


def reciprocal_rank_fusion(
    bm25_results: List[BM25Result],
    dense_results: List[object],
    k: int = 60,
) -> List[dict]:
    scores: Dict[int, float] = {}

    for rank, r in enumerate(bm25_results, 1):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)

    for rank, r in enumerate(dense_results, 1):
        cid = r.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    return [
        {"chunk_id": cid, "rrf_score": scores[cid]}
        for cid in sorted_ids
    ]