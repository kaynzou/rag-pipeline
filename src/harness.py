from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.generator import RAGResponse, Source
from src.guardrails import InputGuardrail, RetrievalGuardrail, OutputGuardrail, GuardrailResult


@dataclass
class LatencyRecord:
    stt_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class HarnessResponse:
    answer: str
    sources: List[Source]
    chunks_retrieved: int
    chunks_used: int
    latency: LatencyRecord
    guardrail_passed: bool
    guardrail_reason: Optional[str] = None
    guardrail_category: Optional[str] = None
    fallback: bool = False


class LatencyTracker:
    """Tracks latency records and computes percentiles."""

    def __init__(self) -> None:
        self._records: List[LatencyRecord] = []

    def record(self, record: LatencyRecord) -> None:
        self._records.append(record)

    def percentile(self, p: float) -> float:
        if not self._records:
            return 0.0
        sorted_total = sorted(r.total_ms for r in self._records)
        k = (len(sorted_total) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(sorted_total[int(k)])
        return float(sorted_total[int(f)] * (c - k) + sorted_total[int(c)] * (k - f))

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p70(self) -> float:
        return self.percentile(70)

    @property
    def p100(self) -> float:
        return max((r.total_ms for r in self._records), default=0.0)

    @property
    def count(self) -> int:
        return len(self._records)


class RAGHarness:
    """Structured orchestration around the RAG pipeline with retries, guardrails, and latency tracking."""

    def __init__(self, pipeline, max_retries: int = 2) -> None:
        self._pipeline = pipeline
        self._max_retries = max_retries
        self._input_guardrail = InputGuardrail()
        self._retrieval_guardrail = RetrievalGuardrail()
        self._tracker = LatencyTracker()

    @property
    def latency(self) -> LatencyTracker:
        return self._tracker

    def query(self, question: str, top_k: int = 5, required_facts: Optional[List[str]] = None) -> HarnessResponse:
        total_start = time.perf_counter()

        input_check = self._input_guardrail.check(question)
        if not input_check.passed:
            record = LatencyRecord(total_ms=(time.perf_counter() - total_start) * 1000)
            self._tracker.record(record)
            return HarnessResponse(
                answer=input_check.reason or "Invalid input.",
                sources=[],
                chunks_retrieved=0,
                chunks_used=0,
                latency=record,
                guardrail_passed=False,
                guardrail_reason=input_check.reason,
                guardrail_category=input_check.category,
                fallback=True,
            )

        last_error = None
        response = None
        for attempt in range(self._max_retries):
            try:
                response = self._pipeline.query(question, top_k=top_k)
                break
            except Exception as e:
                last_error = e

        retrieval_end = time.perf_counter()
        retrieval_ms = (retrieval_end - total_start) * 1000

        if response is None:
            record = LatencyRecord(retrieval_ms=round(retrieval_ms, 2), total_ms=(time.perf_counter() - total_start) * 1000)
            self._tracker.record(record)
            return HarnessResponse(
                answer=f"Pipeline error: {last_error}",
                sources=[],
                chunks_retrieved=0,
                chunks_used=0,
                latency=record,
                guardrail_passed=False,
                guardrail_reason=str(last_error),
                fallback=True,
            )

        top_score = response.reranked_chunks[0].rerank_score if response.reranked_chunks else 0.0
        retrieval_check = self._retrieval_guardrail.check(top_score, len(response.reranked_chunks))

        generation_end = time.perf_counter()
        generation_ms = (generation_end - retrieval_end) * 1000
        total_ms = (generation_end - total_start) * 1000

        record = LatencyRecord(
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            total_ms=round(total_ms, 2),
        )
        self._tracker.record(record)

        if not retrieval_check.passed:
            return HarnessResponse(
                answer="I don't have enough information in the provided context to answer that question.",
                sources=response.sources,
                chunks_retrieved=response.chunks_retrieved,
                chunks_used=0,
                latency=record,
                guardrail_passed=False,
                guardrail_reason=retrieval_check.reason,
                guardrail_category=retrieval_check.category,
                fallback=True,
            )

        context = " ".join(c.text for c in response.reranked_chunks)
        output_check = OutputGuardrail(context=context, required_facts=required_facts).check(response.answer)

        if not output_check.passed:
            return HarnessResponse(
                answer="I don't have enough information in the provided context to answer that question.",
                sources=response.sources,
                chunks_retrieved=response.chunks_retrieved,
                chunks_used=response.chunks_used,
                latency=record,
                guardrail_passed=False,
                guardrail_reason=output_check.reason,
                guardrail_category=output_check.category,
                fallback=True,
            )

        return HarnessResponse(
            answer=response.answer,
            sources=response.sources,
            chunks_retrieved=response.chunks_retrieved,
            chunks_used=response.chunks_used,
            latency=record,
            guardrail_passed=True,
        )
