"""
Latency benchmark for the RAG pipeline.

Measures P50 / P70 / P100 across a set of queries and prints a report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.pipeline import RAGPipeline
from src.harness import RAGHarness, LatencyTracker


@dataclass
class BenchmarkResult:
    query: str
    answer: str
    latency_ms: float
    guardrail_passed: bool


def run_benchmark(
    persist_dir: str = "data/index",
    queries: List[str] | None = None,
    top_k: int = 5,
    num_runs: int = 20,
) -> dict:
    if queries is None:
        queries = [
            "What is BM25?",
            "How does RAG work?",
            "What are embeddings?",
            "Explain hybrid search.",
            "What is a cross-encoder?",
            "How do you evaluate RAG?",
            "What is chunking?",
            "What is a vector store?",
            "What is reranking?",
            "What is the purpose of a generator in RAG?",
        ]

    pipeline = RAGPipeline(persist_dir=persist_dir)
    if not pipeline.indexed:
        raise RuntimeError(f"Pipeline not indexed at {persist_dir}")

    harness = RAGHarness(pipeline=pipeline)
    tracker = harness.latency

    results: List[BenchmarkResult] = []
    for i in range(num_runs):
        q = queries[i % len(queries)]
        start = time.perf_counter()
        result = harness.query(q, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        results.append(BenchmarkResult(
            query=q,
            answer=result.answer[:100],
            latency_ms=round(elapsed, 2),
            guardrail_passed=result.guardrail_passed,
        ))

    report = {
        "num_queries": num_runs,
        "p50_ms": round(tracker.p50, 2),
        "p70_ms": round(tracker.p70, 2),
        "p100_ms": round(tracker.p100, 2),
        "results": [
            {
                "query": r.query,
                "answer": r.answer,
                "latency_ms": r.latency_ms,
                "guardrail_passed": r.guardrail_passed,
            }
            for r in results
        ],
    }
    return report


def main():
    report = run_benchmark()
    print(json.dumps(report, indent=2))

    out_path = Path("data/latency_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
