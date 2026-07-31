from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.pipeline import RAGPipeline


@dataclass
class LabeledExample:
    question: str
    answer: str
    relevant_chunk_ids: List[int]
    required_facts: List[str] = None

    def __post_init__(self) -> None:
        if self.required_facts is None:
            self.required_facts = []


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    num_queries: int


@dataclass
class GenerationMetrics:
    faithfulness: float
    groundedness: float
    num_queries: int


@dataclass
class EvaluationReport:
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    detailed: List[dict]


class EvaluationHarness:
    def __init__(
        self,
        pipeline: RAGPipeline,
        labeled_examples: List[LabeledExample],
        top_k: int = 5,
    ) -> None:
        self._pipeline = pipeline
        self._labeled_examples = labeled_examples
        self._top_k = top_k

    def evaluate(self) -> EvaluationReport:
        detailed: List[dict] = []

        total_precision = 0.0
        total_recall = 0.0
        total_mrr = 0.0
        total_faithfulness = 0.0
        total_groundedness = 0.0
        num_with_results = 0
        num_with_generation = 0

        for example in self._labeled_examples:
            response = self._pipeline.query(example.question, top_k=self._top_k)

            retrieved_ids = [c.chunk_id for c in response.reranked_chunks]
            relevant_ids = set(example.relevant_chunk_ids)
            retrieved_set = set(retrieved_ids)

            hits = len(retrieved_set & relevant_ids)
            precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
            recall = hits / len(relevant_ids) if relevant_ids else 0.0

            mrr = 0.0
            for rank, chunk_id in enumerate(retrieved_ids, 1):
                if chunk_id in relevant_ids:
                    mrr = 1.0 / rank
                    break

            faithfulness = self._compute_faithfulness(response.answer, example.required_facts)
            groundedness = self._compute_groundedness(response, relevant_ids)

            total_precision += precision
            total_recall += recall
            total_mrr += mrr
            total_faithfulness += faithfulness
            total_groundedness += groundedness
            num_with_results += 1
            num_with_generation += 1

            detailed.append(
                {
                    "question": example.question,
                    "answer": response.answer,
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": list(relevant_ids),
                    "precision@k": precision,
                    "recall@k": recall,
                    "mrr": mrr,
                    "faithfulness": faithfulness,
                    "groundedness": groundedness,
                    "sources": [
                        {"chunk_id": s.chunk_id, "source_file": s.source_file}
                        for s in response.sources
                    ],
                }
            )

        n = num_with_results if num_with_results > 0 else 1
        ng = num_with_generation if num_with_generation > 0 else 1

        return EvaluationReport(
            retrieval=RetrievalMetrics(
                precision_at_k=total_precision / n,
                recall_at_k=total_recall / n,
                mrr=total_mrr / n,
                num_queries=num_with_results,
            ),
            generation=GenerationMetrics(
                faithfulness=total_faithfulness / ng,
                groundedness=total_groundedness / ng,
                num_queries=num_with_generation,
            ),
            detailed=detailed,
        )

    def _compute_faithfulness(self, answer: str, required_facts: List[str]) -> float:
        if not required_facts:
            return 1.0

        answer_lower = answer.lower()
        found = sum(1 for fact in required_facts if fact.lower() in answer_lower)
        return found / len(required_facts)

    def _compute_groundedness(self, response: "RAGResponse", relevant_ids: set) -> float:
        if not response.reranked_chunks:
            return 0.0

        grounded = sum(1 for c in response.reranked_chunks if c.chunk_id in relevant_ids)
        return grounded / len(response.reranked_chunks)

    def save_report(self, report: EvaluationReport, path: str) -> None:
        data = {
            "retrieval": {
                "precision@k": report.retrieval.precision_at_k,
                "recall@k": report.retrieval.recall_at_k,
                "mrr": report.retrieval.mrr,
                "num_queries": report.retrieval.num_queries,
            },
            "generation": {
                "faithfulness": report.generation.faithfulness,
                "groundedness": report.generation.groundedness,
                "num_queries": report.generation.num_queries,
            },
            "detailed": report.detailed,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
