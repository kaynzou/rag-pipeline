from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from src.reranker import RerankResult


@dataclass

class Source:
    chunk_id: int
    source_file: str
    text_preview: str
    rerank_score: float


@dataclass
class RAGResponse:
    answer: str
    sources: List[Source]
    chunks_retrieved: int
    chunks_used: int
    reranked_chunks: List[RerankResult]


class Generator: 
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sarvam-105b",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        relevance_threshold: float = 0.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._relevance_threshold = relevance_threshold
        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://api.sarvam.ai/v1",
            )
            if not self._client.api_key:
                raise ValueError(
                    "Sarvam API key required. Pass api_key or set SARVAM_API_KEY env var."
                )
        return self._client

    def _build_system_prompt(self) -> str:
        return """You are a helpful assistant that answers questions using ONLY the provided context.

RULES:
1. Answer the question based SOLELY on the provided context chunks.
2. If the context does not contain enough information to answer, respond with: "I don't have enough information in the provided context to answer that question."
3. Do NOT use any outside knowledge or make up information.
4. For every factual claim in your answer, add a citation like [chunk_id] at the end of the sentence.
5. Only cite chunks that directly support the specific claim.
6. Be concise and accurate.
7. If you quote text from a chunk, indicate it with quotation marks and a citation."""

    def _build_user_prompt(self, query: str, chunks: List[RerankResult]) -> str:
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"[chunk_id: {chunk.chunk_id}]\n{chunk.text}")

        context = "\n\n".join(context_parts)

        return f"""Context:
{context}

Question: {query}

Answer:"""

    def generate(
        self,
        query: str,
        chunks: List[RerankResult],
    ) -> RAGResponse:
        if not chunks:
            return RAGResponse(
                answer="I don't have enough information in the provided context to answer that question.",
                sources=[],
                chunks_retrieved=0,
                chunks_used=0,
                reranked_chunks=[],
            )

        if chunks[0].rerank_score < self._relevance_threshold:
            return RAGResponse(
                answer="I don't have enough information in the provided context to answer that question.",
                sources=[],
                chunks_retrieved=len(chunks),
                chunks_used=0,
                reranked_chunks=chunks,
            )

        top_chunk = chunks[0]
        answer = (
            f"[MOCKED — no LLM call made] Based on the most relevant chunk: "
            f"\"{top_chunk.text}\" [{top_chunk.chunk_id}]"
            )
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query, chunks)

            client = self._get_client()

            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self._max_tokens,
                reasoning_effort="low",
                temperature=self._temperature,
            )

            answer = response.choices[0].message.content.strip()
        except Exception as e:
            return self.generate_fallback(query, chunks)

        sources = [
            Source(
                chunk_id=c.chunk_id,
                source_file=c.source,
                text_preview=c.text[:100] + "..." if len(c.text) > 100 else c.text,
                rerank_score=c.rerank_score,
            )
            for c in chunks
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            chunks_retrieved=len(chunks),
            chunks_used=len(chunks),
            reranked_chunks=chunks,
        )

    def generate_fallback(self, query: str, chunks: List[RerankResult]) -> RAGResponse:
        """Return a context-grounded answer without calling an external LLM."""
        if not chunks:
            return RAGResponse(
                answer="I don't have enough information in the provided context to answer that question.",
                sources=[],
                chunks_retrieved=0,
                chunks_used=0,
                reranked_chunks=[],
            )

        lines = [f"Query: {query}", "Top retrieved context:"]
        for c in chunks[:3]:
            lines.append(f"- [{c.chunk_id}] {c.text[:220]}")
        lines.append("")
        lines.append("Answer based on retrieved context only:")
        lines.append(f"The most relevant result is chunk {chunks[0].chunk_id} with score {chunks[0].rerank_score:.3f}.")
        answer = "\n".join(lines)

        sources = [
            Source(
                chunk_id=c.chunk_id,
                source_file=c.source,
                text_preview=c.text[:100] + "..." if len(c.text) > 100 else c.text,
                rerank_score=c.rerank_score,
            )
            for c in chunks
        ]
        return RAGResponse(
            answer=answer,
            sources=sources,
            chunks_retrieved=len(chunks),
            chunks_used=len(chunks),
            reranked_chunks=chunks,
        )
