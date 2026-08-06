from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ChatResponse(RAGResponse):
    history: List[Message] = field(default_factory=list)


class ConversationalRAGPipeline(RAGPipeline):
    def __init__(self, *args, max_history: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: List[Message] = []
        self._max_history = max_history

    def chat(self, question: str, top_k: int = 5) -> ChatResponse:
        if not self._indexed:
            raise RuntimeError("Pipeline not indexed.")

        # Rewrite follow-up questions to be self-contained
        if self._history:
            question = self._rewrite_question(question)

        hybrid_results = self._hybrid.search(question)
        reranked = self._reranker.rerank(question, hybrid_results, top_k=top_k)
        response = self._generator.generate(question, reranked)

        # Add to history
        self._history.append(Message(role="user", content=question))
        self._history.append(Message(role="assistant", content=response.answer))

        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
            chunks_retrieved=len(hybrid_results),
            chunks_used=len(reranked),
            reranked_chunks=reranked,
            history=list(self._history),
        )

    def _rewrite_question(self, question: str) -> str:
        words = question.split()
        if len(words) < 10 and self._history:
            last_user_msg = self._history[-2].content if len(self._history) >= 2 else ""
            return f"Context: {last_user_msg}\nFollow-up: {question}"
        return question

    def clear_history(self) -> None:
        self._history = []

    @property
    def history(self) -> List[Message]:
        return list(self._history)
