"""
Conversational RAG with message history.

Extends the base RAG pipeline to support multi-turn conversations
by maintaining chat history and incorporating it into retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ChatResponse(RAGResponse):
    """Extended response with chat history."""
    history: List[Message] = field(default_factory=list)


class ConversationalRAGPipeline(RAGPipeline):
    """
    RAG pipeline with conversational memory.

    Maintains a history of user messages and assistant responses.
    Follow-up questions are rewritten to be self-contained before retrieval.
    """

    def __init__(self, *args, max_history: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: List[Message] = []
        self._max_history = max_history

    def chat(self, question: str, top_k: int = 5) -> ChatResponse:
        """
        Have a conversation turn.

        Args:
            question: The user's question (can be a follow-up).
            top_k: Number of results to retrieve.

        Returns:
            ChatResponse with answer and updated history.
        """
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
        """
        Rewrite follow-up questions to be self-contained.

        Simple heuristic: if the question is short (< 10 words) and the history
        contains a previous question, append context from the last exchange.
        """
        words = question.split()
        if len(words) < 10 and self._history:
            last_user_msg = self._history[-2].content if len(self._history) >= 2 else ""
            return f"Context: {last_user_msg}\nFollow-up: {question}"
        return question

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history = []

    @property
    def history(self) -> List[Message]:
        """Current conversation history."""
        return list(self._history)
