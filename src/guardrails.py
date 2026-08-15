from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GuardrailResult:
    passed: bool
    reason: Optional[str] = None
    category: Optional[str] = None


class InputGuardrail:
    """Validates user input before retrieval/generation."""

    _PROFANITY_PATTERNS = re.compile(
        r'\b(?:badword|damn|hell|shit|fuck|ass|bitch|crap|dick|piss)\b',
        re.IGNORECASE,
    )

    def check(self, text: str) -> GuardrailResult:
        if not text or not text.strip():
            return GuardrailResult(passed=False, reason="Empty input", category="invalid")

        if len(text) > 2000:
            return GuardrailResult(passed=False, reason="Input too long", category="invalid")

        if self._PROFANITY_PATTERNS.search(text):
            return GuardrailResult(passed=False, reason="Inappropriate content detected", category="unsafe")

        return GuardrailResult(passed=True)


class RetrievalGuardrail:
    """Validates retrieval results before generation."""

    def __init__(self, min_similarity: float = 0.25) -> None:
        self.min_similarity = min_similarity

    def check(self, top_score: float, num_chunks: int) -> GuardrailResult:
        if num_chunks == 0:
            return GuardrailResult(passed=False, reason="No relevant context found", category="no_context")
        if top_score < self.min_similarity:
            return GuardrailResult(passed=False, reason="Top result below relevance threshold", category="low_relevance")
        return GuardrailResult(passed=True)


class OutputGuardrail:
    """Validates generated answer for grounding and safety."""

    def __init__(self, context: str, required_facts: Optional[List[str]] = None) -> None:
        self.context = context.lower()
        self.required_facts = [f.lower() for f in (required_facts or [])]

    def check(self, answer: str) -> GuardrailResult:
        if not answer or not answer.strip():
            return GuardrailResult(passed=False, reason="Empty answer", category="invalid")

        lowered = answer.lower()
        if any(marker in lowered for marker in ["i think", "maybe", "possibly", "unfortunately", "as an ai"]):
            return GuardrailResult(passed=False, reason="Answer appears speculative or conversational", category="hallucination")

        if self.required_facts:
            missing = [f for f in self.required_facts if f not in lowered]
            if missing:
                return GuardrailResult(passed=False, reason=f"Missing required facts: {missing}", category="hallucination")

        return GuardrailResult(passed=True)
