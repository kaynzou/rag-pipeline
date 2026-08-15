import pytest
from src.guardrails import InputGuardrail, RetrievalGuardrail, OutputGuardrail, GuardrailResult


class TestInputGuardrail:
    def test_valid_input(self):
        guard = InputGuardrail()
        result = guard.check("What is BM25?")
        assert result.passed is True

    def test_empty_input(self):
        guard = InputGuardrail()
        result = guard.check("")
        assert result.passed is False
        assert result.category == "invalid"

    def test_too_long_input(self):
        guard = InputGuardrail()
        result = guard.check("x" * 2001)
        assert result.passed is False
        assert result.category == "invalid"

    def test_profanity_detected(self):
        guard = InputGuardrail()
        result = guard.check("This is a badword test")
        assert result.passed is False
        assert result.category == "unsafe"


class TestRetrievalGuardrail:
    def test_passes_with_results(self):
        guard = RetrievalGuardrail(min_similarity=0.1)
        result = guard.check(top_score=0.5, num_chunks=3)
        assert result.passed is True

    def test_fails_no_chunks(self):
        guard = RetrievalGuardrail()
        result = guard.check(top_score=0.0, num_chunks=0)
        assert result.passed is False
        assert result.category == "no_context"

    def test_fails_low_score(self):
        guard = RetrievalGuardrail(min_similarity=0.8)
        result = guard.check(top_score=0.3, num_chunks=3)
        assert result.passed is False
        assert result.category == "low_relevance"


class TestOutputGuardrail:
    def test_valid_answer(self):
        guard = OutputGuardrail(context="bm25 is a keyword scoring algorithm", required_facts=["bm25", "keyword"])
        result = guard.check("BM25 is a keyword scoring algorithm.")
        assert result.passed is True

    def test_speculative_answer(self):
        guard = OutputGuardrail(context="some context")
        result = guard.check("I think maybe the answer is possibly yes.")
        assert result.passed is False
        assert result.category == "hallucination"

    def test_missing_facts(self):
        guard = OutputGuardrail(context="some context", required_facts=["bm25", "keyword"])
        result = guard.check("BM25 is a retrieval algorithm.")
        assert result.passed is False
        assert result.category == "hallucination"
