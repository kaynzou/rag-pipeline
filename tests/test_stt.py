import pytest
from src.stt import STTResult, SarvamSTT, STTHarness


class TestSTTResult:
    def test_success(self):
        r = STTResult(text="hello", latency_ms=100.0, success=True)
        assert r.text == "hello"
        assert r.success is True

    def test_failure(self):
        r = STTResult(text="", latency_ms=0.0, success=False, error="boom")
        assert r.success is False
        assert r.error == "boom"


class TestSTTHarness:
    def test_missing_api_key_returns_failure(self):
        harness = STTHarness()
        import asyncio
        result = asyncio.run(harness.transcribe_with_retry(b"audio"))
        assert result.success is False
        assert "SARVAM_API_KEY" in result.error
