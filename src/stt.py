from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from src.generator import RAGResponse


@dataclass
class STTResult:
    text: str
    latency_ms: float
    success: bool
    error: Optional[str] = None


class SarvamSTT:
    """Sarvam Speech-to-Text client."""

    BASE_URL = "https://api.sarvam.ai"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError("SARVAM_API_KEY is required for SarvamSTT.")

    async def transcribe(self, audio_bytes: bytes, language_code: str = "en-IN") -> STTResult:
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/speech-to-text",
                    headers={"api-subscription-key": self._api_key},
                    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                    data={
                        "model": "saaras:v3",
                        "mode": "transcribe",
                        "language_code": language_code,
                    },
                )
                response.raise_for_status()
                result = response.json()

            latency = (time.perf_counter() - start) * 1000
            return STTResult(
                text=result.get("transcript", ""),
                latency_ms=round(latency, 2),
                success=True,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return STTResult(
                text="",
                latency_ms=round(latency, 2),
                success=False,
                error=str(e),
            )


class STTHarness:
    """Harness for STT with retries and fallback."""

    def __init__(self, max_retries: int = 3) -> None:
        self._max_retries = max_retries
        self._client: Optional[SarvamSTT] = None

    def _get_client(self) -> Optional[SarvamSTT]:
        if self._client is None:
            try:
                self._client = SarvamSTT()
            except ValueError:
                return None
        return self._client

    async def transcribe_with_retry(self, audio_bytes: bytes, language_code: str = "en-IN") -> STTResult:
        client = self._get_client()
        if client is None:
            return STTResult(
                text="",
                latency_ms=0.0,
                success=False,
                error="STT not configured. Set SARVAM_API_KEY.",
            )

        last_result = None
        for attempt in range(self._max_retries):
            result = await client.transcribe(audio_bytes, language_code=language_code)
            if result.success:
                return result
            last_result = result

        return last_result or STTResult(
            text="",
            latency_ms=0.0,
            success=False,
            error="STT failed after retries.",
        )
