from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse

from src.pipeline import RAGPipeline
from src.harness import RAGHarness, LatencyTracker
from src.stt import STTHarness

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceResponse(BaseModel):
    chunk_id: int
    source_file: str
    text_preview: str
    rerank_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    chunks_retrieved: int
    chunks_used: int
    latency_ms: float
    guardrail_passed: bool
    guardrail_reason: Optional[str] = None
    guardrail_category: Optional[str] = None
    fallback: bool = False


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    indexed: bool


class LatencyResponse(BaseModel):
    p50_ms: float
    p70_ms: float
    p100_ms: float
    num_queries: int


class IndexRequest(BaseModel):
    source: str = "sample_corpus"
    strategies: Optional[list[str]] = None


class IndexResponse(BaseModel):
    indexed: bool
    message: str


class VoiceQueryResponse(BaseModel):
    transcript: str
    stt_latency_ms: float
    answer: str
    sources: list[SourceResponse]
    chunks_retrieved: int
    chunks_used: int
    latency_ms: float
    guardrail_passed: bool
    guardrail_reason: Optional[str] = None
    guardrail_category: Optional[str] = None
    fallback: bool = False


pipeline: Optional[RAGPipeline] = None
harness: Optional[RAGHarness] = None
stt_harness: Optional[STTHarness] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, harness, stt_harness
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    try:
        pipeline = RAGPipeline(openai_api_key=openai_key, anthropic_api_key=anthropic_key)
        index_dir = Path(__file__).parent.parent / "data" / "index"
        if index_dir.exists():
            pipeline.load(str(index_dir))
            logger.info("Pipeline loaded from %s", index_dir)
        harness = RAGHarness(pipeline=pipeline)
        stt_harness = STTHarness()
    except ImportError as e:
        logger.warning("Pipeline not initialized: %s", e)
        pipeline = None
    yield
    pipeline = None
    harness = None
    stt_harness = None


app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation pipeline built from scratch.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready():
    return ReadyResponse(ready=pipeline is not None and pipeline.indexed, indexed=pipeline.indexed if pipeline else False)


@app.get("/latency", response_model=LatencyResponse)
def latency():
    if harness is None:
        raise HTTPException(status_code=503, detail="Harness not initialized.")
    tracker: LatencyTracker = harness.latency
    return LatencyResponse(
        p50_ms=round(tracker.p50, 2),
        p70_ms=round(tracker.p70, 2),
        p100_ms=round(tracker.p100, 2),
        num_queries=tracker.count,
    )


@app.post("/index", response_model=IndexResponse)
def index_corpus(request: IndexRequest):
    global pipeline, harness
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    if pipeline.indexed:
        return IndexResponse(indexed=True, message="Already indexed.")

    corpus_path = Path(__file__).parent.parent / "data" / "sample_corpus.txt"
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus not found at {corpus_path}")

    try:
        text = corpus_path.read_text(encoding="utf-8")
        if request.strategies:
            pipeline.index_with_strategies(text, source=request.source, strategies=request.strategies)
        else:
            pipeline.index(text, source=request.source)
        logger.info("Indexed corpus from %s", corpus_path)
        harness = RAGHarness(pipeline=pipeline)
        return IndexResponse(indexed=True, message="Corpus indexed successfully.")
    except Exception as e:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if pipeline is None or not pipeline.indexed or harness is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready. Index not loaded.")

    result = harness.query(request.question, top_k=request.top_k)

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                chunk_id=s.chunk_id,
                source_file=s.source_file,
                text_preview=s.text_preview,
                rerank_score=s.rerank_score,
            )
            for s in result.sources
        ],
        chunks_retrieved=result.chunks_retrieved,
        chunks_used=result.chunks_used,
        latency_ms=round(result.latency.total_ms, 2),
        guardrail_passed=result.guardrail_passed,
        guardrail_reason=result.guardrail_reason,
        guardrail_category=result.guardrail_category,
        fallback=result.fallback,
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    if pipeline is None or not pipeline.indexed:
        raise HTTPException(status_code=503, detail="Pipeline not ready. Index not loaded.")

    start = time.perf_counter()

    def event_generator():
        try:
            result = harness.query(request.question, top_k=request.top_k) if harness else None
            if result is None:
                raise RuntimeError("Harness not available.")

            latency_ms = (time.perf_counter() - start) * 1000

            payload = {
                "answer": result.answer,
                "sources": [
                    {
                        "chunk_id": s.chunk_id,
                        "source_file": s.source_file,
                        "text_preview": s.text_preview,
                        "rerank_score": s.rerank_score,
                    }
                    for s in result.sources
                ],
                "chunks_retrieved": result.chunks_retrieved,
                "chunks_used": result.chunks_used,
                "latency_ms": round(latency_ms, 2),
                "guardrail_passed": result.guardrail_passed,
                "guardrail_reason": result.guardrail_reason,
                "guardrail_category": result.guardrail_category,
                "fallback": result.fallback,
            }

            yield f"data: {json.dumps(payload)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logger.exception("Streaming error")
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/voice/query", response_model=VoiceQueryResponse)
async def voice_query(file: UploadFile = File(...), top_k: int = 5):
    if pipeline is None or not pipeline.indexed or harness is None or stt_harness is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    audio_bytes = await file.read()
    stt_result = await stt_harness.transcribe_with_retry(audio_bytes, language_code="hi-IN")

    if not stt_result.success:
        raise HTTPException(status_code=500, detail=f"STT failed: {stt_result.error}")

    result = harness.query(stt_result.text, top_k=top_k)

    return VoiceQueryResponse(
        transcript=stt_result.text,
        stt_latency_ms=stt_result.latency_ms,
        answer=result.answer,
        sources=[
            SourceResponse(
                chunk_id=s.chunk_id,
                source_file=s.source_file,
                text_preview=s.text_preview,
                rerank_score=s.rerank_score,
            )
            for s in result.sources
        ],
        chunks_retrieved=result.chunks_retrieved,
        chunks_used=result.chunks_used,
        latency_ms=round(result.latency.total_ms, 2),
        guardrail_passed=result.guardrail_passed,
        guardrail_reason=result.guardrail_reason,
        guardrail_category=result.guardrail_category,
        fallback=result.fallback,
    )


@app.post("/voice/query-debug")
async def voice_query_debug(file: UploadFile = File(...), top_k: int = 5):
    if pipeline is None or not pipeline.indexed or harness is None or stt_harness is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    audio_bytes = await file.read()
    stt_result = await stt_harness.transcribe_with_retry(audio_bytes, language_code="hi-IN")

    if not stt_result.success:
        return {
            "transcript": "",
            "stt_latency_ms": stt_result.latency_ms,
            "stt_error": stt_result.error,
            "answer": f"STT failed: {stt_result.error}",
            "sources": [],
            "chunks_retrieved": 0,
            "chunks_used": 0,
            "latency_ms": 0.0,
            "guardrail_passed": False,
            "fallback": True,
        }

    try:
        result = harness.query(stt_result.text, top_k=top_k)
    except Exception as e:
        return {
            "transcript": stt_result.text,
            "stt_latency_ms": stt_result.latency_ms,
            "stt_error": None,
            "answer": f"RAG error: {e}",
            "sources": [],
            "chunks_retrieved": 0,
            "chunks_used": 0,
            "latency_ms": 0.0,
            "guardrail_passed": False,
            "fallback": True,
        }

    return {
        "transcript": stt_result.text,
        "stt_latency_ms": stt_result.latency_ms,
        "stt_error": None,
        "answer": result.answer,
        "sources": [
            {
                "chunk_id": s.chunk_id,
                "source_file": s.source_file,
                "text_preview": s.text_preview,
                "rerank_score": s.rerank_score,
            }
            for s in result.sources
        ],
        "chunks_retrieved": result.chunks_retrieved,
        "chunks_used": result.chunks_used,
        "latency_ms": round(result.latency.total_ms, 2),
        "guardrail_passed": result.guardrail_passed,
        "guardrail_reason": result.guardrail_reason,
        "guardrail_category": result.guardrail_category,
        "fallback": result.fallback,
    }


@app.get("/")
def root():
    return {
        "message": "RAG Pipeline API is running",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "latency": "/latency",
        "voice": "/voice/query",
    }
