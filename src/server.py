from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse

from src.pipeline import RAGPipeline

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


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    indexed: bool


pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    try:
        pipeline = RAGPipeline(openai_api_key=openai_key, anthropic_api_key=anthropic_key)
        index_dir = Path(__file__).parent.parent / "data" / "index"
        if index_dir.exists():
            pipeline.load(str(index_dir))
            logger.info("Pipeline loaded from %s", index_dir)
    except ImportError as e:
        logger.warning("Pipeline not initialized: %s", e)
        pipeline = None
    yield
    pipeline = None


app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation pipeline built from scratch.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready():
    return ReadyResponse(ready=pipeline is not None and pipeline.indexed, indexed=pipeline.indexed if pipeline else False)


class IndexRequest(BaseModel):
    source: str = "sample_corpus"


class IndexResponse(BaseModel):
    indexed: bool
    message: str


@app.post("/index", response_model=IndexResponse)
def index_corpus(request: IndexRequest):
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    if pipeline.indexed:
        return IndexResponse(indexed=True, message="Already indexed.")

    corpus_path = Path(__file__).parent.parent / "data" / "sample_corpus.txt"
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus not found at {corpus_path}")

    try:
        text = corpus_path.read_text(encoding="utf-8")
        pipeline.index(text, source=request.source)
        logger.info("Indexed corpus from %s", corpus_path)
        return IndexResponse(indexed=True, message="Corpus indexed successfully.")
    except Exception as e:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


class IndexRequest(BaseModel):
    source: str = "sample_corpus"


class IndexResponse(BaseModel):
    indexed: bool
    message: str


@app.post("/index", response_model=IndexResponse)
def index_corpus(request: IndexRequest):
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    if pipeline.indexed:
        return IndexResponse(indexed=True, message="Already indexed.")

    corpus_path = Path(__file__).parent.parent / "data" / "sample_corpus.txt"
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus not found at {corpus_path}")

    try:
        text = corpus_path.read_text(encoding="utf-8")
        pipeline.index(text, source=request.source)
        logger.info("Indexed corpus from %s", corpus_path)
        return IndexResponse(indexed=True, message="Corpus indexed successfully.")
    except Exception as e:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if pipeline is None or not pipeline.indexed:
        raise HTTPException(status_code=503, detail="Pipeline not ready. Index not loaded.")

    start = time.perf_counter()
    try:
        response = pipeline.query(request.question, top_k=request.top_k)
    except ValueError as e:
        logger.exception("Pipeline configuration or data error")
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail="Internal server error")

    latency_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=response.answer,
        sources=[
            SourceResponse(
                chunk_id=s.chunk_id,
                source_file=s.source_file,
                text_preview=s.text_preview,
                rerank_score=s.rerank_score,
            )
            for s in response.sources
        ],
        chunks_retrieved=response.chunks_retrieved,
        chunks_used=response.chunks_used,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    if pipeline is None or not pipeline.indexed:
        raise HTTPException(status_code=503, detail="Pipeline not ready. Index not loaded.")

    start = time.perf_counter()

    def event_generator():
        try:
            if pipeline._reranker is None:
                from src.reranker import CrossEncoderReranker
                pipeline._reranker = CrossEncoderReranker()

            if pipeline._generator is None:
                pipeline._generator = Generator(api_key=pipeline._openai_api_key, model=pipeline._claude_model)

            hybrid_results = pipeline._hybrid.search(request.question)
            reranked = pipeline._reranker.rerank(request.question, hybrid_results, top_k=request.top_k)
            response = pipeline._generator.generate(request.question, reranked)

            latency_ms = (time.perf_counter() - start) * 1000

            payload = {
                "answer": response.answer,
                "sources": [
                    {
                        "chunk_id": s.chunk_id,
                        "source_file": s.source_file,
                        "text_preview": s.text_preview,
                        "rerank_score": s.rerank_score,
                    }
                    for s in response.sources
                ],
                "chunks_retrieved": len(hybrid_results),
                "chunks_used": len(reranked),
                "latency_ms": round(latency_ms, 2),
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