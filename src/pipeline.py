from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.chunking import TextSplitter, Chunk, MultiStrategyChunker
from src.embedding import EmbeddingPipeline, OpenAIEmbeddingModel
from src.bm25 import BM25Index
from src.vector_store import VectorStore
from src.hybrid_search import HybridSearch
from src.generator import Generator, RAGResponse

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        persist_dir: str = "data/index",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        claude_model: str = "gpt-4o-mini",
        chunking_strategies: Optional[List[str]] = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._claude_model = claude_model
        self._chunking_strategies = chunking_strategies or ["fixed"]

        self._setup_models()
        self._setup_components()

    def _setup_models(self) -> None:
        if self._openai_api_key:
            self._encoder = OpenAIEmbeddingModel(api_key=self._openai_api_key)
        else:
            try:
                from src.embedding import LocalEmbeddingModel
                self._encoder = LocalEmbeddingModel()
            except ImportError:
                raise ImportError(
                    "No embedding model available. Either install sentence-transformers "
                    "(`pip install -e '.[local]'`) or set OPENAI_API_KEY."
                )

        self._pipeline = EmbeddingPipeline(model=self._encoder)

    def _setup_components(self) -> None:
        self._vector_store = VectorStore(
            dimension=self._encoder.dimension,
            model_name=getattr(self._encoder, "model_name", "local"),
        )
        self._bm25 = BM25Index()
        self._chunker = TextSplitter()
        self._hybrid = None
        self._reranker = None
        self._generator = None
        self._indexed = False

    def index(self, text: str, source: str = "corpus") -> None:
        chunks = self._chunker.split(text, source=source)
        embedded = self._pipeline.embed_chunks(chunks)

        self._vector_store = VectorStore(
            dimension=self._encoder.dimension,
            model_name=getattr(self._encoder, "model_name", "local"),
        )
        self._vector_store.add(embedded)

        self._bm25 = BM25Index()
        self._bm25.build(chunks)

        self._hybrid = HybridSearch(
            bm25_index=self._bm25,
            vector_store=self._vector_store,
            encoder=self._encoder,
            top_k=10,
            candidates_per_method=10,
        )

        self._reranker = None
        self._generator = None
        self._indexed = True

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._vector_store.save(self._persist_dir)
        with open(Path(self._persist_dir) / "bm25_chunks.json", "w") as f:
            json.dump(
                [
                    {
                        "text": c.text,
                        "source": c.source,
                        "start_token": c.start_token,
                        "end_token": c.end_token,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ],
                f,
                indent=2,
            )

    def index_with_strategies(self, text: str, source: str = "corpus", strategies: Optional[List[str]] = None) -> None:
        strategies = strategies or self._chunking_strategies
        chunker = MultiStrategyChunker(strategies=strategies)
        chunks = chunker.chunk(text, source=source)
        embedded = self._pipeline.embed_chunks(chunks)

        self._vector_store = VectorStore(
            dimension=self._encoder.dimension,
            model_name=getattr(self._encoder, "model_name", "local"),
        )
        self._vector_store.add(embedded)

        self._bm25 = BM25Index()
        self._bm25.build(chunks)

        self._hybrid = HybridSearch(
            bm25_index=self._bm25,
            vector_store=self._vector_store,
            encoder=self._encoder,
            top_k=10,
            candidates_per_method=10,
        )

        self._reranker = None
        self._generator = None
        self._indexed = True

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._vector_store.save(self._persist_dir)
        with open(Path(self._persist_dir) / "bm25_chunks.json", "w") as f:
            json.dump(
                [
                    {
                        "text": c.text,
                        "source": c.source,
                        "start_token": c.start_token,
                        "end_token": c.end_token,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ],
                f,
                indent=2,
            )

    def load(self, persist_dir: Optional[str] = None) -> None:
        persist_dir = persist_dir or self._persist_dir
        self._vector_store = VectorStore.load(persist_dir)

        with open(Path(persist_dir) / "bm25_chunks.json") as f:
            chunk_data = json.load(f)

        chunks = [
            Chunk(
                text=c["text"],
                chunk_id=i,
                source=c["source"],
                start_token=c["start_token"],
                end_token=c["end_token"],
                metadata=c.get("metadata", {}),
            )
            for i, c in enumerate(chunk_data)
        ]

        self._bm25 = BM25Index()
        self._bm25.build(chunks)

        self._hybrid = HybridSearch(
            bm25_index=self._bm25,
            vector_store=self._vector_store,
            encoder=self._encoder,
            top_k=10,
            candidates_per_method=10,
        )

        self._reranker = None
        self._generator = None
        self._indexed = True

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        if not self._indexed:
            raise RuntimeError("Pipeline not indexed. Call index() or load() first.")

        if self._reranker is None:
            try:
                from src.reranker import CrossEncoderReranker
                self._reranker = CrossEncoderReranker()
            except ImportError:
                logger.warning("sentence-transformers not installed; skipping reranker.")
                self._reranker = None

        if self._generator is None:
            self._generator = Generator(api_key=self._anthropic_api_key, model=self._claude_model)

        hybrid_results = self._hybrid.search(question)

        if self._reranker is not None:
            reranked = self._reranker.rerank(question, hybrid_results, top_k=top_k)
        else:
            reranked = hybrid_results[:top_k]

        response = self._generator.generate(question, reranked)

        response.chunks_retrieved = len(hybrid_results)
        return response

    @property
    def indexed(self) -> bool:
        return self._indexed
