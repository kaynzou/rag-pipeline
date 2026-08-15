# RAG Pipeline — Built From Scratch

A voice-enabled, production-grade Retrieval-Augmented Generation (RAG) pipeline
implemented from scratch in Python — no LangChain, no LlamaIndex. Every core
component (chunking, embeddings, vector search, keyword search, hybrid retrieval,
reranking, grounded generation, guardrails, latency analytics) is implemented
directly so the underlying algorithms are transparent and testable.

## What's new (HH Goa 2026 build)

- **Voice input**: record or upload audio; transcribed via Sarvam STT before retrieval.
- **Multi-strategy chunking**: fixed-size, semantic (sentence-based), paragraph-aware,
  and section-aware (metadata-aware) chunkers, with deduplication.
- **Structured harness**: `RAGHarness` wraps the pipeline with retries, structured
  input/output, and latency tracking.
- **Guardrails**: input validation, retrieval relevance checks, and output grounding
  checks that block unsafe, off-topic, or hallucinated answers.
- **Latency analytics**: per-query breakdown (retrieval / generation / total) with
  P50 / P70 / P100 reporting via `/latency`.
- **Dataset-ready**: `ingest_msmarco_xi.py` loads the MSMARCO-XI corpus from Hugging Face.

## Live Demo

- **Chat UI (Streamlit):** https://upgraded-trout-r4xwgr6w96qv3wpp6-8501.app.github.dev/
- **API (FastAPI):** https://rag-pipeline-o0f1.onrender.com/

> **Note:** Free-tier Render instances spin down after inactivity — the
> first request may take 30-50 seconds to wake up.

## Architecture

```
Indexing:  Raw text → Multi-Strategy Chunker → Embedder → Vector Store
                                                       ↘
                                               BM25 Index (keyword index)

Query:     Voice/Text → STT (optional) → Hybrid Search (BM25 + dense, fused via RRF)
                              → Cross-Encoder Reranker
                              → Guardrails (input / retrieval / output)
                              → LLM Generation (grounded, with citations)
                              → Answer + Sources + Latency Stats
```

## What's implemented

| Component | Details |
|---|---|
| **Chunking** | Fixed-size, semantic (sentence), paragraph-aware, section-aware (markdown headers), plus `MultiStrategyChunker` with deduplication |
| **Embeddings** | Local (`sentence-transformers`) or OpenAI API-backed |
| **Vector Store** | Brute-force cosine similarity search, pure numpy, with model-mismatch protection |
| **BM25** | Classic TF-IDF-style keyword scoring, implemented from the formula |
| **Hybrid Search** | BM25 + dense retrieval fused with Reciprocal Rank Fusion |
| **Reranker** | Cross-encoder second-stage reranking on the fused candidate set |
| **Generator** | OpenAI-based grounded generation — answers only from retrieved context, with `[chunk_id]` citations |
| **Guardrails** | Input safety/relevance, retrieval relevance threshold, output grounding + hallucination checks |
| **Harness** | `RAGHarness` with retry logic, structured I/O, and `LatencyTracker` (P50/P70/P100) |
| **STT** | Sarvam AI Speech-to-Text client with retries and fallback |
| **Evaluation** | Precision@k, recall@k, MRR, faithfulness, and groundedness against a labeled question set |
| **Serving** | FastAPI backend + Streamlit chat frontend, containerized with Docker |

**Test coverage:** 101 tests across all modules, all passing.

## Tech stack

Python · FastAPI · Streamlit · sentence-transformers · OpenAI · Sarvam STT · Docker · Render

## Local setup

```bash
pip install -e ".[dev,local]"
python3 -m pytest tests/ -v          # run the test suite (101 tests)
uvicorn src.server:app --reload      # start the API
streamlit run app.py                 # start the chat UI
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/ready` | Pipeline loaded and indexed? |
| GET | `/latency` | P50 / P70 / P100 latency stats |
| POST | `/index` | Index `data/sample_corpus.txt` |
| POST | `/query` | Text query → grounded answer |
| POST | `/query/stream` | Streaming text query |
| POST | `/voice/query` | Upload audio → transcript → answer |

## Dataset ingestion

```bash
python ingest_msmarco_xi.py --lang en --split train --max-examples 5000 \
    --strategies fixed semantic paragraph
```

## Latency benchmark

```bash
python scripts/latency_benchmark.py
# Saves data/latency_report.json with P50 / P70 / P100
```

## Key Concepts

### 1. Multi-Strategy Chunking
- **Fixed-size**: token-based sliding window with overlap
- **Semantic**: sentence-boundary aware splitting
- **Paragraph**: preserves paragraph boundaries
- **Section**: metadata-aware splitting on markdown headers
- **Deduplication**: `MultiStrategyChunker` merges strategies and removes exact-duplicate chunks

### 2. Voice-Enabled Input
- **Sarvam STT**: async client with retries
- **Fallback**: graceful degradation when `SARVAM_API_KEY` is missing

### 3. Harness & Retries
- **Structured orchestration**: `RAGHarness` wraps `RAGPipeline.query`
- **Retries**: configurable retry loop around the pipeline
- **Latency tracking**: `LatencyTracker` records per-query timings and computes percentiles

### 4. Guardrails
- **Input**: empty/too-long/profanity checks
- **Retrieval**: minimum relevance threshold; "no context" fallback
- **Output**: speculative-language detection; required-fact verification; hallucination guard

### 5. Latency Analytics
- **Per-query**: retrieval_ms, generation_ms, total_ms
- **Aggregated**: P50, P70, P100 exposed via `/latency`

## Project Structure

```
rag-pipeline/
├── src/
│   ├── chunking.py       # Text splitting with multiple strategies
│   ├── embedding.py      # Vector embeddings + cosine similarity
│   ├── vector_store.py   # Brute-force cosine similarity search
│   ├── bm25.py           # Keyword retrieval with TF-IDF saturation
│   ├── hybrid_search.py  # BM25 + dense fusion via RRF
│   ├── reranker.py       # Cross-encoder reranking
│   ├── generator.py      # Grounded generation with OpenAI
│   ├── pipeline.py       # End-to-end orchestration
│   ├── eval.py           # Evaluation harness
│   ├── server.py         # FastAPI REST API
│   ├── stt.py            # Sarvam Speech-to-Text client
│   ├── guardrails.py     # Input / retrieval / output guardrails
│   └── harness.py        # RAGHarness + LatencyTracker
├── tests/                # 101 tests across 12 files
├── scripts/
│   └── latency_benchmark.py
├── data/
│   ├── sample_corpus.txt # Sample RAG documentation
│   └── index/            # Persisted vector store + BM25
├── app.py                # Streamlit chat UI with voice support
├── ingest_msmarco_xi.py  # MSMARCO-XI dataset ingestion
├── pyproject.toml
└── README.md
```

### Run demos

```bash
python demo_chunking.py      # See text splitting
python demo_embedding.py     # See vector embeddings
python demo_vector_store.py  # See similarity search
python demo_bm25.py          # See keyword scoring
python demo_hybrid_search.py # See hybrid fusion
python demo_reranker.py      # See cross-encoder reranking
python demo_generator.py     # See grounded generation
python demo_full_pipeline.py # See end-to-end pipeline
```

### Run tests

```bash
pytest tests/ -v
```

### One-command local deployment (Docker)

```bash
docker compose up --build
```
```
Retrieval:
  precision@5: 0.85    (85% of retrieved chunks were relevant)
  recall@5: 0.72       (72% of all relevant chunks were retrieved)
  mrr: 0.91            (first relevant result at rank 1.1 on average)

Generation:
  faithfulness: 0.88   (88% of answer claims supported by context)
  groundedness: 0.90   (90% of sentences have valid citations)
```

These numbers are what make this a *system* instead of a *demo*.

## Real-World Applications

This pipeline solves the same problems that power:

| Application | What It Retrieves |
|-------------|-------------------|
| Customer support | Help center articles, troubleshooting guides |
| Internal docs | Confluence, Notion, GitHub wiki |
| Legal/Compliance | Contracts, regulations, case law |
| Research | Academic papers, technical reports |
| Code assistance | Documentation, Stack Overflow, internal wikis |
| Medical info | Clinical guidelines, peer-reviewed literature |

## License

MIT
