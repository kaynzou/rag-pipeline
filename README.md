# RAG Pipeline — Built From Scratch

A retrieval-augmented generation (RAG) pipeline implemented from scratch in
Python — no LangChain, no LlamaIndex. Every core component (chunking,
embeddings, vector search, keyword search, hybrid retrieval, reranking,
grounded generation) is implemented directly so the underlying algorithms
are transparent and testable.

## Live Demo

- **Chat UI (Streamlit):** https://rag-pipeline-o0f1.onrender.com
- **API (FastAPI):** https://rag-pipeline-1-y3x7.onrender.com/health

> **Note:** Free-tier Render instances spin down after inactivity — the
> first request may take 30-50 seconds to wake up.

## Architecture

​```
Indexing:  Raw text → Chunker → Embedder → Vector Store
                                          ↘
                                  BM25 Index (keyword index)

Query:     User query → Hybrid Search (BM25 + dense, fused via RRF)
                       → Cross-Encoder Reranker
                       → LLM Generation (grounded, with citations)
                       → Answer + Sources
​```
## What's implemented

| Component | Details |
|---|---|
| **Chunking** | Sliding-window token splitter with overlap, built on `tiktoken` |
| **Embeddings** | Local (`sentence-transformers`) or OpenAI API-backed |
| **Vector Store** | Brute-force cosine similarity search, pure numpy, with model-mismatch protection |
| **BM25** | Classic TF-IDF-style keyword scoring, implemented from the formula |
| **Hybrid Search** | BM25 + dense retrieval fused with Reciprocal Rank Fusion |
| **Reranker** | Cross-encoder second-stage reranking on the fused candidate set |
| **Generator** | Claude-based grounded generation — answers only from retrieved context, with `[chunk_id]` citations |
| **Evaluation** | Precision@k, recall@k, MRR, faithfulness, and groundedness against a labeled question set |
| **Serving** | FastAPI backend + Streamlit chat frontend, containerized with Docker |

**Test coverage:** 74 tests across all modules, all passing.

## Tech stack

Python · FastAPI · Streamlit · sentence-transformers · Claude API · Docker · Render

## Local setup

```bash
pip install -e ".[dev,local]"
python3 -m pytest tests/ -v          # run the test suite
uvicorn src.server:app --reload      # start the API
streamlit run app.py                 # start the chat UI
```

## What This Project Does

Turns raw text into a question-answering system that:
- Retrieves relevant chunks using **hybrid search** (keyword + semantic)
- Reranks candidates with a **cross-encoder** for precision
- Generates cited answers using **Claude** constrained to retrieved context
- Evaluates quality with **precision@k**, **recall@k**, and **faithfulness** metrics
- Serves everything via a **FastAPI** REST endpoint

## Why This Exists

LLMs are powerful but flawed:
- **Knowledge cutoff**: They don't know about events after their training date
- **Hallucination**: They make up plausible-sounding but false information
- **No source attribution**: They can't tell you *where* they learned something
- **No domain specificity**: They're generalists, not experts on your data

RAG solves this by injecting external knowledge at query time. Instead of retraining the model (expensive, slow), you give it the facts it needs *right now* and constrain it to use only those facts.

## Project Structure

```
rag-pipeline/
├── src/
│   ├── chunking.py       # Text splitting with token-based sliding window
│   ├── embedding.py      # Vector embeddings + cosine similarity
│   ├── vector_store.py   # Brute-force cosine similarity search
│   ├── bm25.py           # Keyword retrieval with TF-IDF saturation
│   ├── hybrid_search.py  # BM25 + dense fusion via RRF
│   ├── reranker.py       # Cross-encoder reranking
│   ├── generator.py      # Grounded generation with Claude
│   ├── pipeline.py       # End-to-end orchestration
│   ├── eval.py           # Evaluation harness
│   └── server.py         # FastAPI REST API
├── tests/                # 74 tests across 9 files
├── data/
│   ├── sample_corpus.txt # Sample RAG documentation
│   └── index/            # Persisted vector store + BM25
├── demo_*.py             # 7 standalone demos
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
**Total: ~1,200 lines of production code + 2,000 lines of tests**

## Key Concepts Learned

### 1. Chunking
- **Tokenization**: Text → tokens (what the LLM actually sees)
- **Chunk size**: 256 tokens is the sweet spot for most use cases
- **Overlap**: 10-20% of chunk size prevents context loss at boundaries
- **Metadata**: `source`, `start_token`, `end_token` enable source attribution

### 2. Embeddings
- **Semantic space**: Similar meaning = close vectors
- **Cosine similarity**: Direction-only comparison, ignores magnitude
- **Model consistency**: Same model for corpus and query, always
- **Normalization**: Unit-length vectors make search a single matrix multiply

### 3. Vector Store
- **Brute force**: O(N × dim) per query — fine for <100K chunks
- **argpartition**: O(N) partial sort vs O(N log N) full sort
- **Persistence**: Save vectors + metadata to disk for reuse
- **Dimension validation**: Prevents silent corruption from model mismatches

### 4. BM25
- **TF saturation**: Diminishing returns as term frequency increases
- **IDF**: Rare terms get boosted, common terms get zeroed out
- **Length normalization**: Long documents are slightly penalized
- **Parameters**: k1=1.2 (saturation), b=0.75 (length norm) — production defaults

### 5. Hybrid Search
- **Complementary signals**: BM25 (exact) + dense (semantic)
- **RRF**: Rank-based fusion, no score normalization needed
- **k=60**: Empirically optimal RRF constant
- **No normalization**: BM25 scores (0-5) and cosine scores (0-1) are fused by rank, not magnitude

### 6. Reranking
- **Bi-encoder vs Cross-encoder**: Independent encoding vs joint attention
- **Two-stage pattern**: Fast coarse retrieval → slow precise reranking
- **Cross-attention**: Enables disambiguation ("bank" in "river bank" vs "bank account")
- **Candidate set**: Only rerank top-20 from hybrid search, not all 100K

### 7. Generation
- **System prompt as constitution**: Behavioral constraints set before content
- **Temperature=0.0**: Deterministic generation for factual Q&A
- **Citations**: `[chunk_id]` ties every claim to a source
- **Fallback**: "I don't have enough information..." for empty contexts

### 8. Evaluation
- **Precision@k**: Of retrieved chunks, how many were relevant?
- **Recall@k**: Of all relevant chunks, how many were retrieved?
- **MRR**: How high does the first relevant result appear?
- **Faithfulness**: Are answer claims supported by context?
- **Groundedness**: Do claims have valid citations?

### 9. Serving
- **POST /query**: JSON body for safe, flexible query input
- **Pydantic validation**: Automatic 422 errors for invalid input
- **Health vs Ready**: Process alive vs. pipeline loaded
- **Lifespan**: Load index on startup, clean up on shutdown

## Evaluation Metrics

When you build a labeled question set (20-50 questions with known answers), you'll get numbers like:

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
