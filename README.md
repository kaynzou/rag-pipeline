# RAG Pipeline — Built From Scratch

A retrieval-augmented generation (RAG) pipeline implemented from scratch in
Python — no LangChain, no LlamaIndex. Every core component (chunking,
embeddings, vector search, keyword search, hybrid retrieval, reranking,
grounded generation) is implemented directly so the underlying algorithms
are transparent and testable.

## Live Demo

- **Chat UI (Streamlit):** https://upgraded-trout-r4xwgr6w96qv3wpp6-8501.app.github.dev/
- **API (FastAPI):** https://rag-pipeline-o0f1.onrender.com/

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
