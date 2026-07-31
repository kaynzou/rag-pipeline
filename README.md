# RAG Pipeline — Built From Scratch

A complete Retrieval-Augmented Generation (RAG) pipeline implemented from first principles. No LangChain, no LlamaIndex, no external vector database. Every component is built and understood from the ground up.

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

## Quick Start

### Option A: Run the API (backend only)

```bash
pip install -e ".[dev]"
uvicorn src.server:app --reload
# Visit http://localhost:8000/docs
```

### Option B: Run with Docker (full stack)

```bash
docker compose up --build
# API: http://localhost:8000/docs
# Frontend: http://localhost:8501
```

### Option C: Run the Streamlit frontend

```bash
# Terminal 1: start API
uvicorn src.server:app --reload

# Terminal 2: start frontend
pip install streamlit requests
streamlit run app.py
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
```

### Run tests

```bash
pytest tests/ -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Pipeline Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INDEXING (one-time)                                         │
│  Raw Text → Chunking → Embedding → Vector Store + BM25      │
│                                                              │
│  QUERYING (per question)                                     │
│  Question → Hybrid Search → Rerank → Generate → Answer      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Flow

```
User Question
     │
     ▼
┌─────────────────┐
│  Hybrid Search  │
│                 │
│  ┌───────────┐  │
│  │ BM25      │  │ ──► Keyword matching (exact terms, acronyms)
│  └───────────┘  │
│  ┌───────────┐  │
│  │ Dense     │  │ ──► Semantic matching (paraphrase, concepts)
│  └───────────┘  │
│       │         │
│       ▼         │
│  RRF Fusion     │ ──► Merge both signal sources
└─────────────────┘
     │
     ▼
┌─────────────────┐
│   Reranker      │ ──► Cross-encoder re-scores top candidates
└─────────────────┘
     │
     ▼
┌─────────────────┐
│   Generator     │ ──► Claude answers from context + citations
└─────────────────┘
     │
     ▼
  RAGResponse
  (answer + sources + metadata)
```

## Module Reference

| Module | Lines | Purpose |
|--------|-------|---------|
| `chunking.py` | 128 | Token-based text splitting with overlap |
| `embedding.py` | 130 | Vector embeddings + cosine similarity |
| `vector_store.py` | 130 | Brute-force similarity search + persistence |
| `bm25.py` | 150 | BM25 keyword ranking from scratch |
| `hybrid_search.py` | 100 | BM25 + dense fusion via RRF |
| `reranker.py` | 130 | Cross-encoder reranking |
| `generator.py` | 120 | Grounded generation with Claude |
| `pipeline.py` | 180 | End-to-end orchestration |
| `eval.py` | 130 | Evaluation harness |
| `server.py` | 100 | FastAPI REST API |

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

## What Makes This Resume-Worthy

1. **From scratch**: You understand every component, every parameter, every tradeoff
2. **Full stack**: Chunking → embeddings → retrieval → reranking → generation → API
3. **Evaluated**: Precision@k, recall@k, faithfulness — not just "it works"
4. **Production concerns**: Persistence, error handling, API design, logging
5. **Systems thinking**: You can explain why each design decision was made and what the alternatives are

## Interview Talking Points

### "Walk me through how RAG works"
> RAG supplements an LLM's knowledge with external documents. Instead of relying on what the model learned during training, we retrieve relevant documents at query time, inject them into the prompt, and constrain the model to answer only from that context. This gives us accuracy, source attribution, and up-to-date knowledge without retraining the model.

### "Why did you build this from scratch instead of using LangChain?"
> LangChain is great for prototyping, but it abstracts away the important details. By building from scratch, I understand exactly what's happening at each stage: how chunking affects retrieval quality, why cosine similarity is the right metric, how BM25 complements embeddings, why cross-encoders are more accurate than bi-encoders. If something goes wrong in production, I can debug it because I built it.

### "How do you evaluate your RAG system?"
> I use two categories of metrics. For retrieval: precision@k tells me what fraction of retrieved chunks are relevant, recall@k tells me what fraction of all relevant chunks I retrieved, and MRR tells me how high the first relevant result appears. For generation: faithfulness measures whether answer claims are supported by the context, and groundedness measures whether claims have valid citations. I built an evaluation harness that runs a labeled question set and produces these metrics automatically.

### "What's the difference between BM25 and embeddings for retrieval?"
> BM25 is a keyword-based method — it matches exact terms and is great for acronyms, rare words, and exact-match queries. Embeddings are semantic — they match concepts and paraphrases. Neither is perfect alone. BM25 misses semantic matches, and embeddings miss exact matches. That's why I use hybrid search with Reciprocal Rank Fusion to combine both signals.

### "Why do you need a reranker if you already have hybrid search?"
> Hybrid search is fast but coarse. BM25 and dense retrieval both produce approximate rankings. The reranker uses a cross-encoder, which processes the query and document together via attention, enabling much more fine-grained relevance judgments. The tradeoff is speed — cross-encoders are slow, so we only run them on the top 20 candidates from hybrid search.

### "How do you prevent hallucination?"
> Three layers: first, the system prompt explicitly forbids using outside knowledge and requires citations. Second, the model can only see the retrieved context — it doesn't have access to its training data for this query. Third, the citation mechanism forces the model to tie every claim to a specific chunk. If a claim isn't in the context, the model can't cite it, which makes hallucination visible.

## Future Enhancements

If you want to extend this project:

1. **Approximate Nearest Neighbor**: Add IVF or HNSW index for scaling beyond 100K chunks
2. **Streaming responses**: Server-Sent Events for real-time answer generation
3. **Multi-document support**: Handle multiple file formats (PDF, DOCX, HTML)
4. **Conversational RAG**: Add chat history for follow-up questions
5. **Advanced chunking**: Semantic splitting, recursive splitting, parent-child chunking
6. **Metadata filtering**: Filter by source, date, author before retrieval
7. **A/B testing**: Compare different chunk sizes, models, rerankers
8. **Observability**: Add structured logging, metrics, tracing

## License

MIT
