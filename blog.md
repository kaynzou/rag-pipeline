# Building a RAG Pipeline From Scratch

I built a complete Retrieval-Augmented Generation (RAG) pipeline — every component, from scratch. No LangChain, no LlamaIndex, no external vector database. Just Python, numpy, and a clear understanding of what's happening at each stage.

This post explains why I built it, how each piece works, and what I learned.

## The Problem with LLMs

Large Language Models are impressive, but they have a fundamental limitation: they only know what they were trained on. Their knowledge is:
- **Static**: Cut off at their training date
- **Generic**: They know about the world, not your company's internal documents
- **Unverifiable**: They can't tell you where they learned something
- **Hallucination-prone**: They'll make up plausible-sounding but false information rather than say "I don't know"

The traditional solution — retraining or fine-tuning — is expensive, slow, and requires ML expertise. There's a better way.

## What is RAG?

Retrieval-Augmented Generation (RAG) supplements the LLM's knowledge with external documents at query time.

```
Traditional LLM:
User Question → LLM (training data only) → Answer (may be wrong)

RAG Pipeline:
User Question → Retrieve relevant docs → Inject into prompt → LLM → Answer (grounded in real sources)
```

Instead of teaching the model new facts (expensive), you give it the facts it needs right now (cheap) and constrain it to use only those facts (effective).

## The Architecture

I broke the pipeline into 9 modular components, each responsible for one stage of the pipeline:

```
INDEXING (one-time)
Raw Text → Chunking → Embedding → Vector Store + BM25

QUERYING (per question)
Question → Hybrid Search → Rerank → Generate → Answer
```

### Stage 1: Chunking

LLMs have limited context windows. You can't feed entire documents — you need smaller pieces called "chunks."

I implemented a token-based sliding window splitter using tiktoken (OpenAI's tokenizer). Key decisions:
- **256 tokens per chunk**: Sweet spot between precision and context
- **40-token overlap**: Prevents losing context at chunk boundaries
- **Metadata on every chunk**: `source`, `start_token`, `end_token` for traceability

### Stage 2: Embeddings

Text is meaningless to computers. Embeddings translate text into vectors where *similar meaning = close proximity*.

I used `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, runs locally). The key insight: cosine similarity ignores vector magnitude and only compares direction. Two texts with identical meaning but different lengths should score the same.

Critical rule: use the same model for both corpus and queries. Models A and B produce vectors in different spaces — mixing them breaks the math.

### Stage 3: Vector Store

Stores embedding vectors and finds nearest neighbors. My implementation is brute-force: a single matrix multiply scores all chunks at once.

```python
scores = self._vectors @ query_vector  # (N, 384) @ (384,) = (N,)
```

For 10,000 chunks, this takes ~4ms. Fine for small-to-medium projects. I used `argpartition` instead of `argsort` for top-k retrieval — O(N) vs O(N log N).

### Stage 4: BM25

Embeddings are great for semantic similarity but blind to exact terms. "API" in a query might not match "API" in a document if the embedding model doesn't weight that token heavily.

BM25 (Best Matching 25) is a 30-year-old keyword ranking algorithm that solves this. It scores documents based on:
- **TF saturation**: A word appearing 10 times is more relevant than once, but 100 times isn't 100x more relevant
- **IDF**: Rare terms get boosted, common terms get zeroed out
- **Length normalization**: Long documents mentioning a term once are less relevant than short documents mentioning it once

### Stage 5: Hybrid Search

Neither BM25 nor embeddings alone is perfect. Hybrid search combines both using Reciprocal Rank Fusion (RRF):

```python
RRF_score(d) = Σ 1 / (k + rank_d_in_list_i)
```

RRF only cares about rank position, not score magnitude. This means BM25 scores (0-5) and cosine scores (0-1) can be fused without normalization. The constant k=60 makes the fusion robust — top ranks contribute slightly more but not overwhelmingly.

### Stage 6: Reranking

Initial retrieval is fast but coarse. The reranker uses a cross-encoder for fine-grained relevance scoring.

**Bi-encoder** (what I used for embeddings): encodes query and document independently. Fast but limited interaction.

**Cross-encoder**: processes query and document together via self-attention. Every token in the query can attend to every token in the document. This enables disambiguation ("bank" in "river bank" vs "bank account").

The tradeoff: cross-encoders are slow (~100ms per candidate), so I only run them on the top 20 candidates from hybrid search. This two-stage pattern (fast coarse → slow precise) is standard in production RAG systems.

### Stage 7: Generation

The LLM (Claude in my implementation) generates the final answer. The critical part is the prompt:

**System prompt:**
```
You are a helpful assistant that answers questions using ONLY the provided context.
1. Answer based SOLELY on the provided context chunks.
2. If context doesn't contain enough info, say "I don't have enough information..."
3. Do NOT use outside knowledge or make up information.
4. For every factual claim, add a citation like [chunk_id].
```

Three layers prevent hallucination:
1. System prompt explicitly forbids outside knowledge
2. Model can only see retrieved context
3. Citations force traceability — if a claim isn't in the context, it can't be cited

Temperature is set to 0.0 for deterministic, factual answers. No creativity allowed.

### Stage 8: Pipeline + Evaluation

The `RAGPipeline` class orchestrates everything. Indexing happens once; querying happens per question.

Evaluation is what makes this a system instead of a demo. I built a harness that computes:

- **Precision@k**: Of retrieved chunks, how many were relevant?
- **Recall@k**: Of all relevant chunks, how many were retrieved?
- **MRR**: How high does the first relevant result appear?
- **Faithfulness**: Are answer claims supported by context?
- **Groundedness**: Do claims have valid citations?

These numbers let me compare configurations objectively. "Precision@5: 0.85" is worth more on a resume than "it works pretty well."

### Stage 9: Serving Layer

FastAPI exposes the pipeline as a REST API. Key design decisions:
- **POST /query**: JSON body for safe, flexible input
- **Pydantic validation**: Automatic 422 errors for invalid input
- **/health and /ready**: Process alive vs. pipeline loaded
- **Lifespan context manager**: Load index on startup, clean up on shutdown

## What I Learned

### 1. Chunking is harder than it looks

The chunk size tradeoff is real. Too small and you lose context. Too large and you dilute the signal. I started with 256 tokens as a baseline, but the optimal size depends on your document type and query patterns.

### 2. BM25 still matters

In 2024, with embeddings everywhere, a 30-year-old keyword algorithm is still essential. Embeddings miss exact terms, acronyms, and rare words. BM25 catches what embeddings miss.

### 3. Reranking is non-negotiable

Hybrid search gets you 80% of the way there. The cross-encoder reranker gets you the last 20% — the difference between "good enough" and "precise." The two-stage pattern (fast coarse → slow precise) is how production systems work.

### 4. Evaluation is everything

You can't improve what you can't measure. A labeled question set with precision/recall/faithfulness metrics turns guesswork into engineering.

### 5. From scratch > from framework

Building from first principles forced me to understand every design decision. I can explain why cosine similarity, why RRF, why k=60, why temperature=0.0. That understanding is what separates a developer from an engineer.

## The Code

The full implementation is ~1,200 lines of production code with 74 tests. No external retrieval dependencies — just numpy, sentence-transformers, and FastAPI.

[View the full project on GitHub](https://github.com/kaynzou/rag-pipeline)

## Try It Yourself

```bash
git clone https://github.com/kaynzou/rag-pipeline.git
cd rag-pipeline
pip install -e ".[dev]"
uvicorn src.server:app --reload
# Visit http://localhost:8000/docs
```

## What's Next

- Streaming responses for real-time answer generation
- Conversational memory for multi-turn chat
- PDF and DOCX ingestion
- Semantic chunking (split at topic boundaries)
- A real corpus (indexing actual Wikipedia articles or documentation)

The full source code and detailed explanations of each module are in the [GitHub repository](https://github.com/kaynzou/rag-pipeline).