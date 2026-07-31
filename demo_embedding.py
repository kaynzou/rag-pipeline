"""Demo script for the embedding module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chunking import TextSplitter
from src.embedding import LocalEmbeddingModel, OpenAIEmbeddingModel, cosine_similarity, EmbeddingPipeline
import numpy as np


def demo_local() -> None:
    print("=" * 80)
    print("Demo: Local Embedding Model (all-MiniLM-L6-v2)")
    print("=" * 80)

    model = LocalEmbeddingModel()
    print(f"Model dimension: {model.dimension}")

    texts = [
        "RAG combines retrieval with generation.",
        "Vector search finds similar documents.",
        "The cat sat on the mat.",
        "BM25 is a keyword scoring algorithm.",
    ]

    vectors = model.embed(texts)
    print(f"Embedded {len(vectors)} texts -> shape {vectors.shape}")

    query = "How does semantic search work?"
    query_vec = model.embed([query])

    sims = cosine_similarity(vectors, query_vec).flatten()
    ranked = sorted(zip(texts, sims), key=lambda x: x[1], reverse=True)

    print("\nQuery:", query)
    print("\nRanked results:")
    for i, (text, score) in enumerate(ranked, 1):
        print(f"  {i}. [{score:.4f}] {text}")


def demo_pipeline() -> None:
    print("\n" + "=" * 80)
    print("Demo: Embedding Pipeline with Chunker")
    print("=" * 80)

    corpus_path = Path(__file__).parent / "data" / "sample_corpus.txt"
    text = corpus_path.read_text()

    splitter = TextSplitter(chunk_size=256, chunk_overlap=40)
    chunks = splitter.split(text, source="sample_corpus.txt")

    pipeline = EmbeddingPipeline()
    embedded = pipeline.embed_chunks(chunks)

    print(f"Embedded {len(embedded)} chunks with dimension {embedded[0].embedding.shape[0]}")

    query_vec = pipeline.model.embed(["What is RAG evaluation?"])
    sims = cosine_similarity(
        np.array([e.embedding for e in embedded]), query_vec
    ).flatten()

    best_idx = int(np.argmax(sims))
    print(f"\nBest match: chunk {embedded[best_idx].chunk_id}, score={sims[best_idx]:.4f}")
    print(f"Text: {embedded[best_idx].text[:120]}...")


if __name__ == "__main__":
    demo_local()
    demo_pipeline()
