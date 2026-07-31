"""Demo script for the reranker module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chunking import TextSplitter
from src.embedding import LocalEmbeddingModel, EmbeddingPipeline
from src.bm25 import BM25Index
from src.vector_store import VectorStore
from src.hybrid_search import HybridSearch
from src.reranker import CrossEncoderReranker


def main() -> None:
    print("=" * 80)
    print("Demo: Cross-Encoder Reranker")
    print("=" * 80)

    corpus_path = Path(__file__).parent / "data" / "sample_corpus.txt"
    text = corpus_path.read_text()

    splitter = TextSplitter(chunk_size=256, chunk_overlap=40)
    chunks = splitter.split(text, source="sample_corpus.txt")

    encoder = LocalEmbeddingModel()
    pipeline = EmbeddingPipeline(model=encoder)
    embedded = pipeline.embed_chunks(chunks)

    vector_store = VectorStore(dimension=encoder.dimension, model_name="all-MiniLM-L6-v2")
    vector_store.add(embedded)

    bm25 = BM25Index()
    bm25.build(chunks)

    hybrid = HybridSearch(
        bm25_index=bm25,
        vector_store=vector_store,
        encoder=encoder,
        top_k=5,
        candidates_per_method=10,
    )

    reranker = CrossEncoderReranker()

    queries = [
        "BM25 keyword scoring algorithm",
        "RAG evaluation metrics",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        hybrid_results = hybrid.search(query)
        print("\nBefore reranking:")
        for r in hybrid_results:
            print(f"  rrf={r.rrf_score:.4f} bm25={r.bm25_score:.3f} dense={r.dense_score:.3f} | {r.text[:60]}...")

        reranked = reranker.rerank(query, hybrid_results, top_k=3)
        print("\nAfter reranking:")
        for r in reranked:
            print(
                f"  rerank={r.rerank_score:.4f} rrf={r.rrf_score:.4f} | "
                f"bm25={r.bm25_score:.3f} dense={r.dense_score:.3f} | {r.text[:60]}..."
            )


if __name__ == "__main__":
    main()