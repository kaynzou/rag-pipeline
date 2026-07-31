"""Demo script for the generator module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chunking import TextSplitter
from src.embedding import LocalEmbeddingModel, EmbeddingPipeline
from src.bm25 import BM25Index
from src.vector_store import VectorStore
from src.hybrid_search import HybridSearch
from src.reranker import CrossEncoderReranker
from src.generator import Generator


def main() -> None:
    print("=" * 80)
    print("Demo: Grounded Generation with Claude")
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
        top_k=3,
        candidates_per_method=10,
    )

    reranker = CrossEncoderReranker()

    generator = Generator()

    queries = [
        "What is RAG evaluation?",
        "How does BM25 work?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        hybrid_results = hybrid.search(query)
        reranked = reranker.rerank(query, hybrid_results, top_k=3)
        response = generator.generate(query, reranked)

        print(f"\nAnswer:\n{response.answer}")
        print(f"\nSources ({response.chunks_used} chunks):")
        for s in response.sources:
            print(f"  [chunk_id={s.chunk_id}] score={s.rerank_score:.4f} | {s.text_preview}")


if __name__ == "__main__":
    main()