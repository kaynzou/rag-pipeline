"""Demo script for the vector store module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chunking import TextSplitter
from src.embedding import LocalEmbeddingModel, EmbeddingPipeline
from src.vector_store import VectorStore
import numpy as np


def main() -> None:
    print("=" * 80)
    print("Demo: Vector Store — Build, Search, Persist")
    print("=" * 80)

    corpus_path = Path(__file__).parent / "data" / "sample_corpus.txt"
    text = corpus_path.read_text()

    splitter = TextSplitter(chunk_size=256, chunk_overlap=40)
    chunks = splitter.split(text, source="sample_corpus.txt")
    print(f"Chunks created: {len(chunks)}")

    pipeline = EmbeddingPipeline()
    embedded = pipeline.embed_chunks(chunks)

    store = VectorStore(dimension=pipeline.model.dimension, model_name="all-MiniLM-L6-v2")
    store.add(embedded)
    print(f"Vectors stored: {store.size}")
    print(f"Vector shape: {store._vectors.shape}\n")

    queries = [
        "What is RAG evaluation?",
        "How does chunking affect retrieval?",
        "What is BM25?",
    ]

    for query in queries:
        query_vec = pipeline.model.embed([query])[0]
        results = store.search(query_vec, top_k=2)

        print(f"Query: {query}")
        for r in results:
            print(f"  [{r.score:.4f}] {r.text[:90]}...")
        print()

    persist_dir = Path(__file__).parent / "data" / "vector_store"
    store.save(str(persist_dir))
    print(f"Store saved to {persist_dir}")

    loaded = VectorStore.load(str(persist_dir))
    print(f"Loaded store size: {loaded.size}")
    print(f"Model name: {loaded._model_name}")

    query_vec = pipeline.model.embed(["What is RAG evaluation?"])[0]
    results = loaded.search(query_vec, top_k=1)
    print(f"Search after reload: [{results[0].score:.4f}] {results[0].text[:60]}...")


if __name__ == "__main__":
    main()
