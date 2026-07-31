"""Demo script for the BM25 module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chunking import TextSplitter
from src.bm25 import BM25Index


def main() -> None:
    print("=" * 80)
    print("Demo: BM25 Keyword Search")
    print("=" * 80)

    corpus_path = Path(__file__).parent / "data" / "sample_corpus.txt"
    text = corpus_path.read_text()

    splitter = TextSplitter(chunk_size=256, chunk_overlap=40)
    chunks = splitter.split(text, source="sample_corpus.txt")
    print(f"Chunks created: {len(chunks)}\n")

    index = BM25Index()
    index.build(chunks)
    print(f"Index built (avg doc length: {index._avgdl:.1f} tokens)\n")

    queries = [
        "BM25 keyword scoring",
        "RAG evaluation metrics",
        "vector embeddings",
        "hybrid search",
    ]

    for query in queries:
        results = index.search(query, top_k=3)
        print(f"Query: {query}")
        for r in results:
            print(f"  [{r.score:.4f}] {r.text[:80]}...")
        print()


if __name__ == "__main__":
    main()