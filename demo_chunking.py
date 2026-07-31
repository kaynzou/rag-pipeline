"""Demo script for the chunking module."""

import sys
from pathlib import Path

# Add src to path so imports work when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import TextSplitter, Tokenizer


def main() -> None:
    corpus_path = Path(__file__).parent / "data" / "sample_corpus.txt"
    text = corpus_path.read_text()

    print(f"Corpus size: {len(text)} characters")
    print(f"Corpus tokens: ~{len(text.split())} words\n")

    tokenizer = Tokenizer()
    total_tokens = tokenizer.count_tokens(text)
    print(f"Total tokens in corpus: {total_tokens}\n")

    splitter = TextSplitter(chunk_size=256, chunk_overlap=40)
    chunks = splitter.split(text, source="sample_corpus.txt")

    print(f"Number of chunks: {len(chunks)}\n")
    print("=" * 80)
    for chunk in chunks:
        print(f"\n--- Chunk {chunk.chunk_id} (tokens {chunk.start_token}-{chunk.end_token}) ---")
        print(chunk.text)
        print("-" * 80)


if __name__ == "__main__":
    main()
