"""
Real corpus ingestion script.

Downloads Wikipedia articles on a specific topic and indexes them
with the RAG pipeline. Demonstrates the system with real-world content.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGPipeline


def download_wikipedia_articles(topic: str, count: int = 10) -> str:
    """Download Wikipedia articles on a topic."""
    try:
        import wikipedia
    except ImportError:
        print("Installing wikipedia library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wikipedia"])
        import wikipedia

    wikipedia.set_lang("en")
    corpus_parts = []

    print(f"Searching Wikipedia for '{topic}'...")
    search_results = wikipedia.search(topic, results=count * 2)
    print(f"Found {len(search_results)} articles")

    downloaded = 0
    for title in search_results:
        if downloaded >= count:
            break
        try:
            page = wikipedia.page(title, auto_suggest=False)
            corpus_parts.append(f"\n\n## {page.title}\n\n{page.content}")
            downloaded += 1
            print(f"  Downloaded: {page.title} ({len(page.content)} chars)")
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"  Skipped (disambiguation): {title}")
            continue
        except wikipedia.exceptions.PageError:
            print(f"  Skipped (not found): {title}")
            continue

    return "\n".join(corpus_parts)


def main():
    parser = argparse.ArgumentParser(description="Download and index a Wikipedia corpus")
    parser.add_argument("topic", help="Wikipedia topic to search for")
    parser.add_argument("--count", type=int, default=10, help="Number of articles to download")
    parser.add_argument("--persist-dir", default="data/index", help="Directory to save the index")
    args = parser.parse_args()

    print("=" * 80)
    print(f"Wikipedia Corpus Ingestion: '{args.topic}'")
    print("=" * 80)

    corpus = download_wikipedia_articles(args.topic, args.count)
    if not corpus:
        print("No content downloaded. Exiting.")
        sys.exit(1)

    total_chars = len(corpus)
    print(f"\nTotal corpus size: {total_chars:,} characters")

    print("\nIndexing corpus...")
    pipeline = RAGPipeline(persist_dir=args.persist_dir)
    pipeline.index(corpus, source=f"wikipedia_{args.topic}")
    print(f"Indexed successfully: {pipeline.indexed}")

    print("\nSample queries:")
    queries = [
        f"What is {args.topic}?",
        f"Explain the key concepts of {args.topic}",
        f"What are the applications of {args.topic}?",
    ]

    for query in queries:
        print(f"\nQ: {query}")
        response = pipeline.query(query, top_k=3)
        print(f"A: {response.answer[:200]}...")
        print(f"   Sources: {[s.source_file for s in response.sources]}")


if __name__ == "__main__":
    main()
