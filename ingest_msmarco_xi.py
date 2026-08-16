"""
Ingest MSMARCO-XI dataset from Hugging Face and index with the RAG pipeline.

Usage:
    python ingest_msmarco_xi.py \
        --max-examples 1000 \
        --strategies fixed semantic paragraph
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import RAGPipeline


def load_msmarco_xi(max_examples: int = 1000) -> str:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        from datasets import load_dataset

    print("Loading MSMARCO-XI default config ...")
    ds = load_dataset("ai4bharat/MSMARCO-XI", "default", split="train", streaming=True)

    corpus_parts = []
    for i, example in enumerate(ds):
        if i >= max_examples:
            break

        query = example.get("Eng_Query") or example.get("query") or ""
        answer = example.get("Eng_Answer") or example.get("Answer") or ""
        passages = example.get("passages") or {}
        english_passages = passages.get("English_passages") or []
        selected_texts = []
        if english_passages:
            is_selected = passages.get("is_selected") or []
            for idx, flag in enumerate(is_selected):
                if flag == 1 and idx < len(english_passages):
                    selected_texts.append(english_passages[idx])
        passage_text = "\n".join(selected_texts) if selected_texts else "\n".join(english_passages[:3])

        text = f"Query: {query}\nAnswer: {answer}\nPassages:\n{passage_text}"
        corpus_parts.append(text)

    print(f"Loaded {len(corpus_parts)} examples")
    return "\n\n".join(corpus_parts)


def main():
    parser = argparse.ArgumentParser(description="Ingest MSMARCO-XI into RAG pipeline")
    parser.add_argument("--max-examples", type=int, default=1000, help="Max examples to load")
    parser.add_argument("--persist-dir", default="data/index", help="Index directory")
    parser.add_argument("--strategies", nargs="*", default=None, help="Chunking strategies")
    args = parser.parse_args()

    corpus = load_msmarco_xi(args.max_examples)
    if not corpus:
        print("No content loaded. Exiting.")
        sys.exit(1)

    print(f"\nTotal corpus size: {len(corpus):,} characters")

    print("\nIndexing corpus...")
    pipeline = RAGPipeline(persist_dir=args.persist_dir)
    if args.strategies:
        pipeline.index_with_strategies(corpus, source="msmarco_xi_default", strategies=args.strategies)
    else:
        pipeline.index(corpus, source="msmarco_xi_default")
    print(f"Indexed successfully: {pipeline.indexed}")

    print("\nSample queries:")
    queries = [
        "What is the answer to the first query?",
        "Summarize the passages.",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        resp = pipeline.query(q, top_k=3)
        print(f"A: {resp.answer[:200]}...")


if __name__ == "__main__":
    main()
