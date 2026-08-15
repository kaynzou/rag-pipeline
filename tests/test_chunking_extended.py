import pytest
from src.chunking import (
    TextSplitter,
    SentenceSplitter,
    ParagraphSplitter,
    SectionSplitter,
    MultiStrategyChunker,
    Chunk,
    Tokenizer,
)


class TestSentenceSplitter:
    def test_basic_split(self):
        splitter = SentenceSplitter(max_chunk_size=20)
        text = "First sentence. Second sentence. Third sentence."
        chunks = splitter.split(text, source="test.txt")
        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.source == "test.txt"

    def test_empty_text(self):
        splitter = SentenceSplitter()
        chunks = splitter.split("", source="test.txt")
        assert chunks == []


class TestParagraphSplitter:
    def test_basic_split(self):
        splitter = ParagraphSplitter(max_chunk_size=20)
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = splitter.split(text, source="test.txt")
        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.source == "test.txt"


class TestSectionSplitter:
    def test_markdown_sections(self):
        splitter = SectionSplitter(max_chunk_size=20)
        text = "# Header One\nBody one.\n\n# Header Two\nBody two."
        chunks = splitter.split(text, source="test.txt")
        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.source == "test.txt"

    def test_metadata_section(self):
        splitter = SectionSplitter()
        text = "# Intro\nSome intro text."
        chunks = splitter.split(text, source="test.txt")
        assert any(c.metadata.get("section") for c in chunks)


class TestMultiStrategyChunker:
    def test_multiple_strategies(self):
        text = (
            "First sentence. Second sentence. Third sentence. Fourth sentence. "
            "Fifth sentence. Sixth sentence.\n\n"
            "Para one contains some text.\n\n"
            "Para two contains more text here.\n\n"
            "Para three has even more content.\n\n"
            + " ".join([f"word{i}" for i in range(500)])
        )
        chunker = MultiStrategyChunker(strategies=["fixed", "semantic", "paragraph"])
        chunks = chunker.chunk(text, source="test.txt")
        assert len(chunks) > 0
        strategies = {c.metadata.get("strategy") for c in chunks}
        assert "fixed" in strategies
        assert "semantic" in strategies
        assert "paragraph" in strategies

    def test_deduplication(self):
        text = "Hello world. Hello world."
        chunker = MultiStrategyChunker(strategies=["fixed", "semantic"])
        chunks = chunker.chunk(text, source="test.txt")
        texts = [c.text for c in chunks]
        assert len(texts) == len(set(texts))
