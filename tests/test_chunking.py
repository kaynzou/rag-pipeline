import pytest
from src.chunking import TextSplitter, Tokenizer, Chunk


class TestTokenizer:
    def test_count_tokens(self):
        tokenizer = Tokenizer()
        count = tokenizer.count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_encode_decode_roundtrip(self):
        tokenizer = Tokenizer()
        text = "RAG is awesome"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text


class TestTextSplitter:
    def setup_method(self):
        self.splitter = TextSplitter(chunk_size=10, chunk_overlap=2)

    def test_basic_split(self):
        text = "This is a test sentence that should be split into multiple chunks by the splitter"
        chunks = self.splitter.split(text, source="test.txt")
        assert len(chunks) > 1

    def test_chunk_metadata(self):
        text = "This is a test sentence that should be split into multiple chunks by the splitter"
        chunks = self.splitter.split(text, source="test.txt")
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.source == "test.txt"
            assert chunk.start_token < chunk.end_token

    def test_overlap(self):
        text = "word " * 100
        chunks = self.splitter.split(text, source="test.txt")
        for i in range(len(chunks) - 1):
            assert chunks[i].end_token - chunks[i + 1].start_token == 2

    def test_overlap_validation(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=10, chunk_overlap=10)
