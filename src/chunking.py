from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass
class Chunk:
    text: str
    chunk_id: int
    source: str
    start_token: int
    end_token: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, source={self.source}, tokens={self.start_token}-{self.end_token}, text=\"{preview}...\")"


class Tokenizer:
    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        if tiktoken is None:
            raise ImportError("tiktoken is required. Install with: pip install tiktoken")
        self.encoding = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 40,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer or Tokenizer()

    def split(self, text: str, source: str) -> List[Chunk]:
        tokens = self.tokenizer.encode(text)
        chunks: List[Chunk] = []
        chunk_id = 0
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source=source,
                    start_token=start,
                    end_token=end,
                )
            )
            chunk_id += 1

            if end >= len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap

        return chunks


class SentenceSplitter:
    """Semantic chunking by sentences with token-based size limit."""

    def __init__(
        self,
        max_chunk_size: int = 256,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tokenizer or Tokenizer()

    def split(self, text: str, source: str) -> List[Chunk]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[Chunk] = []
        chunk_id = 0
        current: List[str] = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            tokens = self.tokenizer.encode(sentence)
            if current_size + len(tokens) > self.max_chunk_size and current:
                chunk_text = " ".join(current)
                start = self.tokenizer.encode(" ".join(current[:1]))[0] if current else 0
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        source=source,
                        start_token=chunks[-1].end_token if chunks else 0,
                        end_token=(chunks[-1].end_token if chunks else 0) + current_size,
                    )
                )
                chunk_id += 1
                current = []
                current_size = 0
            current.append(sentence)
            current_size += len(tokens)

        if current:
            chunk_text = " ".join(current)
            start = chunks[-1].end_token if chunks else 0
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source=source,
                    start_token=start,
                    end_token=start + current_size,
                )
            )

        return chunks


class ParagraphSplitter:
    """Paragraph-aware chunking with token-based size limit."""

    def __init__(
        self,
        max_chunk_size: int = 256,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tokenizer or Tokenizer()

    def split(self, text: str, source: str) -> List[Chunk]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks: List[Chunk] = []
        chunk_id = 0
        current: List[str] = []
        current_size = 0

        for para in paragraphs:
            tokens = self.tokenizer.encode(para)
            if current_size + len(tokens) > self.max_chunk_size and current:
                chunk_text = "\n\n".join(current)
                start = chunks[-1].end_token if chunks else 0
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        source=source,
                        start_token=start,
                        end_token=start + current_size,
                    )
                )
                chunk_id += 1
                current = []
                current_size = 0
            current.append(para)
            current_size += len(tokens)

        if current:
            chunk_text = "\n\n".join(current)
            start = chunks[-1].end_token if chunks else 0
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source=source,
                    start_token=start,
                    end_token=start + current_size,
                )
            )

        return chunks


class SectionSplitter:
    """Metadata-aware chunking that splits on markdown-style headers."""

    def __init__(
        self,
        max_chunk_size: int = 256,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tokenizer or Tokenizer()

    def split(self, text: str, source: str) -> List[Chunk]:
        lines = text.splitlines()
        sections: List[tuple[str, str]] = []
        current_header = ""
        current_body: List[str] = []

        for line in lines:
            if line.startswith('#'):
                if current_body:
                    sections.append((current_header, "\n".join(current_body)))
                current_header = line.strip()
                current_body = []
            else:
                current_body.append(line)

        if current_body:
            sections.append((current_header, "\n".join(current_body)))

        chunks: List[Chunk] = []
        chunk_id = 0
        token_offset = 0

        for header, body in sections:
            tokens = self.tokenizer.encode(body)
            start = 0
            while start < len(tokens):
                end = min(start + self.max_chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                if header:
                    chunk_text = f"{header}\n{chunk_text}"

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        source=source,
                        start_token=token_offset + start,
                        end_token=token_offset + end,
                        metadata={"section": header},
                    )
                )
                chunk_id += 1
                if end >= len(tokens):
                    break
                start += self.max_chunk_size

            token_offset += len(tokens)

        return chunks if chunks else [
            Chunk(
                text=text,
                chunk_id=0,
                source=source,
                start_token=0,
                end_token=len(self.tokenizer.encode(text)),
            )
        ]


class MultiStrategyChunker:
    """Run multiple chunking strategies and merge results."""

    def __init__(self, strategies: Optional[List[str]] = None) -> None:
        self.strategies = strategies or ["fixed", "semantic", "paragraph", "section"]

    def chunk(self, text: str, source: str) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        seen_texts = set()
        next_id = 0

        if "fixed" in self.strategies:
            splitter = TextSplitter()
            for c in splitter.split(text, source):
                if c.text not in seen_texts:
                    all_chunks.append(Chunk(text=c.text, chunk_id=next_id, source=c.source,
                                            start_token=c.start_token, end_token=c.end_token,
                                            metadata={**c.metadata, "strategy": "fixed"}))
                    seen_texts.add(c.text)
                    next_id += 1

        if "semantic" in self.strategies:
            splitter = SentenceSplitter()
            for c in splitter.split(text, source):
                if c.text not in seen_texts:
                    all_chunks.append(Chunk(text=c.text, chunk_id=next_id, source=c.source,
                                            start_token=c.start_token, end_token=c.end_token,
                                            metadata={**c.metadata, "strategy": "semantic"}))
                    seen_texts.add(c.text)
                    next_id += 1

        if "paragraph" in self.strategies:
            splitter = ParagraphSplitter()
            for c in splitter.split(text, source):
                if c.text not in seen_texts:
                    all_chunks.append(Chunk(text=c.text, chunk_id=next_id, source=c.source,
                                            start_token=c.start_token, end_token=c.end_token,
                                            metadata={**c.metadata, "strategy": "paragraph"}))
                    seen_texts.add(c.text)
                    next_id += 1

        if "section" in self.strategies:
            splitter = SectionSplitter()
            for c in splitter.split(text, source):
                if c.text not in seen_texts:
                    all_chunks.append(Chunk(text=c.text, chunk_id=next_id, source=c.source,
                                            start_token=c.start_token, end_token=c.end_token,
                                            metadata={**c.metadata, "strategy": "section"}))
                    seen_texts.add(c.text)
                    next_id += 1

        return all_chunks

