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
