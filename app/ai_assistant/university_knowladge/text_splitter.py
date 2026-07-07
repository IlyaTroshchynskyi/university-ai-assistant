"""Text splitting behind a small interface so the chunking strategy is a swappable dependency.
Today only a recursive character splitter exists; a semantic or model-based splitter can later
implement the same ``TextSplitter`` interface without touching its consumers."""

from abc import ABC, abstractmethod

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter(ABC):
    """Splits a document's text into chunks ready for embedding."""

    @abstractmethod
    def split(self, text: str) -> list[str]: ...


class RecursiveTextSplitter(TextSplitter):
    """Recursive character splitter (wraps langchain-text-splitters): tries to split on
    paragraph, then line, then sentence, then word boundaries so chunks stay coherent."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, text: str) -> list[str]:
        return self._splitter.split_text(text)
