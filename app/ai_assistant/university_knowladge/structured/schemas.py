from dataclasses import dataclass
from typing import Literal

ElementType = Literal['text', 'table', 'image']


@dataclass(slots=True)
class DocElement:
    """One ordered piece of a document. ``raw`` holds the source form (plain Markdown
    text, a Markdown table, or an image file path); ``text`` is the final,
    possibly enriched, text used for the chunk (filled by enrichment for tables and
    images; equal to ``raw`` for text elements)."""

    kind: ElementType
    page: int
    section: str | None
    raw: str
    text: str | None = None


@dataclass(slots=True)
class Chunk:
    """A retrieval-ready unit with the metadata retrieval needs to filter and cite."""

    text: str
    element_type: ElementType
    page: int
    section: str | None
    source: str
    index: int
