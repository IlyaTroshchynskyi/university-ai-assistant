"""Turns enriched :class:`DocElement`s into retrieval-ready :class:`Chunk`s.

Tables and images are emitted as single atomic chunks (never split). Consecutive
text elements sharing a section are joined and only split when they exceed the
size budget, with the section heading kept as standalone context on each chunk."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ai_assistant.university_knowladge.structured.schemas import Chunk, DocElement


class ElementChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, elements: list[DocElement], source: str) -> list[Chunk]:
        # A chunk takes its metadata from the first element of the group it came from.
        texts = [(text, group[0]) for group in self._group_elements(elements) for text in self._group_texts(group)]
        return [
            Chunk(
                text=text,
                element_type=element.kind,
                page=element.page,
                section=element.section,
                source=source,
                index=index,
            )
            for index, (text, element) in enumerate(texts)
        ]

    def _group_elements(self, elements: list[DocElement]) -> list[list[DocElement]]:
        """Split the document into groups that chunk as a unit: a run of text sharing
        one section, or a single table/image."""
        groups: list[list[DocElement]] = []
        for element in elements:
            if groups and self._continues_text(groups[-1][-1], element):
                groups[-1].append(element)
            else:
                groups.append([element])
        return groups

    def _group_texts(self, group: list[DocElement]) -> list[str]:
        """The chunk texts one group contributes: one for a table or an image, one or
        more for text."""
        if group[0].kind == 'text':
            return self._split_text_with_heading(group)
        return [self._enriched_text(group[0])]

    def _split_text_with_heading(self, group: list[DocElement]) -> list[str]:
        """The group's text joined, split to the size budget, and each piece prefixed
        with the section heading so a chunk retrieved alone still carries its context."""
        section = group[0].section
        body = '\n\n'.join(element.raw for element in group)
        pieces = self._splitter.split_text(body)
        if not section:
            return pieces
        return [f'{section}\n\n{piece}' for piece in pieces]

    @staticmethod
    def _continues_text(previous: DocElement, element: DocElement) -> bool:
        """Whether ``element`` keeps the same run of text going. Text merges into the
        element before it only while the section holds; a table, an image or a new section
        starts a group of its own."""
        return previous.kind == 'text' and element.kind == 'text' and previous.section == element.section

    @staticmethod
    def _enriched_text(element: DocElement) -> str:
        """A table's or image's enriched text, falling back to the raw form when enrichment
        is disabled or failed."""
        return element.text if element.text is not None else element.raw
